from data_generator.twelvedata_service import TwelveDataService
from data_generator.polygon_data_service import PolygonDataService
from time_util.time_util import TimeUtil

from vali_config import TradePair
from vali_objects.utils.vali_utils import ValiUtils
from shared_objects.retry import retry
import bittensor as bt

class LivePriceFetcher():
    def __init__(self, secrets):
        if "twelvedata_apikey" in secrets:
            self.twelve_data = TwelveDataService(api_key=secrets["twelvedata_apikey"])
        else:
            self.twelve_data = None
        if "polygon_apikey" in secrets:
            self.polygon_data_provider = PolygonDataService(api_key=secrets["polygon_apikey"])
        else:
            self.polygon_data_provider = None

        assert self.twelvedata_available or self.polygon_available, \
            "No data provider available. Make sure your API keys are correctly configured in secrets.json"

    @property
    def twelvedata_available(self):
        return self.twelve_data is not None

    @property
    def polygon_available(self):
        return self.polygon_data_provider is not None

    @retry(tries=2, delay=5, backoff=2)
    def get_close(self, trade_pair):
        if self.polygon_available:
            ans = self.polygon_data_provider.get_close(trade_pair=trade_pair)
            if ans:
                return ans
        if self.twelvedata_available:
            return self.twelve_data.get_close(trade_pair=trade_pair)

    @retry(tries=2, delay=5, backoff=2)
    def get_closes(self, trade_pairs: list):
        ans = {}
        if self.polygon_available:
            ans = self.polygon_data_provider.get_closes(trade_pairs=trade_pairs)
            invalid_closes = {k: v for k, v in ans.items() if v is None}
            if ans and len(invalid_closes) == 0:
                return ans
        if self.twelvedata_available:
            td_closes = self.twelve_data.get_closes(trade_pairs=list(set(trade_pairs) - set(ans.keys())))
            ans.update(td_closes)
        return ans

    def time_since_last_ping_s(self, trade_pair: TradePair) -> float | None:
        if self.polygon_available:
            return self.polygon_data_provider.get_websocket_lag_for_trade_pair_s(trade_pair=trade_pair)
        # Don't want to use twelvedata for this
        return None

    def get_candles(self, trade_pairs, start_time_ms, end_time_ms) -> dict:
        ans = {}
        if self.polygon_available:
            ans = self.polygon_data_provider.get_candles(trade_pairs=trade_pairs, start_time_ms=start_time_ms, end_time_ms=end_time_ms)
            if isinstance(ans, dict) and len(ans) > 0:
                debug = {k.trade_pair: len(v) for k, v in ans.items() if v and isinstance(v, list) and len(v) > 0}
            bt.logging.info(f"Fetched candles from Polygon for {debug} from"
                            f" {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
                            f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)}")
        # If Polygon has any missing keys, it is intentional and corresponds to a closed market. We don't want to use twelvedata for this
        if self.twelvedata_available and len(ans) == 0:
            bt.logging.info(f"Fetching candles from TD for {[x.trade_pair for x in trade_pairs]} from {start_time_ms} to {end_time_ms}")
            closes = self.twelve_data.get_closes(trade_pairs=trade_pairs)
            ans.update(closes)
        return ans

    def get_close_at_date(self, trade_pair, timestamp_ms):
        ans = self.polygon_data_provider.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
        if ans is None:
            ans = self.twelve_data.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
        return ans

    def is_market_closed_for_trade_pair(self, trade_pair):
        return self.twelve_data.trade_pair_market_likely_closed(trade_pair)



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets)
    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD]
    ans = live_price_fetcher.get_closes(trade_pairs)
    for k, v in ans.items():
        print(f"{k.trade_pair_id}: {v}")
    print("Done")

