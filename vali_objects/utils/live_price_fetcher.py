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
            trade_pairs_no_candles = [k for k in trade_pairs if k not in ans or not ans[k]]
            live_prices = self.polygon_data_provider.get_closes(trade_pairs=trade_pairs_no_candles, websocket_only=True)
            ans.update(live_prices)
            if isinstance(ans, dict) and len(ans) > 0:
                debug = {k.trade_pair: '[' + str(len(v)) + ']' if v and isinstance(v, list) else v for k, v in ans.items() }
            bt.logging.info(f"Fetched candles from Polygon for {debug} from"
                            f" {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
                            f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)}")

        # Figure out which trade pairs are miss
        missing_trade_pairs = [k for k in trade_pairs if k not in ans or not ans[k]]
        # Use TD websocket to fill remaining gas. No rest candles.
        if self.twelvedata_available and missing_trade_pairs:
            bt.logging.info(f"Fetching TD websocket data for {missing_trade_pairs}")
            live_prices = self.twelve_data.get_closes(trade_pairs=missing_trade_pairs)
            #closes = self.twelve_data.get_closes(trade_pairs=trade_pairs)
            ans.update(live_prices)
        return ans

    def get_close_at_date(self, trade_pair, timestamp_ms):
        ans = self.polygon_data_provider.get_close_at_date_second(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
        if ans is None:
            ans = self.twelve_data.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
            if ans is not None:
                bt.logging.warning(f"Fell back to TwelveData get_date for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, ms: {timestamp_ms}")

        if ans is None:
            ans = self.polygon_data_provider.get_close_at_date_minute_fallback(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
            if ans:
                bt.logging.warning(f"Fell back to Polygon get_date_minute_fallback for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, ms: {timestamp_ms}")
        if ans is None:
            ans = self.polygon_data_provider.get_close_in_past_hour_fallback(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
            if ans:
                formatted_date = TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)
                bt.logging.warning(f"Fell back to Polygon get_close_in_past_hour_fallback for price of {trade_pair.trade_pair} at {formatted_date}, ms: {timestamp_ms}")
        if ans is None:
            formatted_date = TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)
            bt.logging.error(
                f"Failed to get data at ET date {formatted_date} for {trade_pair.trade_pair}. Timestamp ms: {timestamp_ms}."
                f" Ask a team member to investigate this issue.")

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

