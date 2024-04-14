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
    def get_closes(self, trade_pairs: list, websocket_only=False):
        ans = {}
        if self.polygon_available:
            ans = self.polygon_data_provider.get_closes(trade_pairs=trade_pairs, websocket_only=websocket_only)
            invalid_closes = {k: v for k, v in ans.items() if v is None}
            if ans and len(invalid_closes) == 0:
                return ans
        if self.twelvedata_available:
            td_closes = self.twelve_data.get_closes(trade_pairs=list(set(trade_pairs) - set(ans.keys())), websocket_only=websocket_only)
            ans.update(td_closes)
        return ans

    def time_since_last_ping_s(self, trade_pair: TradePair) -> float | None:
        if self.polygon_available:
            return self.polygon_data_provider.get_websocket_lag_for_trade_pair_s(trade_pair=trade_pair)
        # Don't want to use twelvedata for this
        return None

    def get_candles(self, trade_pairs, start_time_ms, end_time_ms, fallback_to_live_price=True) -> dict:
        delta_time_ms = end_time_ms - start_time_ms
        delta_time_seconds = delta_time_ms / 1000
        delta_time_minutes = delta_time_seconds / 60
        delta_time_hours = delta_time_minutes / 60

        if delta_time_seconds < 70:
            timespan = "second"
        elif delta_time_minutes < 70:
            timespan = "minute"
        elif delta_time_hours < 70:
            timespan = "hour"
        else:
            timespan = "day"

        ans = {}
        if self.polygon_available:
            ans = self.polygon_data_provider.get_candles(trade_pairs=trade_pairs, start_time_ms=start_time_ms, end_time_ms=end_time_ms, timespan=timespan, strict_time_bounds=True)
            debug = {}
            no_candles = []
            for k, v in ans.items():
                if v and isinstance(v, list):
                    debug[k.trade_pair] = '[' + str(len(v)) + ']'
                else:
                    debug[k.trade_pair] = v
                    no_candles.append(k)

            if fallback_to_live_price:
                live_prices = self.get_closes(no_candles, websocket_only=True)
                debug.update(live_prices)
                ans.update(live_prices)

            bt.logging.info(f"Fetched candles/live_price from Polygon for {debug} from"
                            f" {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
                            f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)} timespan {timespan}")



        if (timespan == "minute" and self.twelvedata_available and not fallback_to_live_price):
            # Get TD candles
            ans2 = self.twelve_data.get_candles_at_timespan(trade_pairs=no_candles, start_time_ms=start_time_ms, end_time_ms=end_time_ms, timespan='1min')
            ans.update(ans2)
        elif (timespan == "hour" and self.twelvedata_available and not fallback_to_live_price):
            # Get TD candles
            ans2 = self.twelve_data.get_candles_at_timespan(trade_pairs=no_candles, start_time_ms=start_time_ms, end_time_ms=end_time_ms, timespan='1h')
            ans.update(ans2)
        elif (timespan == "day" and self.twelvedata_available and not fallback_to_live_price):
            # Get TD candles
            ans2 = self.twelve_data.get_candles_at_timespan(trade_pairs=no_candles, start_time_ms=start_time_ms, end_time_ms=end_time_ms, timespan='1day')
            ans.update(ans2)

        # If Polygon has any missing keys, it is intentional and corresponds to a closed market. We don't want to use twelvedata for this
        # Also TD candles are only 1 min granularity
        if (fallback_to_live_price and self.twelvedata_available and len(ans) == 0):
            bt.logging.info(f"Fetching live prices from TD {[x.trade_pair for x in trade_pairs]} from {start_time_ms} to {end_time_ms}")
            closes = self.twelve_data.get_closes(trade_pairs=trade_pairs)
            ans.update(closes)
        return ans

    def get_close_at_date(self, trade_pair, timestamp_ms):
        ans = self.polygon_data_provider.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms, timespan='second')
        if ans is None:
            ans = self.twelve_data.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
            if ans is not None:
                bt.logging.warning(f"Fell back to TwelveData get_date for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, ms: {timestamp_ms}")

        if ans is None:
            ans = self.polygon_data_provider.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms, timespan='minute')
            if ans:
                bt.logging.warning(f"Fell back to Polygon get_date_minute_fallback for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, ms: {timestamp_ms}")
        if ans is None:
            ans = self.polygon_data_provider.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms, timespan='hour')
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


    @staticmethod
    def parse_price_from_closing_prices(signal_closing_prices, trade_pair):
        if trade_pair not in signal_closing_prices:
            raise ValueError(f"Trade pair [{trade_pair}] not in closing prices. Closing price keys: {signal_closing_prices.keys()}")

        dat = signal_closing_prices[trade_pair]
        if dat is None:
            # Market is closed for this trade pair
            return None
        if isinstance(dat, float) or isinstance(dat, int):
            # From TwelveData
            return float(dat)

        # Get the newest price in the window
        price = None
        for a in dat:
            #bt.logging.info(f"in _parse_price_from_closing_prices. timestamp: {a.timestamp}, close: {a.close}")
            if a.vwap is not None:
                price = float(a.vwap)

        #bt.logging.info(f"in _parse_price_from_closing_prices. price: {price}. trade pair {trade_pair.trade_pair_id}")
        return price

    @staticmethod
    def parse_extreme_price_in_window(signal_closing_prices, open_position, parse_min=True, filter_min_timestamp=True):
        trade_pair = open_position.trade_pair
        dat = signal_closing_prices[trade_pair]
        if dat is None:
            # Market is closed for this trade pair
            return None
        if isinstance(dat, float) or isinstance(dat, int):
            # From TwelveData
            return float(dat)

        price = None
        for a in dat:
            timestamp_ms = a['timestamp'] if isinstance(a, dict) else a.timestamp
            low = float(a['low']) if isinstance(a, dict) else a.low
            high = float(a['high']) if isinstance(a, dict) else a.high
            open = float(a['open']) if isinstance(a, dict) else a.open
            close = float(a['close']) if isinstance(a, dict) else a.close
            vwap = (open + close) / 2.0 if isinstance(a, dict) else a.vwap
            candle_epoch_ms = timestamp_ms
            # Handle the case where an order gets placed in between MDD checks.
            if filter_min_timestamp and candle_epoch_ms < open_position.orders[-1].processed_ms:
                continue
            #bt.logging.info(f"in _parse_min_price_in_window. timestamp: {a.timestamp}, close: {a.close}")
            if parse_min:
                if vwap is not None:
                    price = vwap if price is None else min(price, vwap)
            else:
                if vwap is not None:
                    price = vwap if price is None else max(price, vwap)
        #print(f"in _parse_min_price_in_window min_price: {min_price}. trade_pair {trade_pair.trade_pair_id}")
        return float(price) if price else None



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets)
    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD]
    ans = live_price_fetcher.get_closes(trade_pairs)
    for k, v in ans.items():
        print(f"{k.trade_pair_id}: {v}")
    print("Done")

