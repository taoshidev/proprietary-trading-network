import time
from typing import List, Tuple, Dict

import numpy as np

from data_generator.tiingo_data_service import TiingoDataService
from data_generator.polygon_data_service import PolygonDataService
from time_util.time_util import TimeUtil

from vali_objects.vali_config import TradePair
from vali_objects.position import Position
from vali_objects.utils.vali_utils import ValiUtils
from shared_objects.retry import retry
import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource
from statistics import median


class LivePriceFetcher:
    def __init__(self, secrets, disable_ws=False):
        if "tiingo_apikey" in secrets:
            self.tiingo_data_service = TiingoDataService(api_key=secrets["tiingo_apikey"], disable_ws=disable_ws)
        else:
            raise Exception("Tiingo API key not found in secrets.json")
        if "polygon_apikey" in secrets:
            self.polygon_data_service = PolygonDataService(api_key=secrets["polygon_apikey"], disable_ws=disable_ws)
        else:
            raise Exception("Polygon API key not found in secrets.json")

    def stop_all_threads(self):
        if self.tiingo_data_service.websocket_manager_thread:
            self.tiingo_data_service.websocket_manager_thread.join()
        self.tiingo_data_service.stop_threads()

        if self.polygon_data_service.websocket_manager_thread:
            self.polygon_data_service.websocket_manager_thread.join()
        self.polygon_data_service.close_websockets()
        self.polygon_data_service.stop_threads()

    def determine_best_price(self, price_events: List[PriceSource | None], current_time_ms: int,
                             filter_recent_only=True) -> Tuple:
        """
        Determines the best price from a list of price events based on their recency and validity.
        """
        valid_events = [event for event in price_events if event]
        if not valid_events:
            return None, None

        best_event = PriceSource.get_winning_event(valid_events, current_time_ms)
        if not best_event:
            return None, None

        if filter_recent_only and best_event.time_delta_from_now_ms(current_time_ms) > 2000:
            return None, None

        return best_event.parse_best_price(current_time_ms), PriceSource.non_null_events_sorted(valid_events, current_time_ms)

    def fetch_prices(self, tps: List[TradePair], trade_pair_to_last_order_time_ms, ws_only=False) -> (
            dict[str: Tuple[float, List[PriceSource]]] | dict[str: Tuple[None, None]]):
        """
        Fetches data using WebSockets first; uses REST APIs if WebSocket data is outdated or missing.
        """
        websocket_prices_polygon = self.polygon_data_service.get_closes_websocket(trade_pairs=tps, trade_pair_to_last_order_time_ms=trade_pair_to_last_order_time_ms)
        websocket_prices_tiingo_data = self.tiingo_data_service.get_closes_websocket(trade_pairs=tps, trade_pair_to_last_order_time_ms=trade_pair_to_last_order_time_ms)
        trade_pairs_needing_rest_data = []

        results = {}

        # Initial check using WebSocket data
        for trade_pair in tps:
            current_time_ms = trade_pair_to_last_order_time_ms[trade_pair]
            events = [websocket_prices_polygon.get(trade_pair), websocket_prices_tiingo_data.get(trade_pair)]
            price, sources = self.determine_best_price(events, current_time_ms)
            if price:
                results[trade_pair] = (price, sources)
            else:
                trade_pairs_needing_rest_data.append(trade_pair)

        # Fetch from REST APIs if needed
        if not trade_pairs_needing_rest_data or ws_only:
            return results

        rest_prices_polygon = self.polygon_data_service.get_closes_rest(trade_pairs_needing_rest_data)
        rest_prices_tiingo_data = self.tiingo_data_service.get_closes_rest(trade_pairs_needing_rest_data)

        for trade_pair in trade_pairs_needing_rest_data:
            current_time_ms = trade_pair_to_last_order_time_ms[trade_pair]
            events = [
                websocket_prices_polygon.get(trade_pair),
                websocket_prices_tiingo_data.get(trade_pair),
                rest_prices_polygon.get(trade_pair),
                rest_prices_tiingo_data.get(trade_pair)
            ]
            results[trade_pair] = self.determine_best_price(events, current_time_ms, filter_recent_only=False)

        return results

    def get_ws_price_sources_in_window(self, trade_pair: TradePair, start_ms: int, end_ms: int) -> List[PriceSource]:
        # Utilize get_events_in_range
        poly_sources = self.polygon_data_service.trade_pair_to_recent_events[trade_pair.trade_pair].get_events_in_range(start_ms, end_ms)
        t_sources = self.tiingo_data_service.trade_pair_to_recent_events[trade_pair.trade_pair].get_events_in_range(start_ms, end_ms)
        return poly_sources + t_sources

    @retry(tries=2, delay=5, backoff=2)
    def get_latest_price(self, trade_pair: TradePair, time_ms=None) -> Tuple[float, List[PriceSource]] | Tuple[
        None, None]:
        """
        Gets the latest price for a single trade pair by utilizing WebSocket and possibly REST data sources.
        Tries to get the price as close to time_ms as possible.
        """
        if not time_ms:
            time_ms = TimeUtil.now_in_millis()
        return self.fetch_prices([trade_pair], {trade_pair: time_ms})[trade_pair]

    @retry(tries=2, delay=5, backoff=2)
    def get_latest_prices(self, trade_pairs: List[TradePair],
                          trade_pair_to_last_order_time_ms: Dict[TradePair, int] = None) -> Dict:
        """
        Retrieves the latest prices for multiple trade pairs, leveraging both WebSocket and REST APIs as needed.
        """
        if not trade_pair_to_last_order_time_ms:
            current_time_ms = TimeUtil.now_in_millis()
            trade_pair_to_last_order_time_ms = {tp: current_time_ms for tp in trade_pairs}
        return self.fetch_prices(trade_pairs, trade_pair_to_last_order_time_ms)

    def time_since_last_ws_ping_s(self, trade_pair: TradePair) -> float | None:
        if trade_pair in self.polygon_data_service.UNSUPPORTED_TRADE_PAIRS:
            return None
        now_ms = TimeUtil.now_in_millis()
        t1 = self.polygon_data_service.get_websocket_lag_for_trade_pair_s(tp=trade_pair.trade_pair, now_ms=now_ms)
        t2 = self.tiingo_data_service.get_websocket_lag_for_trade_pair_s(tp=trade_pair.trade_pair, now_ms=now_ms)
        return max([x for x in (t1, t2) if x])

    def filter_outliers(self, unique_data: List[PriceSource]) -> List[PriceSource]:
        """
        Filters out outliers and duplicates from a list of price sources.
        """
        if not unique_data:
            return []

        # Function to calculate bounds
        def calculate_bounds(prices):
            median = np.median(prices)
            # Calculate bounds as 5% less than and more than the median
            lower_bound = median * 0.95
            upper_bound = median * 1.05
            return lower_bound, upper_bound

        # Calculate bounds for each price type
        close_prices = np.array([x.close for x in unique_data])
        # high_prices = np.array([x.high for x in unique_data])
        # low_prices = np.array([x.low for x in unique_data])

        close_lower_bound, close_upper_bound = calculate_bounds(close_prices)
        # high_lower_bound, high_upper_bound = calculate_bounds(high_prices)
        # low_lower_bound, low_upper_bound = calculate_bounds(low_prices)

        # Filter data by checking all price points against their respective bounds
        filtered_data = [x for x in unique_data if close_lower_bound <= x.close <= close_upper_bound]
        # filtered_data = [x for x in unique_data if close_lower_bound <= x.close <= close_upper_bound and
        #                 high_lower_bound <= x.high <= high_upper_bound and
        #                 low_lower_bound <= x.low <= low_upper_bound]

        # Sort the data by timestamp in ascending order
        filtered_data.sort(key=lambda x: x.start_ms, reverse=True)
        return filtered_data

    def parse_price_from_candle_data(self, data: List[PriceSource], trade_pair: TradePair) -> float | None:
        if not data or len(data) == 0:
            # Market is closed for this trade pair
            bt.logging.trace(f"No ps data to parse for realtime price for trade pair {trade_pair.trade_pair_id}. data: {data}")
            return None

        # Data by timestamp in ascending order so that the largest timestamp is first
        return data[0].close

    def parse_extreme_price_in_window(self, candle_data: Dict[TradePair, List[PriceSource]], open_position: Position, parse_min: bool = True) -> Tuple[float, PriceSource] | Tuple[None, None]:
        trade_pair = open_position.trade_pair
        dat = candle_data.get(trade_pair)
        if dat is None:
            # Market is closed for this trade pair
            return None, None

        min_allowed_timestamp_ms = open_position.orders[-1].processed_ms
        prices = []
        corresponding_sources = []

        for a in dat:
            if a.end_ms < min_allowed_timestamp_ms:
                continue
            price = a.low if parse_min else a.high
            if price is not None:
                prices.append(price)
                corresponding_sources.append(a)

        if not prices:
            return None, None

        if len(prices) % 2 == 1:
            med_price = median(prices)  # Direct median if the list is odd
        else:
            # If even, choose the lower middle element to ensure it exists in the list
            sorted_prices = sorted(prices)
            middle_index = len(sorted_prices) // 2 - 1
            med_price = sorted_prices[middle_index]

        med_index = prices.index(med_price)
        med_source = corresponding_sources[med_index]

        return med_price, med_source

    def get_candles(self, trade_pairs, start_time_ms, end_time_ms) -> dict:
        ans = {}
        debug = {}
        one_second_rest_candles = self.polygon_data_service.get_candles(
            trade_pairs=trade_pairs, start_time_ms=start_time_ms, end_time_ms=end_time_ms)

        for tp in trade_pairs:
            rest_candles = one_second_rest_candles.get(tp, [])
            ws_candles = self.get_ws_price_sources_in_window(tp, start_time_ms, end_time_ms)
            non_null_sources = list(set(rest_candles + ws_candles))
            filtered_sources = self.filter_outliers(non_null_sources)
            # Get the sources removed to debug
            removed_sources = [x for x in non_null_sources if x not in filtered_sources]
            ans[tp] = filtered_sources
            min_time = min((x.start_ms for x in non_null_sources)) if non_null_sources else 0
            max_time = max((x.end_ms for x in non_null_sources)) if non_null_sources else 0
            debug[
                tp.trade_pair] = f"R{len(rest_candles)}W{len(ws_candles)}U{len(non_null_sources)}T[{(max_time - min_time) / 1000.0:.2f}]"
            if removed_sources:
                mi = min((x.close for x in non_null_sources))
                ma = max((x.close for x in non_null_sources))
                debug[tp.trade_pair] += f" Removed {[x.close for x in removed_sources]} Original min/max {mi}/{ma}"

        bt.logging.info(f"Fetched candles {debug} in window"
                        f" {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
                        f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)}")

        # If Polygon has any missing keys, it is intentional and corresponds to a closed market. We don't want to use twelvedata for this TODO: fall back to live price from TD/POLY.
        # if self.twelvedata_available and len(ans) == 0:
        #    bt.logging.info(f"Fetching candles from TD for {[x.trade_pair for x in trade_pairs]} from {start_time_ms} to {end_time_ms}")
        #    closes = self.twelve_data.get_closes(trade_pairs=trade_pairs)
        #    ans.update(closes)
        return ans

    def get_close_at_date(self, trade_pair, timestamp_ms):
        price, time_delta = self.polygon_data_service.get_close_at_date_second(trade_pair=trade_pair, target_timestamp_ms=timestamp_ms)
        if price is None:
            price, time_delta = self.tiingo_data_service.get_close_at_date(trade_pair=trade_pair, timestamp_ms=timestamp_ms)
            if price is not None:
                bt.logging.warning(
                    f"Fell back to Tiingo get_date for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, ms: {timestamp_ms}")

        if price is None:
            price, time_delta = self.polygon_data_service.get_close_at_date_minute_fallback(trade_pair=trade_pair,
                                                                               timestamp_ms=timestamp_ms)
            if price:
                bt.logging.warning(
                    f"Fell back to Polygon get_date_minute_fallback for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, ms: {timestamp_ms}")
        """
        if price is None:
            price, time_delta = self.polygon_data_service.get_close_in_past_hour_fallback(trade_pair=trade_pair,
                                                                             timestamp_ms=timestamp_ms)
            if price:
                formatted_date = TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)
                bt.logging.warning(
                    f"Fell back to Polygon get_close_in_past_hour_fallback for price of {trade_pair.trade_pair} at {formatted_date}, ms: {timestamp_ms}")
        if price is None:
            formatted_date = TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)
            bt.logging.error(
                f"Failed to get data at ET date {formatted_date} for {trade_pair.trade_pair}. Timestamp ms: {timestamp_ms}."
                f" Ask a team member to investigate this issue.")
        """
        return price, time_delta


if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets)
    time.sleep(100000)
    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, ]
    while True:
        for tp in TradePair:
            print(f"{tp.trade_pair}: {live_price_fetcher.get_close(tp)}")
        time.sleep(10)
    # ans = live_price_fetcher.get_closes(trade_pairs)
    # for k, v in ans.items():
    #    print(f"{k.trade_pair_id}: {v}")
    # print("Done")
