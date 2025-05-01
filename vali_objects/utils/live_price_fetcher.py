import asyncio
import time
from typing import List, Tuple, Dict

import numpy as np

from data_generator.tiingo_data_service import TiingoDataService
from data_generator.polygon_data_service import PolygonDataService
from time_util.time_util import TimeUtil, timeme

from vali_objects.vali_config import TradePair
from vali_objects.position import Position
from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout


from vali_objects.vali_dataclasses.price_source import PriceSource
from statistics import median

class LivePriceFetcher:
    def __init__(self, secrets, disable_ws=False, ipc_manager=None, is_backtesting=False):
        self.is_backtesting = is_backtesting
        if "tiingo_apikey" in secrets:
            self.tiingo_data_service = TiingoDataService(api_key=secrets["tiingo_apikey"], disable_ws=disable_ws,
                                                         ipc_manager=ipc_manager)
        else:
            raise Exception("Tiingo API key not found in secrets.json")
        if "polygon_apikey" in secrets:
            self.polygon_data_service = PolygonDataService(api_key=secrets["polygon_apikey"], disable_ws=disable_ws,
                                                           ipc_manager=ipc_manager, is_backtesting=is_backtesting)
        else:
            raise Exception("Polygon API key not found in secrets.json")

    def cleanup_async_io(self):
        self.tiingo_data_service.stop()
        self.polygon_data_service.stop()

    def sorted_valid_price_sources(self, price_events: List[PriceSource | None], current_time_ms: int, filter_recent_only=True) -> List[PriceSource] | None:
        """
        Sorts a list of price events by their recency and validity.
        """
        valid_events = [event for event in price_events if event]
        if not valid_events:
            return None

        best_event = PriceSource.get_winning_event(valid_events, current_time_ms)
        if not best_event:
            return None

        if filter_recent_only and best_event.time_delta_from_now_ms(current_time_ms) > 3000:
            return None

        return PriceSource.non_null_events_sorted(valid_events, current_time_ms)

    def get_ws_price_sources_in_window(self, trade_pair: TradePair, start_ms: int, end_ms: int) -> List[PriceSource]:
        # Utilize get_events_in_range
        poly_sources = self.polygon_data_service.trade_pair_to_recent_events[trade_pair.trade_pair].get_events_in_range(start_ms, end_ms)
        t_sources = self.tiingo_data_service.trade_pair_to_recent_events[trade_pair.trade_pair].get_events_in_range(start_ms, end_ms)
        return poly_sources + t_sources

    @timeme
    def get_latest_price(self, trade_pair: TradePair, time_ms=None) -> Tuple[float, List[PriceSource]] | Tuple[None, None]:
        """
        Gets the latest price for a single trade pair by utilizing WebSocket and possibly REST data sources.
        Tries to get the price as close to time_ms as possible.
        """
        if not time_ms:
            time_ms = TimeUtil.now_in_millis()
        price_sources = self.get_sorted_price_sources_for_trade_pair(trade_pair, time_ms)
        winning_event = PriceSource.get_winning_event(price_sources, time_ms)
        return winning_event.parse_best_best_price_legacy(time_ms), price_sources

    def get_sorted_price_sources_for_trade_pair(self, trade_pair: TradePair, time_ms:int) -> List[PriceSource] | None:
        temp = self.get_tp_to_sorted_price_sources([trade_pair], {trade_pair: time_ms})
        return temp.get(trade_pair)

    @timeme
    def get_tp_to_sorted_price_sources(self, trade_pairs: List[TradePair],
                                       trade_pair_to_last_order_time_ms: Dict[TradePair, int] = None) -> Dict[TradePair, List[PriceSource]]:
        """
        Retrieves the latest prices for multiple trade pairs, leveraging both WebSocket and REST APIs as needed.
        """
        if not trade_pair_to_last_order_time_ms:
            current_time_ms = TimeUtil.now_in_millis()
            trade_pair_to_last_order_time_ms = {tp: current_time_ms for tp in trade_pairs}
        websocket_prices_polygon = self.polygon_data_service.get_closes_websocket(trade_pairs=trade_pairs,
                                                                                  trade_pair_to_last_order_time_ms=trade_pair_to_last_order_time_ms)
        websocket_prices_tiingo_data = self.tiingo_data_service.get_closes_websocket(trade_pairs=trade_pairs,
                                                                                     trade_pair_to_last_order_time_ms=trade_pair_to_last_order_time_ms)
        trade_pairs_needing_rest_data = []

        results = {}

        # Initial check using WebSocket data
        for trade_pair in trade_pairs:
            current_time_ms = trade_pair_to_last_order_time_ms[trade_pair]
            events = [websocket_prices_polygon.get(trade_pair), websocket_prices_tiingo_data.get(trade_pair)]
            sources = self.sorted_valid_price_sources(events, current_time_ms, filter_recent_only=True)
            if sources:
                results[trade_pair] = sources
            else:
                trade_pairs_needing_rest_data.append(trade_pair)

        # Fetch from REST APIs if needed
        if not trade_pairs_needing_rest_data:
            return results

        rest_prices_polygon, rest_prices_tiingo_data = self.dual_rest_get(trade_pairs_needing_rest_data)

        for trade_pair in trade_pairs_needing_rest_data:
            current_time_ms = trade_pair_to_last_order_time_ms[trade_pair]
            sources = self.sorted_valid_price_sources([
                websocket_prices_polygon.get(trade_pair),
                websocket_prices_tiingo_data.get(trade_pair),
                rest_prices_polygon.get(trade_pair),
                rest_prices_tiingo_data.get(trade_pair)
            ], current_time_ms, filter_recent_only=False)
            results[trade_pair] = sources

        return results

    def dual_rest_get(
            self,
            trade_pairs: List[TradePair]
    ) -> Tuple[Dict[TradePair, PriceSource], Dict[TradePair, PriceSource]]:
        """
        Fetch REST closes from both Polygon and Tiingo in parallel,
        on a dedicated thread+event-loop (no impact to your main loop).
        """

        def _worker() -> Tuple[Dict[TradePair, PriceSource], Dict[TradePair, PriceSource]]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(asyncio.gather(
                    self.polygon_data_service.get_closes_rest(trade_pairs),
                    self.tiingo_data_service.get_closes_rest(trade_pairs),
                ))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut = executor.submit(_worker)
            try:
                polygon_results, tiingo_results = fut.result(timeout=10)
            except FutureTimeout:
                fut.cancel()
                raise TimeoutError("REST API requests timed out")

        return polygon_results, tiingo_results

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
            m = np.median(prices)
            # Calculate bounds as 5% less than and more than the median
            lower_bound = m * 0.95
            upper_bound = m * 1.05
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

    def get_quote(self, trade_pair: TradePair, processed_ms: int) -> (float, float, int):
        """
        returns the bid and ask quote for a trade_pair at processed_ms. Only Polygon supports point-in-time bid/ask.
        """
        return self.polygon_data_service.get_quote(trade_pair, processed_ms)

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



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets, disable_ws=False)
    time.sleep(100000)
    assert 0
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
