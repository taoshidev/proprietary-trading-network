import json
import threading
import time
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional
from polygon.websocket import WebSocketClient

import bittensor as bt
from setproctitle import setproctitle

from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_objects.vali_config import TradePair, TradePairCategory
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker
from vali_objects.vali_dataclasses.price_source import PriceSource

POLYGON_PROVIDER_NAME = "Polygon"
TIINGO_PROVIDER_NAME = "Tiingo"

def exception_handler_decorator():
    """
    Decorator to handle exceptions, log them, and return a default value.

    Uses a global logger (bt.logging.error) for logging.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                thread_id = threading.get_native_id()
                bt.logging.error(f"Failed to get {func_name} with error: {e}, type: {type(e).__name__} in thread {thread_id}")
                #bt.logging.error(traceback.format_exc())
                return {}

        return wrapper

    return decorator

class BaseDataService():
    def __init__(self, provider_name, ipc_manager=None):
        self.DEBUG_LOG_INTERVAL_S = 180
        self.MAX_TIME_NO_EVENTS_S = 120

        self.provider_name = provider_name
        self.tpc_to_n_events = {x: 0 for x in TradePairCategory}
        self.n_equity_events_skipped_afterhours = 0
        self.trade_pair_to_price_history = defaultdict(list)
        self.closed_market_prices = {tp: None for tp in TradePair}
        self.latest_websocket_events = {}
        self.using_ipc = ipc_manager is not None
        self.n_flushes = 0
        self.trade_pair_to_recent_events_realtime = defaultdict(RecentEventTracker)
        if ipc_manager is None:
            self.trade_pair_to_recent_events = defaultdict(RecentEventTracker)
        else:
            self.trade_pair_to_recent_events = ipc_manager.dict()
        self.trade_pair_category_to_longest_allowed_lag_s = {TradePairCategory.CRYPTO: 30, TradePairCategory.FOREX: 30,
                                                           TradePairCategory.INDICES: 30, TradePairCategory.EQUITIES: 30}
        self.timespan_to_ms = {'second': 1000, 'minute': 1000 * 60, 'hour': 1000 * 60 * 60, 'day': 1000 * 60 * 60 * 24}

        self.trade_pair_lookup = {pair.trade_pair: pair for pair in TradePair}
        self.trade_pair_to_longest_seen_lag_s = {}
        self.market_calendar = UnifiedMarketCalendar()

        for trade_pair in TradePair:
            assert trade_pair.trade_pair_category in self.trade_pair_category_to_longest_allowed_lag_s, \
                f"Trade pair {trade_pair} has no allowed lag time"

        self.WEBSOCKET_THREADS = {
            TradePairCategory.CRYPTO: Optional[threading.Thread],
            TradePairCategory.FOREX: Optional[threading.Thread],
            TradePairCategory.EQUITIES: Optional[threading.Thread]
        }

        self.WEBSOCKET_OBJECTS = {
            TradePairCategory.CRYPTO: Optional[WebSocketClient],
            TradePairCategory.FOREX: Optional[WebSocketClient],
            TradePairCategory.EQUITIES: Optional[WebSocketClient]
        }

    def get_close_rest(
            self,
            trade_pair: TradePair,
            timestamp_ms: int
    ) -> PriceSource | None:
        pass

    def is_market_open(self, trade_pair: TradePair, time_ms=None) -> bool:
        if time_ms is None:
            time_ms = TimeUtil.now_in_millis()
        return self.market_calendar.is_market_open(trade_pair, time_ms)

    def get_close(self, trade_pair: TradePair) -> PriceSource | None:
        event = self.get_websocket_event(trade_pair)
        if not event:
            bt.logging.info(
                f"Fetching REST close for trade pair {trade_pair.trade_pair} using {self.provider_name} REST: {trade_pair.trade_pair}")
            event = self.get_close_rest(trade_pair)
            bt.logging.info(f"Received {self.provider_name} REST data for {trade_pair.trade_pair}: {event}")
        else:
            bt.logging.info(f"Using {self.provider_name} websocket data for {trade_pair.trade_pair}")
        if not event:
            bt.logging.warning(
                f"Failed to get close for {trade_pair.trade_pair} using {self.provider_name} websocket or REST.")
        return event

    def get_first_trade_pair_in_category(self, tpc: TradePairCategory) -> TradePair:
        # Use generator expression for efficiency
        return next((x for x in TradePair if x.trade_pair_category == tpc), None)

    def check_flush(self):
        t0 = time.time()
        # Flush the recent events to shared memory
        for k in list(self.trade_pair_to_recent_events_realtime.keys()):
            self.trade_pair_to_recent_events[k] = self.trade_pair_to_recent_events_realtime[k]
            self.n_flushes += 1
            if self.n_flushes % 500 == 0:
                t1 = time.time()
                bt.logging.info(
                    f"Flushed recent {self.provider_name} events to shared memory in {t1 - t0:.2f} seconds, n_flushes {self.n_flushes}")

    def websocket_manager(self):
        setproctitle(f"vali_{self.__class__.__name__}")
        bt.logging.enable_info()
        tpc_to_prev_n_events = {x: 0 for x in TradePairCategory}
        last_ws_health_check_s = 0
        last_market_status_update_s = 0
        while True:
            now = time.time()
            if now - last_ws_health_check_s > self.MAX_TIME_NO_EVENTS_S:
                categories_reset_messages = []
                for tpc in TradePairCategory:
                    if tpc == TradePairCategory.INDICES:
                        continue
                    if ((self.tpc_to_n_events[tpc] == tpc_to_prev_n_events[tpc]) and
                            (last_ws_health_check_s == 0 or self.is_market_open(self.get_first_trade_pair_in_category(tpc)))):
                        if last_ws_health_check_s == 0:
                            msg = f"First websocket for {self.provider_name} {tpc.__str__()} created"
                        else:
                            msg = f'Websocket {self.provider_name} {tpc.__str__()} is stale {tpc_to_prev_n_events[tpc]}/{self.tpc_to_n_events[tpc]}'
                        categories_reset_messages.append(msg)
                        self.stop_start_websocket_threads(tpc=tpc)
                last_ws_health_check_s = now

                tpc_to_prev_n_events = deepcopy(self.tpc_to_n_events)
                if categories_reset_messages:
                    bt.logging.warning(
                        f"Restarted websockets for [{categories_reset_messages}]")

            if now - last_market_status_update_s > self.DEBUG_LOG_INTERVAL_S:
                last_market_status_update_s = now
                self.debug_log()

            time.sleep(1)
            if self.using_ipc:
                self.check_flush()
    def close_create_websocket_objects(self, tpc: TradePairCategory = None):
        raise NotImplementedError

    def main_stocks(self):
        raise NotImplementedError

    def main_forex(self):
        raise NotImplementedError

    def main_crypto(self):
        raise NotImplementedError

    def instantiate_not_pickleable_objects(self):
        raise NotImplementedError

    def stop_start_websocket_threads(self, tpc: TradePairCategory = None):
        bt.logging.enable_info()
        if self.provider_name == POLYGON_PROVIDER_NAME:
            self.close_create_websocket_objects(tpc=tpc)

        tpcs = [tpc] if tpc is not None else TradePairCategory
        for tpc in tpcs:
            if tpc == TradePairCategory.INDICES:
                continue
            elif tpc == TradePairCategory.EQUITIES:
                target = self.main_stocks
            elif tpc == TradePairCategory.FOREX:
                target = self.main_forex
            elif tpc == TradePairCategory.CRYPTO:
                target = self.main_crypto
            else:
                raise ValueError(f"Invalid tpc {tpc}")
            old_thread = self.WEBSOCKET_THREADS.get(tpc)
            if isinstance(old_thread, threading.Thread):
                old_thread.join(timeout=5)
                if old_thread.is_alive():
                    bt.logging.warning(f"Thread for {self.provider_name} tpc {tpc} is still alive, not starting new thread")
                    continue

            self.WEBSOCKET_THREADS[tpc] = threading.Thread(target=target, daemon=True)
            self.WEBSOCKET_THREADS[tpc].start()
            if isinstance(old_thread, threading.Thread):
                old_id = old_thread.native_id
                new_id = self.WEBSOCKET_THREADS[tpc].native_id
                print(f'replaced {self.provider_name} thread for tpc {tpc} with id {old_id} with new thread id {new_id}')



    def stop_threads(self, tpc: TradePairCategory = None):
        threads_to_check = self.WEBSOCKET_THREADS if tpc is None else {tpc: self.WEBSOCKET_THREADS[tpc]}
        for k, thread in threads_to_check.items():
            if isinstance(thread, threading.Thread):
                thread_id = thread.native_id
                print(f'joining {self.provider_name} thread for tpc {k}')
                thread.join(timeout=6)
                if thread.is_alive():
                    print(f'Failed to stop {self.provider_name} thread for tpc {k} thread id {thread_id}')
                else:
                    print(f'terminated {self.provider_name} thread for tpc {k} thread id {thread_id}')
            else:
                print(f'No thread to stop for {self.provider_name} tpc {k} thread {thread}')

    def get_closes_websocket(self, trade_pairs: List[TradePair], trade_pair_to_last_order_time_ms) -> dict[str: PriceSource]:
        events = {}
        for trade_pair in trade_pairs:
            symbol = trade_pair.trade_pair
            if symbol not in self.trade_pair_to_recent_events:
                continue

            # Get the closest aligned event
            time_ms = trade_pair_to_last_order_time_ms[trade_pair]
            symbol = trade_pair.trade_pair
            latest_event = self.trade_pair_to_recent_events[symbol].get_closest_event(time_ms)
            events[trade_pair] = latest_event
            """
            event = self.latest_websocket_events[symbol]
            lag_s = self.get_websocket_lag_for_trade_pair_s(symbol)
            is_stale = lag_s > self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
            if is_stale:
                bt.logging.warning(
                    f"Found stale {self.provider_name} websocket data for {symbol}. Lag: {lag_s} seconds. "
                    f"Max allowed lag for category: "
                    f"{self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]} seconds."
                    f"Ignoring this data.")
            """

        return events

    def get_closes_rest(self, trade_pairs: List[TradePair]) -> dict[str: float]:
        pass

    def get_websocket_lag_for_trade_pair_s(self, tp: str, now_ms: int) -> float | None:
        cur_event = self.latest_websocket_events.get(tp)
        if cur_event:
            return (now_ms - cur_event.end_ms) / 1000.0
        return None

    def spill_price_history(self):
        # Write the price history to disk in a format that will let us plot it
        filename = f"price_history_{self.provider_name}.json"
        with open(filename, 'w') as f:
            json.dump(self.trade_pair_to_price_history, f)

    def debug_log(self):
        now_ms = TimeUtil.now_in_millis()
        trade_pairs_to_track = list(self.latest_websocket_events.keys())
        for tp in trade_pairs_to_track:
            lag_s = self.get_websocket_lag_for_trade_pair_s(tp, now_ms)
            if tp not in self.trade_pair_to_longest_seen_lag_s:
                self.trade_pair_to_longest_seen_lag_s[tp] = lag_s
            else:
                if lag_s > self.trade_pair_to_longest_seen_lag_s[tp]:
                    self.trade_pair_to_longest_seen_lag_s[tp] = lag_s
        # log how long it has been since the last ping
        formatted_lags = {tp: f"{lag:.2f}" for tp, lag in self.trade_pair_to_longest_seen_lag_s.items()}
        formatted_lags = {k: v for k, v in formatted_lags.items() if float(v) > 10}
        bt.logging.info(f"{self.provider_name} Worst lags seen: {formatted_lags}")
        # Log the last time since websocket ping
        now_ms = TimeUtil.now_in_millis()
        formatted_lags = {tp: f"{(now_ms - price_source.end_ms) / 1000.0:.2f}" for tp, price_source in
                          self.latest_websocket_events.items()}
        #formatted_lags = {k:v for k, v in formatted_lags.items() if float(v) > 10}
        bt.logging.info(f"{self.provider_name} Current websocket lags (s): {formatted_lags}")
        # Log the prices
        formatted_prices = {}
        for tp, price_source in self.latest_websocket_events.items():
            if TradePair.get_latest_tade_pair_from_trade_pair_str(tp).is_forex:
                formatted_prices[tp] = f"({price_source.bid:.5f}/{price_source.ask:.5f})"
            else:
                formatted_prices[tp] = f"{price_source.close:.2f}"

        bt.logging.info(f"{self.provider_name} Latest websocket prices: {formatted_prices}")
        bt.logging.info(f'{self.provider_name} websocket n_events_global: {self.tpc_to_n_events}. n_equity_events_skipped_afterhours: {self.n_equity_events_skipped_afterhours}')
        #if self.provider_name == POLYGON_PROVIDER_NAME:
        #    # Log which trade pairs are likely in closed markets
        #    closed_trade_pairs = {}
        #    for trade_pair in TradePair:
        #        if not self.is_market_open(trade_pair):
        #            closed_trade_pairs[trade_pair.trade_pair] = self.closed_market_prices[trade_pair]
        #
        #    bt.logging.info(f"{self.provider_name} Market closed with closing prices for {closed_trade_pairs}")

    def get_price_before_market_close(self, trade_pair: TradePair) -> float | None:
        pass

    def get_websocket_event(self, trade_pair: TradePair) -> PriceSource | None:
        symbol = trade_pair.trade_pair
        cur_event = self.latest_websocket_events.get(symbol)
        if not cur_event:
            return None

        timestamp_ms = cur_event.end_ms
        max_allowed_lag_s = self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
        lag_s = time.time() - timestamp_ms / 1000.0
        is_stale = lag_s > max_allowed_lag_s
        if is_stale:
            bt.logging.info(f"Found stale TD websocket data for {trade_pair.trade_pair}. Lag_s: {lag_s} "
                            f"seconds. Max allowed lag for category: {max_allowed_lag_s} seconds. Ignoring this data.")
        return cur_event
