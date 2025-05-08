import asyncio
import json
import threading
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List

import bittensor as bt
from polygon import WebSocketClient
from setproctitle import setproctitle
from tiingo import TiingoWebsocketClient

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
        self.websocket_manager_thread = None
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

        # initialize our websocket slots to None
        self.WEBSOCKET_THREADS = {}
        self.WEBSOCKET_OBJECTS = {}
        for tpc in [TradePairCategory.CRYPTO, TradePairCategory.FOREX, TradePairCategory.EQUITIES]:
            self.WEBSOCKET_THREADS[tpc] = None
            self.WEBSOCKET_OBJECTS[tpc] = None



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
        t0 = time.time() if self.n_flushes % 500 == 0 else 0
        # Get a list of keys to avoid dictionary changed size during iteration
        # By creating a static list of keys first, we avoid the dictionary size change issue
        keys = list(self.trade_pair_to_recent_events_realtime.keys())
        c = {}
        for k in keys:
            c[k] = deepcopy(self.trade_pair_to_recent_events_realtime[k])
        # Flush the recent events to shared memory in one go
        self.trade_pair_to_recent_events.update(c)
        self.n_flushes += 1
        if t0:
            t1 = time.time()
            bt.logging.info(
                f"Flushed recent {self.provider_name} events to shared memory in {t1 - t0:.2f} seconds, n_flushes {self.n_flushes}")

    def stop_threads(self):
        """
        Stop the threads that are running the websocket clients.
        """
        if self.websocket_manager_thread:
            bt.logging.info(f"Stopping {self.provider_name} websocket manager thread")
            self.websocket_manager_thread.join(timeout=1)
            bt.logging.info(f"Stopped {self.provider_name} websocket manager thread")

        for tpc in TradePairCategory:
            self._kill_ws_for_category(tpc)
            if self.WEBSOCKET_THREADS.get(tpc):
                self.WEBSOCKET_THREADS[tpc].join(timeout=1)
                bt.logging.info(f"Stopped {self.provider_name} websocket thread for {tpc.name.lower()}")
            else:
                bt.logging.warning(f"No websocket thread found for {tpc.name.lower()}")

    def websocket_manager(self):
        """
        This runs in a separate thread. It manages websockets using asyncio tasks
        instead of threads, which simplifies the event loop management.
        """
        setproctitle(f"vali_ws_{self.provider_name}")
        bt.logging.enable_info()

        # Create a single event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Define the websocket coroutine for each category
        async def run_websocket(category):
            while True:
                try:
                    # Create a new client
                    self._create_websocket_client(category)
                    self._subscribe_websockets(category)
                    client = self.WEBSOCKET_OBJECTS.get(category)

                    if client:
                        bt.logging.info(f"Connecting {self.provider_name} websocket for {category}")
                        # Use await instead of run_until_complete since we're in a coroutine
                        await client.connect(self.handle_msg)
                        bt.logging.warning(f"{self.provider_name}[{category}] connection closed, restarting")
                    else:
                        bt.logging.warning(f"{self.provider_name}[{category}] client not created, retrying")
                        await asyncio.sleep(5)
                        continue

                except Exception as e:
                    bt.logging.error(f"{self.provider_name}[{category}] websocket error: {e}")
                    bt.logging.error(traceback.format_exc())

                # Clean up before reconnecting
                try:
                    await self._kill_ws_for_category(category)
                    # Wait before reconnecting
                    await asyncio.sleep(2)
                except Exception as e:
                    bt.logging.error(f"Error during websocket cleanup for {category}: {e}")
                    await asyncio.sleep(5)  # Back off on errors

        async def health_check():
            tpc_to_prev = {t: 0 for t in TradePairCategory}
            last_health_check = time.time()
            last_debug = time.time()

            while True:
                try:
                    now = time.time()
                    if now - last_health_check > self.MAX_TIME_NO_EVENTS_S:
                        resets = []
                        for tpc in TradePairCategory:
                            if tpc == TradePairCategory.INDICES:
                                continue
                            curr = self.tpc_to_n_events[tpc]
                            prev = tpc_to_prev[tpc]

                            if (curr == prev and self.is_market_open(self.get_first_trade_pair_in_category(tpc))):
                                # Get a reference to the current websocket
                                old: WebSocketClient|TiingoWebsocketClient = self.WEBSOCKET_OBJECTS.get(tpc)
                                if old:
                                    resets.append(tpc)
                                    try:
                                        await self._kill_ws_for_category(tpc)
                                        bt.logging.info(f'Health check triggered shutdown for {tpc.name.lower()}')
                                    except Exception as e:
                                        bt.logging.warning(f"Health check shutdown trigger failed for {tpc}: {e}")

                        tpc_to_prev = {t: self.tpc_to_n_events[t] for t in TradePairCategory}
                        if resets:
                            bt.logging.warning(
                                f"Health check restarting {self.provider_name} websockets for {resets!r}. curr {curr} prev {prev}")
                        last_health_check = now

                    if self.using_ipc:
                        self.check_flush()

                    if now - last_debug > self.DEBUG_LOG_INTERVAL_S:
                        try:
                            self.debug_log()
                        except Exception as e:
                            bt.logging.error(f"debug_log() failed: {e}")
                        last_debug = now

                except Exception as e:
                    bt.logging.error(f"Error in health check: {e}")
                    bt.logging.error(traceback.format_exc())

                await asyncio.sleep(1)

        # Create and store tasks for each websocket category
        tasks = []
        for tpc in (TradePairCategory.CRYPTO, TradePairCategory.FOREX, TradePairCategory.EQUITIES):
            task = loop.create_task(run_websocket(tpc))
            tasks.append(task)

        # Add the health check task
        health_task = loop.create_task(health_check())
        tasks.append(health_task)

        # Run the event loop with all tasks
        try:
            loop.run_until_complete(asyncio.gather(*tasks))
        except Exception as e:
            bt.logging.error(f"Main event loop error: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            try:
                # Clean up any pending tasks
                for task in tasks:
                    task.cancel()

                # Close the event loop
                loop.close()
            except Exception as e:
                bt.logging.error(f"Error during shutdown: {e}")

    async def _kill_ws_for_category(self, tpc:TradePairCategory):
        """
        Signal that a websocket should be closed.
        """
        client = self.WEBSOCKET_OBJECTS.get(tpc)
        if client:
            try:
                client.unsubscribe_all()
                bt.logging.info(f'Unsubscribed {self.provider_name} websocket for {tpc.name.lower()}')

                # Set the should_close flag if the client has it
                if hasattr(client, '_should_close'):
                    client._should_close = True

                # Since we're in the same event loop, we can safely close
                await client.close()
                bt.logging.info(f'Closed {self.provider_name} websocket for {tpc.name.lower()}')
                self.WEBSOCKET_OBJECTS[tpc] = None

            except Exception as e:
                bt.logging.warning(f"Failed to initiate shutdown for {self.provider_name} websocket for {tpc}: {e}")

    def _create_websocket_client(self, tpc):
        raise NotImplementedError

    def _subscribe_websockets(self, tpc):
        raise NotImplementedError

    async def handle_msg(self, msg):
        raise NotImplementedError

    def instantiate_not_pickleable_objects(self):
        raise NotImplementedError

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
