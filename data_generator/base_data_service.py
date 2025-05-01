import json
import multiprocessing
import threading
import time
import asyncio
from collections import defaultdict
from typing import List, Optional, Dict, Any

import bittensor as bt

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
                bt.logging.error(
                    f"Failed to get {func_name} with error: {e}, type: {type(e).__name__} in thread {thread_id}")
                return {}

        return wrapper

    return decorator


class BaseDataService():
    def __init__(self, provider_name, ipc_manager=None):
        self.DEBUG_LOG_INTERVAL_S = 180
        self.MAX_TIME_NO_EVENTS_S = 120

        # State
        self.n_flushes = 0
        self.provider_name = provider_name
        self.using_ipc = ipc_manager is not None
        self.tpc_to_n_events = {x: 0 for x in TradePairCategory}
        self._prev_event_count = {x: 0 for x in TradePairCategory}
        self.n_equity_events_skipped_afterhours = 0
        self.trade_pair_to_price_history = defaultdict(list)
        self.closed_market_prices: Dict[TradePair, Optional[PriceSource]] = {tp: None for tp in TradePair}
        self.latest_websocket_events: Dict[str, PriceSource] = {}

        # Event trackers
        self.trade_pair_to_recent_events_realtime = defaultdict(RecentEventTracker)
        if ipc_manager is None:
            self.trade_pair_to_recent_events = defaultdict(RecentEventTracker)
        else:
            self.trade_pair_to_recent_events = ipc_manager.dict()

        # Lag configuration
        self.trade_pair_category_to_longest_allowed_lag_s = {
            TradePairCategory.CRYPTO: 30,
            TradePairCategory.FOREX: 30,
            TradePairCategory.INDICES: 30,
            TradePairCategory.EQUITIES: 30,
        }
        self.timespan_to_ms = {'second': 1000, 'minute': 1000 * 60, 'hour': 1000 * 60 * 60, 'day': 1000 * 60 * 60 * 24}

        # Lookup maps
        self.trade_pair_lookup = {pair.trade_pair: pair for pair in TradePair}
        self.trade_pair_to_longest_seen_lag_s: Dict[str, float] = {}
        self.market_calendar = UnifiedMarketCalendar()

        # Asyncio tasks and clients
        self.websocket_clients: Dict[TradePairCategory, Any] = {}
        self.websocket_tasks: Dict[TradePairCategory, asyncio.Task] = {}
        self.manager_task: Optional[asyncio.Task] = None

        for trade_pair in TradePair:
            assert trade_pair.trade_pair_category in self.trade_pair_category_to_longest_allowed_lag_s, \
                f"Trade pair {trade_pair} has no allowed lag time"

    async def get_close_rest(
            self,
            trade_pair: TradePair,
            timestamp_ms: int = None
    ) -> PriceSource | None:
        """
        Get the most recent price for a trade pair using REST API.
        This needs to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_close_rest")

    def is_market_open(self, trade_pair: TradePair, time_ms=None) -> bool:
        """
        Check if the market for a trade pair is currently open
        """
        if time_ms is None:
            time_ms = TimeUtil.now_in_millis()
        return self.market_calendar.is_market_open(trade_pair, time_ms)

    def get_close(self, trade_pair: TradePair) -> PriceSource | None:
        """
        Get the most recent price for a trade pair using websocket if available,
        falling back to REST.
        """
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
        """Get the first trade pair in a category"""
        # Use generator expression for efficiency
        return next((x for x in TradePair if x.trade_pair_category == tpc), None)

    def flush_ipc(self):
        if self.n_flushes % 500 == 0:
            t0 = time.time()
        else:
            t0 = 0
        """Flush recent events to shared memory. Called every second"""
        # Flush the recent events to shared memory
        self.trade_pair_to_recent_events.update(self.trade_pair_to_recent_events_realtime)

        self.n_flushes += 1
        if t0:
            t1 = time.time()
            bt.logging.info(
                f"Flushed recent {self.provider_name} events to shared memory in {t1 - t0:.2f} seconds, n_flushes "
                f"{self.n_flushes}. len(self.trade_pair_to_recent_events_realtime[k]) = "
                f"{len(self.trade_pair_to_recent_events_realtime)}")

    async def websocket_manager(self):
        bt.logging.enable_info()
        last_health = time.time()
        last_debug = time.time()

        while True:
            now = time.time()

            # — HEALTH CHECK —
            if now - last_health > self.MAX_TIME_NO_EVENTS_S:
                reset = []
                for tpc, prev in list(self._prev_event_count.items()):
                    if tpc == TradePairCategory.INDICES:
                        continue
                    curr = self.tpc_to_n_events.get(tpc, 0)
                    print({'Health check for provider_name': self.provider_name,'tpc': tpc, 'prev':prev, 'curr':curr})
                    if curr == prev and self.is_market_open(self.get_first_trade_pair_in_category(tpc)):
                        reset.append(tpc.name)
                        # properly cancel + restart on your dedicated loop:
                        asyncio.run_coroutine_threadsafe(
                            self._restart_ws_task(tpc),
                            self._loop
                        )
                    self._prev_event_count[tpc] = curr

                if reset:
                    bt.logging.warning(f"{self.provider_name} restarted websockets for: {reset}")
                last_health = now

            # — DEBUG LOG —
            if now - last_debug > self.DEBUG_LOG_INTERVAL_S:
                self.debug_log()
                last_debug = now

            # — FLUSH IPC if needed —
            if self.using_ipc:
                self.flush_ipc()

            await asyncio.sleep(1)

    async def _restart_ws_task(self, tpc: TradePairCategory):
        """
        Gracefully tear down and restart the WS runner for `tpc`.
        """
        # Step 1) Cancel & await the old asyncio Task first
        old_task = self.websocket_tasks.get(tpc)
        if old_task:
            old_task.cancel()
            try:
                await old_task
            except (asyncio.CancelledError, RuntimeError) as e:
                bt.logging.warning(f"Cancellation of old task for {tpc}: {e}")
                pass

        # Step 2) Close the old Polygon client
        old_client = self.websocket_clients.get(tpc)
        if old_client:
            try:
                # Use a non-awaitable method if available
                if hasattr(old_client, 'close_connection'):
                    old_client.close_connection()
                elif hasattr(old_client, 'close'):
                    # Try running synchronously to avoid loop conflicts
                    try:
                        old_client.close()
                    except Exception as e:
                        bt.logging.warning(f"{self.provider_name} {tpc.name} error closing old client: {e}")
            except Exception as e:
                bt.logging.warning(f"{self.provider_name} {tpc.name} error closing old client: {e}")

        # Step 3) Recreate client using a method that handles creating clients properly
        await self._close_create_websocket_objects(tpc)

        # Step 4) Start one new runner task for this category
        self.websocket_tasks[tpc] = self._loop.create_task(self._run_category(tpc))

    async def _run_category(self, tpc):
        while True:
            try:
                if self.provider_name == POLYGON_PROVIDER_NAME:
                    await self._close_create_websocket_objects(tpc)

                if tpc == TradePairCategory.FOREX:
                    await self.main_forex()
                elif tpc == TradePairCategory.EQUITIES:
                    await self.main_stocks()
                elif tpc == TradePairCategory.CRYPTO:
                    await self.main_crypto()
                else:
                    bt.logging.error(f"Unknown category {tpc}, keeping loop alive")
            except Exception as e:
                bt.logging.error(f"{self.provider_name} {self.__name__} for {tpc} crashed: {e}", exc_info=True)
            finally:
                # small delay before reconnecting
                await asyncio.sleep(1)

    # ---------- Abstract hooks ----------
    async def _close_create_websocket_objects(self, tpc: TradePairCategory):
        """Close and recreate websocket objects"""
        raise NotImplementedError

    async def main_stocks(self):
        """Run the websocket for stocks"""
        raise NotImplementedError

    async def main_forex(self):
        """Run the websocket for forex"""
        raise NotImplementedError

    async def main_crypto(self):
        """Run the websocket for crypto"""
        raise NotImplementedError

    # ---------- Async startup ----------
    async def instantiate_not_pickleable_objects(self):
        """Initialize REST clients and other non-pickleable objects"""
        raise NotImplementedError

    def _run_loop(self):
        # new event loop for all WS tasks
        self._loop = asyncio.new_event_loop()

        # tie that loop to this thread
        asyncio.set_event_loop(self._loop)

        # 2) schedule manager + category runners on that loop
        self.manager_task = self._loop.create_task(self.websocket_manager())
        self.websocket_tasks = {}
        for tpc in TradePairCategory:
            if tpc == TradePairCategory.INDICES:
                continue
            self.websocket_tasks[tpc] = self._loop.create_task(self._run_category(tpc))

        # 3) run forever, but catch if it dies
        try:
            self._loop.run_forever()
        except Exception as e:
            bt.logging.error(f"{self.provider_name} event loop crashed: {e}", exc_info=True)
            bt.logging.error(f"Backtrace: {bt.logging.format_exc()}")
    def start_ws_async(self):
        """
        Create a dedicated asyncio loop in a daemon thread,
        schedule manager + per-category runners, and return immediately.
        """
        # launch background thread
        t = multiprocessing.Process(
            target=self._run_loop,
            name=f"{self.provider_name}-{self.__class__.__name__}-asyncio",
            daemon=True,
        )
        t.start()
        self._loop_thread = t

    def stop(self):
        """
        Cancel all WS tasks, stop the event loop, and join its thread.
        """
        # 1) Cancel all running tasks on that loop
        if hasattr(self, '_loop'):
            for task in list(self.websocket_tasks.values()) + [self.manager_task]:
                if task and not task.done():
                    try:
                        task.cancel()
                    except Exception as e:
                        bt.logging.warning(f"Error cancelling task during shutdown: {e}")

            # 2) Stop the loop properly
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError as e:
                bt.logging.warning(f"Error stopping event loop: {e}")

            # 3) Wait for thread to exit if it exists
            if hasattr(self, '_loop_thread'):
                self._loop_thread.join(timeout=5)  # Add timeout to prevent hanging

    def get_closes_websocket(self, trade_pairs: List[TradePair], trade_pair_to_last_order_time_ms) -> dict[
                                                                                                      str: PriceSource]:
        """Get the latest prices from websocket data"""
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

        return events

    async def get_closes_rest(self, trade_pairs: List[TradePair]) -> dict[str: PriceSource]:
        """Get the latest prices from REST API"""
        raise NotImplementedError

    def get_websocket_lag_for_trade_pair_s(self, tp: str, now_ms: int = None) -> float | None:
        """Calculate the lag time for websocket data"""
        if now_ms is None:
            now_ms = TimeUtil.now_in_millis()

        cur_event = self.latest_websocket_events.get(tp)
        if cur_event:
            return (now_ms - cur_event.end_ms) / 1000.0
        return float('inf')

    def spill_price_history(self):
        """Write the price history to disk for analysis"""
        # Write the price history to disk in a format that will let us plot it
        filename = f"price_history_{self.provider_name}.json"
        with open(filename, 'w') as f:
            json.dump(self.trade_pair_to_price_history, f)

    def debug_log(self):
        """Log debugging information"""
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
        bt.logging.info(f"{self.provider_name} Current websocket lags (s): {formatted_lags}")

        # Log the prices
        formatted_prices = {}
        for tp, price_source in self.latest_websocket_events.items():
            trade_pair = TradePair.get_latest_tade_pair_from_trade_pair_str(tp)
            if trade_pair and trade_pair.is_forex:
                formatted_prices[tp] = f"({price_source.bid:.5f}/{price_source.ask:.5f})"
            else:
                formatted_prices[tp] = f"{price_source.close:.2f}"

        bt.logging.info(f"{self.provider_name} Latest websocket prices: {formatted_prices}")
        bt.logging.info(
            f'{self.provider_name} websocket n_events_global: {self.tpc_to_n_events}. n_equity_events_skipped_afterhours: {self.n_equity_events_skipped_afterhours}')

    def get_price_before_market_close(self, trade_pair: TradePair) -> float | None:
        """Get the price before market close"""
        raise NotImplementedError

    def get_websocket_event(self, trade_pair: TradePair) -> PriceSource | None:
        """Get the latest event from websocket data"""
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
            return None
        return cur_event