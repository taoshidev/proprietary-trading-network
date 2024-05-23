import json
import math
import os
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List
import bittensor as bt

from pydantic import BaseModel

from shared_objects.cache_controller import CacheController
from shared_objects.retry import retry, periodic_heartbeat, retry_with_timeout
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_config import ValiConfig, TradePair
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.utils.vali_utils import ValiUtils

TARGET_CHECKPOINT_DURATION_MS = 21600000  # 6 hours
TARGET_LEDGER_WINDOW_MS = 2592000000  # 30 days

class PerfCheckpoint(BaseModel):
    last_update_ms: int
    prev_portfolio_ret: float
    accum_ms: int = 0
    open_ms: int = 0
    n_updates: int = 0
    gain: float = 0.0
    loss: float = 0.0
    mdd: float = 1.0

    def __str__(self):
        return self.to_json_string()

    def to_json_string(self) -> str:
        # Using pydantic's json method with built-in validation
        json_str = self.json()
        # Unfortunately, we can't tell pydantic v1 to strip certain fields so we do that here
        json_loaded = json.loads(json_str)
        return json.dumps(json_loaded)


class PerfLedger(BaseModel):
    max_return: float = 1.0
    target_cp_duration_ms: int = TARGET_CHECKPOINT_DURATION_MS
    target_ledger_window_ms: int = TARGET_LEDGER_WINDOW_MS
    cps: list[PerfCheckpoint] = []

    def __str__(self):
        return self.to_json_string()

    def to_json_string(self) -> str:
        # Using pydantic's json method with built-in validation
        json_str = self.json()
        # Unfortunately, we can't tell pydantic v1 to strip certain fields so we do that here
        json_loaded = json.loads(json_str)
        return json.dumps(json_loaded)

    @property
    def last_update_ms(self):
        if len(self.cps) == 0:
            return 0
        return self.cps[-1].last_update_ms

    @property
    def prev_portfolio_ret(self):
        if len(self.cps) == 0:
            return 1.0 # Initial value
        return self.cps[-1].prev_portfolio_ret

    @property
    def start_time_ms(self):
        if len(self.cps) == 0:
            return 0
        return self.cps[0].last_update_ms

    def create_cps_to_fill_void(self, time_since_last_update_ms: int, now_ms:int, point_in_time_dd:float):
        original_accum_time = self.cps[-1].accum_ms
        delta_accum_time_ms = self.target_cp_duration_ms - original_accum_time
        self.cps[-1].accum_ms += delta_accum_time_ms
        self.cps[-1].last_update_ms += delta_accum_time_ms
        time_since_last_update_ms -= delta_accum_time_ms
        assert time_since_last_update_ms >= 0, (self.cps, time_since_last_update_ms)
        while time_since_last_update_ms > self.target_cp_duration_ms:
            new_cp = PerfCheckpoint(last_update_ms=self.cps[-1].last_update_ms + self.target_cp_duration_ms,
                                    prev_portfolio_ret=self.cps[-1].prev_portfolio_ret,
                                    accum_ms=self.target_cp_duration_ms,
                                    mdd=self.cps[-1].mdd)
            assert new_cp.last_update_ms < now_ms, (self.cps, (now_ms - new_cp.last_update_ms))
            self.cps.append(new_cp)
            time_since_last_update_ms -= self.target_cp_duration_ms

        assert time_since_last_update_ms >= 0
        new_cp = PerfCheckpoint(last_update_ms=self.cps[-1].last_update_ms,
                                prev_portfolio_ret=self.cps[-1].prev_portfolio_ret,
                                mdd=point_in_time_dd)
        assert new_cp.last_update_ms <= now_ms, self.cps
        self.cps.append(new_cp)

    def init_with_first_order(self, order_processed_ms: int, point_in_time_dd:float):
        new_cp = PerfCheckpoint(last_update_ms=order_processed_ms, prev_portfolio_ret=1.0, mdd=point_in_time_dd)
        self.cps.append(new_cp)

    def get_or_create_latest_cp_with_mdd(self, now_ms: int, point_in_time_dd:float):
        if point_in_time_dd is None:  # When "shortcut" is called.
            point_in_time_dd = self.cps[-1].mdd

        if len(self.cps) == 0:
            self.init_with_first_order(now_ms, point_in_time_dd)
            return self.cps[-1]

        time_since_last_update_ms = now_ms - self.cps[-1].last_update_ms
        assert time_since_last_update_ms >= 0, self.cps
        if time_since_last_update_ms + self.cps[-1].accum_ms >= self.target_cp_duration_ms:
            self.create_cps_to_fill_void(time_since_last_update_ms, now_ms, point_in_time_dd)
        else:
            self.cps[-1].mdd = min(self.cps[-1].mdd, point_in_time_dd)

        return self.cps[-1]

    def update_accumulated_time(self, cp: PerfCheckpoint, now_ms: int, miner_hotkey:str, any_open:bool):
        accumulated_time = now_ms - cp.last_update_ms
        if accumulated_time < 0:
            bt.logging.error(f"Negative accumulated time: {accumulated_time} for miner {miner_hotkey}."
                             f" start_time_ms: {self.start_time_ms}, now_ms: {now_ms}")
            accumulated_time = 0
        cp.accum_ms += accumulated_time
        cp.last_update_ms = now_ms
        if any_open:
            cp.open_ms += accumulated_time

    def compute_return_between_ticks(self, current_portfolio_value: float, prev_portfolio_value: float):
        return math.log(current_portfolio_value / prev_portfolio_value)

    def update_returns(self, current_cp: PerfCheckpoint, current_portfolio_value: float):
        delta_return = self.compute_return_between_ticks(current_portfolio_value, current_cp.prev_portfolio_ret)
        n_new_updates = 1
        if current_portfolio_value == current_cp.prev_portfolio_ret:
            n_new_updates = 0
        elif delta_return > 0:
            current_cp.gain += delta_return
        else:
            current_cp.loss += delta_return
        current_cp.prev_portfolio_ret = current_portfolio_value
        current_cp.n_updates += n_new_updates

    def purge_old_cps(self):
        while self.get_total_ledger_duration_ms() > self.target_ledger_window_ms:
            bt.logging.trace(f"Purging old perf cps. Total ledger duration: {self.get_total_ledger_duration_ms()}. Target ledger window: {self.target_ledger_window_ms}")
            self.cps = self.cps[1:]  # Drop the first cp (oldest)

    def update(self, current_portfolio_value:float, now_ms:int, miner_hotkey:str, any_open:bool, point_in_time_dd:float):
        current_cp = self.get_or_create_latest_cp_with_mdd(now_ms, point_in_time_dd)
        self.update_returns(current_cp, current_portfolio_value)
        self.update_accumulated_time(current_cp, now_ms, miner_hotkey, any_open)
        self.purge_old_cps()

    def count_events(self):
        # Return the number of events currently stored
        return len(self.cps)

    def get_product_of_gains(self):
        cumulative_gains = sum(cp.gain for cp in self.cps)
        return math.exp(cumulative_gains)

    def get_product_of_loss(self):
        cumulative_loss = sum(cp.loss for cp in self.cps)
        return math.exp(cumulative_loss)

    def get_total_product(self):
        cumulative_gains = sum(cp.gain for cp in self.cps)
        cumulative_loss = sum(cp.loss for cp in self.cps)
        return math.exp(cumulative_gains + cumulative_loss)

    def get_total_ledger_duration_ms(self):
        return sum(cp.accum_ms for cp in self.cps)


class PerfLedgerManager(CacheController):
    def __init__(self, metagraph, live_price_fetcher=None, running_unit_tests=False, shutdown_dict=None):
        super().__init__(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.shutdown_dict = shutdown_dict
        if live_price_fetcher is None:
            secrets = ValiUtils.get_secrets()
            live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
            self.pds = live_price_fetcher.polygon_data_service
        else:
            self.pds = live_price_fetcher.polygon_data_service
        # Every update, pick a hotkey to rebuild in case polygon 1s candle data changed.
        self.trade_pair_to_price_info = {}
        self.trade_pair_to_position_ret = {}

        self.random_security_screenings = set()
        self.market_calendar = UnifiedMarketCalendar()
        self.n_api_calls = 0
        self.POLYGON_MAX_CANDLE_LIMIT = 49999
        self.UPDATE_LOOKBACK_MS = 600000  # 10 minutes ago. Want to give Polygon time to create candles on the backend.
        self.UPDATE_LOOKBACK_S = self.UPDATE_LOOKBACK_MS // 1000
        self.now_ms = 0  # The largest timestamp we want to buffer candles for. time.time() - UPDATE_LOOKBACK_S
        self.base_dd_stats = {'worst_dd':1.0, 'last_dd':0, 'mrpv':1.0, 'n_closed_pos':0, 'n_checks':0, 'current_portfolio_return': 1.0}
        self.hk_to_dd_stats = defaultdict(lambda: deepcopy(self.base_dd_stats))
        self.n_price_corrections = 0
        self.elimination_rows = []

    @periodic_heartbeat(interval=600, message="perf ledger run_update_loop still running...")
    def run_update_loop(self):
        while not self.shutdown_dict:
            try:
                if self.refresh_allowed(ValiConfig.PERF_LEDGER_REFRESH_TIME_MS):
                    self.update()
                    self.set_last_update_time(skip_message=True)

            except Exception as e:
                # Handle exceptions or log errors
                bt.logging.error(f"Error during perf ledger update: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
            time.sleep(1)

    def get_historical_position(self, position:Position, timestamp_ms:int):
        new_orders = []
        temp_pos = deepcopy(position)
        temp_pos_to_pop = deepcopy(position)
        for o in position.orders:
            if o.processed_ms <= timestamp_ms:
                new_orders.append(o)

        temp_pos.orders = new_orders[:-1]
        temp_pos.rebuild_position_with_updated_orders()
        temp_pos_to_pop.orders = new_orders
        temp_pos_to_pop.rebuild_position_with_updated_orders()
        return temp_pos, temp_pos_to_pop

    def generate_order_timeline(self, positions, now_ms) -> list[tuple]:
        # order to understand timestamps needing checking, position to understand returns per timestamp (will be adjusted)
        # (order, position)
        time_sorted_orders = []
        for p in positions:
            for o in p.orders:
                if o.processed_ms <= now_ms:
                    time_sorted_orders.append((o, p))
        # sort
        time_sorted_orders.sort(key=lambda x: x[0].processed_ms)
        return time_sorted_orders

    def _update_portfolio_debug_counters(self, portfolio_ret):
        self.max_portfolio_value_seen = max(self.max_portfolio_value_seen, portfolio_ret)
        self.min_portfolio_value_seen = min(self.min_portfolio_value_seen, portfolio_ret)

    def replay_all_closed_positions(self,miner_hotkey, tp_to_historical_positions: dict[str:Position]) -> (bool, float):
        max_cuml_return_so_far = 1.0
        cuml_return = 1.0
        n_closed_positions = 0

        # Already sorted
        for _, positions in tp_to_historical_positions.items():
            for position in positions:
                if position.is_open_position:
                    continue
                cuml_return *= position.return_at_close
                n_closed_positions += 1
                max_cuml_return_so_far = max(cuml_return, max_cuml_return_so_far)

        # Replay of closed positions complete.
        stats = self.hk_to_dd_stats[miner_hotkey]
        stats['mrpv'] = max_cuml_return_so_far
        stats['n_closed_pos'] = n_closed_positions
        return max_cuml_return_so_far

    def _can_shortcut(self, tp_to_historical_positions: dict[str: Position], end_time_ms:int,
                      realtime_position_to_pop: Position | None):
        portfolio_value = 1.0
        n_open_positions = 0
        if end_time_ms < TimeUtil.now_in_millis() - TARGET_LEDGER_WINDOW_MS:
            for tp, historical_positions in tp_to_historical_positions.items():
                for i, historical_position in enumerate(historical_positions):
                    if tp == realtime_position_to_pop.trade_pair.trade_pair and i == len(historical_positions) - 1 and realtime_position_to_pop:
                        historical_position = realtime_position_to_pop
                    portfolio_value *= historical_position.return_at_close
                    n_open_positions += historical_position.is_open_position
            # Since this window would be purged anyways, we can skip by using the instantaneous portfolio return.
            #print(f'skipping with {n_open_positions} open positions')
            return True, portfolio_value

        n_positions = 0
        n_closed_positions = 0
        n_positions_newly_opened = 0
        for tp, historical_positions in tp_to_historical_positions.items():
            for historical_position in historical_positions:
                n_positions += 1
                if historical_position.is_closed_position:
                    portfolio_value *= historical_position.return_at_close
                    n_closed_positions += 1
                elif len(historical_position.orders) == 0:
                    n_positions_newly_opened += 1
                n_open_positions += historical_position.is_open_position

        # When building from orders, we will always have at least one open position. When opening a position after a
        # period of all closed positions, we can shortcut by identifying that the new position is the only open position
        # and all other positions are closed. The time before this period, we have only closed positions.
        # Alternatively, we can be attempting to build the ledger after all orders have been accounted for. In this
        # case, we simply need to check if all positions are closed.
        ans = (n_positions == n_closed_positions + n_positions_newly_opened) and (n_positions_newly_opened == 1)
        ans |= (n_positions == n_closed_positions)
        #if ans:
        #    print(f'skipping with {n_open_positions} open positions')
        #if ans:
        #    window_s = (end_time_ms - start_time_ms) // 1000
        #    #print(f"Shortcutting. n_positions: {n_positions}, n_closed_positions: {n_closed_positions}, n_positions_newly_opened: {n_positions_newly_opened}, window_s: {window_s}")
        return ans, portfolio_value



    def new_window_intersects_old_window(self, start_time_s, end_time_s, existing_lb_s, existing_ub_s):
        # Check if new window intersects with the old window
        # An intersection occurs if the start of the new window is before the end of the old window,
        # and the end of the new window is after the start of the old window
        return start_time_s <= existing_ub_s and end_time_s >= existing_lb_s


    def refresh_price_info(self, t_ms, end_time_ms, tp):
        t_s = t_ms // 1000
        existing_lb_s = None
        existing_ub_s = None
        existing_window_s = None
        if tp.trade_pair in self.trade_pair_to_price_info:
            price_info = self.trade_pair_to_price_info[tp.trade_pair]
            existing_ub_s = price_info['ub_s']
            existing_lb_s = price_info['lb_s']
            existing_window_s = existing_ub_s - existing_lb_s
            if existing_lb_s <= t_s <= existing_ub_s:  # No refresh needed
                return
        #else:
        #    print('11111', tp.trade_pair, trade_pair_to_price_info.keys())

        start_time_s = t_s

        end_time_s = end_time_ms // 1000
        requested_seconds = end_time_s - start_time_s
        if requested_seconds > self.POLYGON_MAX_CANDLE_LIMIT:  # Polygon limit
            end_time_s = start_time_s + self.POLYGON_MAX_CANDLE_LIMIT
        elif requested_seconds < 3600:  # Get a batch of candles to minimize number of fetches
            end_time_s = start_time_s + 3600

        end_time_s = min(self.now_ms // 1000, end_time_s)  # Don't fetch candles beyond check time or will fill in null.

        # Always fetch the max number of candles possible to minimize number of fetches
        #end_time_s = start_time_s + self.POLYGON_MAX_CANDLE_LIMIT
        start_time_ms = start_time_s * 1000
        end_time_ms = end_time_s * 1000

        #t0 = time.time()
        #print(f"Starting #{requested_seconds} candle fetch for {tp.trade_pair}")
        price_info, lb_ms, ub_ms = self.pds.get_candles_for_trade_pair_simple(
            trade_pair=tp, start_timestamp_ms=start_time_ms, end_timestamp_ms=end_time_ms)
        self.n_api_calls += 1
        #print(f'Fetched candles for tp {tp.trade_pair} for window {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to {TimeUtil.millis_to_formatted_date_str(end_time_ms)}')
        #print(f'Got {len(price_info)} candles after request of {requested_seconds} candles for tp {tp.trade_pair} in {time.time() - t0}s')

        assert lb_ms >= start_time_ms, (lb_ms, start_time_ms)
        assert ub_ms <= end_time_ms, (ub_ms, end_time_ms)
        # Can we build on top of existing data or should we wipe?
        perform_wipe = True
        if tp.trade_pair in self.trade_pair_to_price_info:
            new_window_size = end_time_s - start_time_s
            if new_window_size + existing_window_s < self.POLYGON_MAX_CANDLE_LIMIT and \
                    self.new_window_intersects_old_window(start_time_s, end_time_s, existing_lb_s, existing_ub_s):
                perform_wipe = False
                self.trade_pair_to_price_info[tp.trade_pair]['ub_s'] = max(existing_ub_s, end_time_s)
                self.trade_pair_to_price_info[tp.trade_pair]['lb_s'] = min(existing_lb_s, start_time_s)
                for k, v in price_info.items():
                    self.trade_pair_to_price_info[tp.trade_pair][k] = v

        if perform_wipe:
            self.trade_pair_to_price_info[tp.trade_pair] = price_info
            self.trade_pair_to_price_info[tp.trade_pair]['lb_s'] = start_time_s
            self.trade_pair_to_price_info[tp.trade_pair]['ub_s'] = end_time_s

        #print(f'Fetched {requested_seconds} s of candles for tp {tp.trade_pair} in {time.time() - t0}s')
        #print('22222', tp.trade_pair, trade_pair_to_price_info.keys())

    def positions_to_portfolio_return(self, tp_to_historical_positions: dict[str: Position], t_ms, miner_hotkey, end_time_ms):
        # Answers "What is the portfolio return at this time t_ms?"
        portfolio_return = 1.0
        t_s = t_ms // 1000
        any_open = False
        #if miner_hotkey.endswith('9osUx') and abs(t_ms - end_time_ms) < 1000:
        #    print('------------------')

        for tp, historical_positions in tp_to_historical_positions.items():  # TODO: multithread over trade pairs?
            for historical_position in historical_positions:
                #if miner_hotkey.endswith('9osUx') and abs(t_ms - end_time_ms) < 1000:
                #    print(f"time {TimeUtil.millis_to_formatted_date_str(t_ms)} hk {miner_hotkey[-5:]} {historical_position.trade_pair.trade_pair} n_orders {len(historical_position.orders)} return {historical_position.current_return} return_at_close {historical_position.return_at_close} closed@{'NA' if historical_position.is_open_position else TimeUtil.millis_to_formatted_date_str(historical_position.orders[-1].processed_ms)}")
                #    if historical_position.trade_pair == TradePair.USDCAD:
                #        for i, order in enumerate(historical_position.orders):
                #            print(f"    {historical_position.trade_pair.trade_pair} order price {order.price} leverage {order.leverage}")
                if self.shutdown_dict:
                    return portfolio_return, any_open
                if len(historical_position.orders) == 0:  # Just opened an order. We will revisit this on the next event as there is no history to replay
                    continue
                if historical_position.is_closed_position:  # We want to process just-closed positions. wont be closed if we are on the corresponding event
                    portfolio_return *= historical_position.return_at_close
                    continue
                if not self.market_calendar.is_market_open(historical_position.trade_pair, t_ms):
                    portfolio_return *= historical_position.return_at_close
                    continue

                any_open = True
                self.refresh_price_info(t_ms, end_time_ms, historical_position.trade_pair)
                price_at_t_s = self.trade_pair_to_price_info[tp].get(t_s)
                if price_at_t_s is not None:
                    self.tp_to_last_price[tp] = price_at_t_s
                    # We are retoractively updating the last order's price if it is the same as the candle price. This is a retro fix for forex bid price propagation as well as nondeterministic failed price filling.
                    if t_ms == historical_position.orders[-1].processed_ms and historical_position.orders[-1].price != price_at_t_s:
                        self.n_price_corrections += 1
                        #bt.logging.warning(f"Price at t_s {t_s} {historical_position.trade_pair.trade_pair} is the same as the last order processed time. Changing price from {historical_position.orders[-1].price} to {price_at_t_s}")
                        historical_position.orders[-1].price = price_at_t_s
                        historical_position.rebuild_position_with_updated_orders()
                    historical_position.set_returns(price_at_t_s, time_ms=t_ms)
                    self.trade_pair_to_position_ret[tp] = historical_position.return_at_close

                portfolio_return *= historical_position.return_at_close
                #assert portfolio_return > 0, f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_s}. opr {opr} rtp {price_at_t_s}, historical position {historical_position}"
        return portfolio_return, any_open

    def update_mdd(self, miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions, perf_ledger):
        perf_ledger.max_return = max(perf_ledger.max_return, portfolio_return)
        dd = self.calculate_drawdown(portfolio_return, perf_ledger.max_return)
        stats = self.hk_to_dd_stats[miner_hotkey]
        if dd < stats['worst_dd']:
            stats['worst_dd'] = dd
        stats['last_dd'] = dd
        stats['n_checks'] += 1
        stats['current_portfolio_return'] = portfolio_return

        mdd_failure = False#self.is_drawdown_beyond_mdd(dd, time_now=TimeUtil.millis_to_datetime(t_ms))
        if mdd_failure:
            bt.logging.warning(f"Drawdown failure for miner {miner_hotkey} at {t_ms}. Portfolio return: {portfolio_return}, max realized portfolio return: {max_realized_portfolio_return}, drawdown: {dd}")
            elimination_row = self.generate_elimination_row(miner_hotkey, dd, mdd_failure, t_ms=t_ms,
                                        price_info=self.tp_to_last_price,
                                        return_info={'dd_stats':stats, 'returns': self.trade_pair_to_position_ret})
            self.elimination_rows.append(elimination_row)
            stats['eliminated'] = True
            print(f'eliminated. Highest portfolio realized return {self.hk_to_dd_stats[miner_hotkey]}. current return {portfolio_return}')
            for _, v in tp_to_historical_positions.items():
                for pos in v:
                    print(f"    time {TimeUtil.millis_to_formatted_date_str(t_ms)} hk {miner_hotkey[-5:]} {pos.trade_pair.trade_pair} return {pos.current_return} return_at_close {pos.return_at_close} closed@{'NA' if pos.is_open_position else TimeUtil.millis_to_formatted_date_str(pos.orders[-1].processed_ms)}")
            return True
        return False

    def init_tp_to_last_price(self, tp_to_historical_positions: dict[str: Position]):
        self.tp_to_last_price = {}
        for k, v in tp_to_historical_positions.items():
            last_pos = v[-1]
            if not last_pos:
                continue
            orders = last_pos.orders
            if not orders:
                continue
            last_order_price = orders[-1].price
            self.tp_to_last_price[k] = last_order_price


    def build_perf_ledger(self, perf_ledger:PerfLedger, tp_to_historical_positions: dict[str: Position], start_time_ms, end_time_ms, miner_hotkey, realtime_position_to_pop) -> bool:
        #print(f"Building perf ledger for {miner_hotkey} from {start_time_ms} to {end_time_ms} ({(end_time_ms - start_time_ms) // 1000} s) order {order}")
        if start_time_ms == 0:  # This ledger is being initialized. First order received.
            perf_ledger.init_with_first_order(end_time_ms, point_in_time_dd=1.0)
            return False
        if start_time_ms == end_time_ms:  # No new orders since last update. Shouldn't happen
            return False

        # "Shortcut" All positions closed and one newly open position OR all closed positions (all orders accounted for).
        can_shortcut, portfolio_return = \
            self._can_shortcut(tp_to_historical_positions, end_time_ms, realtime_position_to_pop)
        if can_shortcut:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, False, point_in_time_dd=None)
            return False

        any_update = any_open = False
        last_dd = None
        self.init_tp_to_last_price(tp_to_historical_positions)

        for t_ms in range(start_time_ms, end_time_ms, 1000):
            if self.shutdown_dict:
                return False
            assert t_ms >= perf_ledger.last_update_ms, f"t_ms: {t_ms}, last_update_ms: {perf_ledger.last_update_ms}, delta_s: {(t_ms - perf_ledger.last_update_ms) // 1000} s. perf ledger {perf_ledger}"
            portfolio_return, any_open = self.positions_to_portfolio_return(tp_to_historical_positions, t_ms, miner_hotkey, end_time_ms)
            self.update_mdd(miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions, perf_ledger)
            assert portfolio_return > 0, f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_ms // 1000}. perf ledger {perf_ledger}"
            last_dd = self.hk_to_dd_stats[miner_hotkey]['last_dd']
            perf_ledger.update(portfolio_return, t_ms, miner_hotkey, any_open, last_dd)
            any_update = True

        # Get last sliver of time
        if any_update and perf_ledger.last_update_ms != end_time_ms:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, any_open, last_dd)
        return False

    def update_all_perf_ledgers(self, hotkey_to_positions: dict[str, List[Position]], existing_perf_ledgers: dict[str, PerfLedger], now_ms: int):
        t_init = time.time()
        self.now_ms = now_ms
        self.elimination_rows = []
        for hotkey_i, (hotkey, positions) in enumerate(hotkey_to_positions.items()):
            eliminated = False
            self.n_api_calls = 0
            if self.shutdown_dict:
                break
            t0 = time.time()
            perf_ledger = existing_perf_ledgers.get(hotkey, PerfLedger())
            existing_perf_ledgers[hotkey] = perf_ledger
            self.trade_pair_to_position_ret = {}
            if hotkey in self.hk_to_dd_stats:
                del self.hk_to_dd_stats[hotkey]
            self.n_price_corrections = 0

            tp_to_historical_positions = defaultdict(list)
            sorted_timeline = self.generate_order_timeline(positions, now_ms)  # Enforces our "now_ms" constraint
            last_event_time = sorted_timeline[-1][0].processed_ms if sorted_timeline else 0
            building_from_new_orders = True
            # There hasn't been a new order since the last update time. Just need to update for open positions
            if last_event_time <= perf_ledger.last_update_ms:
                building_from_new_orders = False

            # There have been order(s) since the last update time. Also, this code is used to initialize a perf ledger.
            realtime_position_to_pop = None
            for event in sorted_timeline:
                if realtime_position_to_pop:
                    symbol = realtime_position_to_pop.trade_pair.trade_pair
                    tp_to_historical_positions[symbol][-1] = realtime_position_to_pop

                order, position = event
                symbol = position.trade_pair.trade_pair
                pos, realtime_position_to_pop = self.get_historical_position(position, order.processed_ms)

                if (symbol in tp_to_historical_positions and
                        pos.position_uuid == tp_to_historical_positions[symbol][-1].position_uuid):
                   tp_to_historical_positions[symbol][-1] = pos
                else:
                    tp_to_historical_positions[symbol].append(pos)

                # Sanity check
                # We want to ensure that all positions or closed or there is only one open position and it is at the end
                n_open_positions = sum(1 for p in tp_to_historical_positions[symbol] if p.is_open_position)
                n_closed_positions = sum(1 for p in tp_to_historical_positions[symbol] if p.is_closed_position)
                assert n_open_positions == 0 or n_open_positions == 1, (n_open_positions, n_closed_positions, tp_to_historical_positions[symbol])
                if n_open_positions == 1:
                    assert tp_to_historical_positions[symbol][-1].is_open_position, (n_open_positions, n_closed_positions, tp_to_historical_positions[symbol])

                # Perf ledger is already built, we just need to run the above loop to build tp_to_historical_positions
                if not building_from_new_orders:
                    continue

                # Already processed this order. Skip until we get to the new order(s)
                if order.processed_ms < perf_ledger.last_update_ms:
                    continue
                # Need to catch up from perf_ledger.last_update_ms to order.processed_ms
                eliminated = self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms, order.processed_ms, hotkey, realtime_position_to_pop)
                if eliminated:
                    break
                #print(f"Done processing order {order}. perf ledger {perf_ledger}")

            if eliminated:
                continue
            # We have processed all orders. Need to catch up to now_ms
            if realtime_position_to_pop:
                symbol = realtime_position_to_pop.trade_pair.trade_pair
                tp_to_historical_positions[symbol][-1] = realtime_position_to_pop
            if now_ms > perf_ledger.last_update_ms:
                self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms, now_ms, hotkey, None)

            if self.shutdown_dict:
                break
            lag = (TimeUtil.now_in_millis() - perf_ledger.last_update_ms) // 1000
            total_product = perf_ledger.get_total_product()
            last_portfolio_value = perf_ledger.prev_portfolio_ret
            bt.logging.info(
                f"Done updating perf ledger for {hotkey} {hotkey_i+1}/{len(hotkey_to_positions)} in {time.time() - t0} "
                f"(s). Lag: {lag} (s). Total product: {total_product}. Last portfolio value: {last_portfolio_value}."
                f" n_api_calls: {self.n_api_calls} dd stats {self.hk_to_dd_stats[hotkey]}. n_price_corrections {self.n_price_corrections}")

        if not self.shutdown_dict:
            PerfLedgerManager.save_perf_ledgers_to_disk(existing_perf_ledgers)
            bt.logging.info(f"Done updating perf ledger for all hotkeys in {time.time() - t_init} s")
            self.write_perf_ledger_eliminations_to_disk(self.elimination_rows)


    @staticmethod
    @retry(tries=10, delay=1, backoff=1)
    def load_perf_ledgers_from_disk() -> dict[str, PerfLedger]:
        file_path = ValiBkpUtils.get_perf_ledgers_path()
        # If the file doesn't exist, return a blank dictionary
        if not os.path.exists(file_path):
            return {}

        with open(file_path, 'r') as file:
            data = json.load(file)

        # Convert the dictionary back to PerfLedger objects
        perf_ledgers = {}
        for key, ledger_data in data.items():
            # Assuming 'cps' field needs to be parsed as a list of PerfCheckpoint
            ledger_data['cps'] = [PerfCheckpoint(**cp) for cp in ledger_data['cps']]
            perf_ledgers[key] = PerfLedger(**ledger_data)

        return perf_ledgers

    def get_positions_perf_ledger(self, testing_one_hotkey=None):
        """
        Since we are running in our own thread, we need to retry in case positions are being written to simultaneously.
        """
        # testing_one_hotkey = '5F1sxW5apTPEYfDJUoHTRG4kGaUmkb3YVi5hwt5A9Fu8Gi6a'
        if testing_one_hotkey:
            hotkey_to_positions = self.get_all_miner_positions_by_hotkey(
                [testing_one_hotkey], sort_positions=True
            )
        else:
            hotkey_to_positions = self.get_all_miner_positions_by_hotkey(
                self.metagraph.hotkeys, sort_positions=True,
                eliminations=self.eliminations
            )
            # Keep only hotkeys with positions
            hotkeys_with_no_positions = set()
            for k, positions in hotkey_to_positions.items():
                if len(positions) == 0:
                    hotkeys_with_no_positions.add(k)
            for k in hotkeys_with_no_positions:
                del hotkey_to_positions[k]

        return hotkey_to_positions

    def update(self, testing_one_hotkey=None):
        perf_ledgers = PerfLedgerManager.load_perf_ledgers_from_disk()
        self._refresh_eliminations_in_memory()
        t_ms = TimeUtil.now_in_millis() - self.UPDATE_LOOKBACK_MS
        #if t_ms < 1714546760000 + 1000 * 60 * 60 * 1:  # Rebuild after bug fix
        #    perf_ledgers = {}
        hotkey_to_positions = self.get_positions_perf_ledger(testing_one_hotkey=testing_one_hotkey)
        eliminated_hotkeys = self.get_eliminated_hotkeys()

        # Remove keys from perf ledgers if they aren't in the metagraph anymore
        metagraph_hotkeys = set(self.metagraph.hotkeys)
        hotkeys_to_delete = set()
        rss_modified = False

        # Determine which hotkeys to remove from the perf ledger
        hotkeys_to_iterate = sorted(list(perf_ledgers.keys()))
        for hotkey in hotkeys_to_iterate:
            if hotkey not in metagraph_hotkeys:
                hotkeys_to_delete.add(hotkey)
            elif hotkey in eliminated_hotkeys: # eliminated hotkeys won't be in positions so they will stop updating. We will keep them in perf ledger for visualizing metrics in the dashboard.
                pass  # Don't want to rebuild. Use this pass statement to avoid rss logic.
            elif not len(hotkey_to_positions.get(hotkey, [])):
                hotkeys_to_delete.add(hotkey)
            elif not rss_modified and hotkey not in self.random_security_screenings:
                rss_modified = True
                self.random_security_screenings.add(hotkey)
                #bt.logging.info(f"perf ledger PLM added {hotkey} with {len(hotkey_to_positions.get(hotkey, []))} positions to rss.")
                hotkeys_to_delete.add(hotkey)

        # Start over again
        if not rss_modified:
            self.random_security_screenings = set()

        perf_ledgers = {k: v for k, v in perf_ledgers.items() if k not in hotkeys_to_delete}
        #hk_to_last_update_date = {k: TimeUtil.millis_to_formatted_date_str(v.last_update_ms)
        #                            if v.last_update_ms else 'N/A' for k, v in perf_ledgers.items()}

        bt.logging.info(f"perf ledger PLM hotkeys to delete: {hotkeys_to_delete}. rss: {self.random_security_screenings}")

        # Time in the past to start updating the perf ledgers
        self.update_all_perf_ledgers(hotkey_to_positions, perf_ledgers, t_ms)

    @staticmethod
    def save_perf_ledgers_to_disk(perf_ledgers: dict[str, PerfLedger]):
        file_path = ValiBkpUtils.get_perf_ledgers_path()
        ValiBkpUtils.write_to_dir(file_path, perf_ledgers)

    def print_perf_ledgers_on_disk(self):
        perf_ledgers = self.load_perf_ledgers_from_disk()
        for hotkey, perf_ledger in perf_ledgers.items():
            print(f"perf ledger for {hotkey}")
            print('    total gain product', perf_ledger.get_product_of_gains())
            print('    total loss product', perf_ledger.get_product_of_loss())
            print('    total product', perf_ledger.get_total_product())


class MockMetagraph():
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys

if __name__ == "__main__":
    all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    all_hotkeys_on_disk = CacheController.get_directory_names(all_miners_dir)
    mmg = MockMetagraph(hotkeys=all_hotkeys_on_disk)
    perf_ledger_manager = PerfLedgerManager(metagraph=mmg, running_unit_tests=False)
    perf_ledger_manager.run_update_loop()

