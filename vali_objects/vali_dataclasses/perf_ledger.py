import json
import math
import os
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List
import bittensor as bt
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS

from pydantic import BaseModel

from shared_objects.cache_controller import CacheController
from shared_objects.retry import retry, periodic_heartbeat
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

TARGET_CHECKPOINT_DURATION_MS = 21600000  # 6 hours
TARGET_LEDGER_WINDOW_MS = 2592000000  # 30 days

class FeeCache():
    def __init__(self):
        self.spread_fee: float = 1.0
        self.spread_fee_last_order_processed_ms: int = 0

        self.carry_fee: float = 1.0  # product of all individual interval fees.
        self.carry_fee_next_increase_time_ms: int = 0  # Compute fees based off the prior interval

    def get_spread_fee(self, position: Position) -> float:
        if position.orders[-1].processed_ms == self.spread_fee_last_order_processed_ms:
            return self.spread_fee

        self.spread_fee = position.get_spread_fee()
        self.spread_fee_last_order_processed_ms = position.orders[-1].processed_ms
        return self.spread_fee

    def get_carry_fee(self, current_time_ms, position: Position) -> float:
        # Calculate the number of times a new day occurred (UTC). If a position is opened at 23:59:58 and this function is
        # called at 00:00:02, the carry fee will be calculated as if a day has passed. Another example: if a position is
        # opened at 23:59:58 and this function is called at 23:59:59, the carry fee will be calculated as 0 days have passed
        if position.is_closed_position:
            current_time_ms = min(current_time_ms, position.close_ms)
        # cache hit?
        if position.trade_pair.is_crypto:
            start_time_cache_hit = self.carry_fee_next_increase_time_ms - MS_IN_8_HOURS
        elif position.trade_pair.is_forex or position.trade_pair.is_indices:
            start_time_cache_hit = self.carry_fee_next_increase_time_ms - MS_IN_24_HOURS
        else:
            raise Exception(f"Unknown trade pair type: {position.trade_pair}")
        if start_time_cache_hit <= current_time_ms < self.carry_fee_next_increase_time_ms:
            return self.carry_fee

        # cache miss
        carry_fee, next_update_time_ms = position.get_carry_fee(current_time_ms)
        assert next_update_time_ms > current_time_ms, [TimeUtil.millis_to_verbose_formatted_date_str(x) for x in (self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms)] + [carry_fee, position] + [self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms]

        assert carry_fee >= 0, (carry_fee, next_update_time_ms, position)
        self.carry_fee = carry_fee
        self.carry_fee_next_increase_time_ms = next_update_time_ms
        return self.carry_fee

class PerfCheckpointData:
    def __init__(self, last_update_ms, prev_portfolio_ret, prev_portfolio_spread_fee=1.0, prev_portfolio_carry_fee=1.0,
                 accum_ms=0, open_ms=0, n_updates=0, gain=0.0, loss=0.0, spread_fee_loss=0.0, carry_fee_loss=0.0, mdd=1.0, mpv=0.0):
        self.last_update_ms = last_update_ms
        self.prev_portfolio_ret = prev_portfolio_ret
        self.prev_portfolio_spread_fee = prev_portfolio_spread_fee
        self.prev_portfolio_carry_fee = prev_portfolio_carry_fee
        self.accum_ms = accum_ms
        self.open_ms = open_ms
        self.n_updates = n_updates
        self.gain = gain
        self.loss = loss
        self.spread_fee_loss = spread_fee_loss
        self.carry_fee_loss = carry_fee_loss
        self.mdd = mdd
        self.mpv = mpv

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return self.__dict__

    @property
    def time_created_ms(self):
        return self.last_update_ms - self.accum_ms

class PerfLedgerData:
    def __init__(self, max_return=1.0, target_cp_duration_ms=TARGET_CHECKPOINT_DURATION_MS, target_ledger_window_ms=TARGET_LEDGER_WINDOW_MS, cps=None):
        if cps is None:
            cps = []
        self.max_return = max_return
        self.target_cp_duration_ms = target_cp_duration_ms
        self.target_ledger_window_ms = target_ledger_window_ms
        self.cps = cps

    def to_dict(self):
        return {
            "max_return": self.max_return,
            "target_cp_duration_ms": self.target_cp_duration_ms,
            "target_ledger_window_ms": self.target_ledger_window_ms,
            "cps": [cp.to_dict() for cp in self.cps]
        }

    @property
    def last_update_ms(self):
        if len(self.cps) == 0:
            return 0
        return self.cps[-1].last_update_ms

    @property
    def prev_portfolio_ret(self):
        if len(self.cps) == 0:
            return 1.0  # Initial value
        return self.cps[-1].prev_portfolio_ret

    @property
    def start_time_ms(self):
        if len(self.cps) == 0:
            return 0
        return self.cps[0].last_update_ms - self.cps[0].accum_ms

    def init_max_portfolio_value(self):
        if self.cps:
            if all(x.mpv != 0 for x in self.cps):  #Some will be 0 if the perf ledger is being incrementally updated right after this PR was merged.
                # Update portfolio max after all trimming and purging has completed.
                self.max_return = max(x.mpv for x in self.cps)
        # Initial portfolio value is 1.0
        self.max_return = max(self.max_return, 1.0)

    def create_cps_to_fill_void(self, time_since_last_update_ms: int, now_ms: int, point_in_time_dd: float):
        original_accum_time = self.cps[-1].accum_ms
        delta_accum_time_ms = self.target_cp_duration_ms - original_accum_time
        self.cps[-1].accum_ms += delta_accum_time_ms
        self.cps[-1].last_update_ms += delta_accum_time_ms
        time_since_last_update_ms -= delta_accum_time_ms
        assert time_since_last_update_ms >= 0, (self.cps, time_since_last_update_ms)
        while time_since_last_update_ms > self.target_cp_duration_ms:
            new_cp = PerfCheckpointData(last_update_ms=self.cps[-1].last_update_ms + self.target_cp_duration_ms,
                                        prev_portfolio_ret=self.cps[-1].prev_portfolio_ret,
                                        prev_portfolio_spread_fee=self.cps[-1].prev_portfolio_spread_fee,
                                        prev_portfolio_carry_fee=self.cps[-1].prev_portfolio_carry_fee,
                                        accum_ms=self.target_cp_duration_ms,
                                        mdd=self.cps[-1].mdd,
                                        mpv=self.cps[-1].prev_portfolio_ret)
            assert new_cp.last_update_ms < now_ms, (self.cps, (now_ms - new_cp.last_update_ms))
            self.cps.append(new_cp)
            time_since_last_update_ms -= self.target_cp_duration_ms

        assert time_since_last_update_ms >= 0
        new_cp = PerfCheckpointData(last_update_ms=self.cps[-1].last_update_ms,
                                    prev_portfolio_ret=self.cps[-1].prev_portfolio_ret,
                                    prev_portfolio_spread_fee=self.cps[-1].prev_portfolio_spread_fee,
                                    prev_portfolio_carry_fee=self.cps[-1].prev_portfolio_carry_fee,
                                    mdd=point_in_time_dd,
                                    mpv=self.cps[-1].prev_portfolio_ret)
        assert new_cp.last_update_ms <= now_ms, self.cps
        self.cps.append(new_cp)

    def init_with_first_order(self, order_processed_ms: int, point_in_time_dd: float):
        new_cp = PerfCheckpointData(last_update_ms=order_processed_ms, prev_portfolio_ret=1.0,
                                    mdd=point_in_time_dd, prev_portfolio_spread_fee=1.0, prev_portfolio_carry_fee=1.0, mpv=1.0)
        self.cps.append(new_cp)

    def get_or_create_latest_cp_with_mdd(self, now_ms: int, point_in_time_dd: float):
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

    def update_accumulated_time(self, cp: PerfCheckpointData, now_ms: int, miner_hotkey: str, any_open: bool):
        accumulated_time = now_ms - cp.last_update_ms
        if accumulated_time < 0:
            bt.logging.error(f"Negative accumulated time: {accumulated_time} for miner {miner_hotkey}."
                             f" start_time_ms: {self.start_time_ms}, now_ms: {now_ms}")
            accumulated_time = 0
        cp.accum_ms += accumulated_time
        cp.last_update_ms = now_ms
        if any_open:
            cp.open_ms += accumulated_time

    def compute_delta_between_ticks(self, cur: float, prev: float):
        return math.log(cur / prev)

    def update_gains_losses(self, current_cp: PerfCheckpointData, current_portfolio_value: float,
                            current_portfolio_fee_spread: float, current_portfolio_carry: float, miner_hotkey: str):
        # TODO: leave as is?
        # current_portfolio_value = current_portfolio_value * current_portfolio_carry  # spread fee already applied
        if current_portfolio_value == current_cp.prev_portfolio_ret:
            n_new_updates = 0
        else:
            n_new_updates = 1
            try:
                delta_return = self.compute_delta_between_ticks(current_portfolio_value, current_cp.prev_portfolio_ret)
            except Exception:
                # print debug info
                raise (Exception(
                    f"hk {miner_hotkey} Error computing delta between ticks. cur: {current_portfolio_value}, prev: {current_cp.prev_portfolio_ret}. cp {current_cp} cpc {current_portfolio_carry}"))
            if delta_return > 0:
                current_cp.gain += delta_return
            else:
                current_cp.loss += delta_return

        if current_cp.prev_portfolio_carry_fee != current_portfolio_carry:
            current_cp.carry_fee_loss += self.compute_delta_between_ticks(current_portfolio_carry, current_cp.prev_portfolio_carry_fee)
        if current_cp.prev_portfolio_spread_fee != current_portfolio_fee_spread:
            current_cp.spread_fee_loss += self.compute_delta_between_ticks(current_portfolio_fee_spread, current_cp.prev_portfolio_spread_fee)

        current_cp.prev_portfolio_ret = current_portfolio_value
        current_cp.prev_portfolio_spread_fee = current_portfolio_fee_spread
        current_cp.prev_portfolio_carry_fee = current_portfolio_carry
        current_cp.mpv = max(current_cp.mpv, current_portfolio_value)
        current_cp.n_updates += n_new_updates

    def purge_old_cps(self):
        while self.get_total_ledger_duration_ms() > self.target_ledger_window_ms:
            bt.logging.trace(
                f"Purging old perf cps. Total ledger duration: {self.get_total_ledger_duration_ms()}. Target ledger window: {self.target_ledger_window_ms}")
            self.cps = self.cps[1:]  # Drop the first cp (oldest)

    def trim_checkpoints(self, cutoff_ms: int):
        new_cps = []
        for cp in self.cps:
            if cp.time_created_ms + self.target_cp_duration_ms >= cutoff_ms:
                continue
            new_cps.append(cp)
        self.cps = new_cps

    def update(self, current_portfolio_value: float, now_ms: int, miner_hotkey: str, any_open: bool,
               point_in_time_dd: float | None, current_portfolio_fee_spread: float, current_portfolio_carry: float):
        current_cp = self.get_or_create_latest_cp_with_mdd(now_ms, point_in_time_dd)
        self.update_gains_losses(current_cp, current_portfolio_value, current_portfolio_fee_spread,
                                 current_portfolio_carry, miner_hotkey)
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


class PerfCheckpoint(BaseModel, PerfCheckpointData):
    last_update_ms: int
    prev_portfolio_ret: float
    prev_portfolio_spread_fee: float = 1.0
    prev_portfolio_carry_fee: float = 1.0

    accum_ms: int = 0
    open_ms: int = 0
    n_updates: int = 0
    gain: float = 0.0
    loss: float = 0.0

    spread_fee_loss: float = 0.0
    carry_fee_loss: float = 0.0

    mdd: float = 1.0
    mpv: float = 0.0

    def __str__(self):
        return self.to_json_string()

    def to_json_string(self) -> str:
        # Using pydantic's json method with built-in validation
        json_str = self.json()
        # Unfortunately, we can't tell pydantic v1 to strip certain fields so we do that here
        json_loaded = json.loads(json_str)
        return json.dumps(json_loaded)

    @classmethod
    def from_data(cls, data: PerfCheckpointData):
        return cls(**data.to_dict())


class PerfLedger(BaseModel, PerfLedgerData):
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

    @classmethod
    def from_data(cls, data: PerfLedgerData):
        cps = [PerfCheckpoint.from_data(cp) for cp in data.cps]
        return cls(max_return=data.max_return, target_cp_duration_ms=data.target_cp_duration_ms,
                   target_ledger_window_ms=data.target_ledger_window_ms, cps=cps)

    @classmethod
    def from_dict(self, data: dict):
        return PerfLedger(max_return=data['max_return'], target_cp_duration_ms=data['target_cp_duration_ms'],
                   target_ledger_window_ms=data['target_ledger_window_ms'], cps=data['cps'])

class PerfLedgerManager(CacheController):
    def __init__(self, metagraph, live_price_fetcher=None, running_unit_tests=False, shutdown_dict=None, position_syncer=None):
        super().__init__(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.shutdown_dict = shutdown_dict
        self.position_syncer = position_syncer
        if live_price_fetcher is None:
            secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
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
        #self.base_dd_stats = {'worst_dd':1.0, 'last_dd':0, 'mrpv':1.0, 'n_closed_pos':0, 'n_checks':0, 'current_portfolio_return': 1.0}
        #self.hk_to_dd_stats = defaultdict(lambda: deepcopy(self.base_dd_stats))
        self.n_price_corrections = 0
        self.elimination_rows = []
        self.hk_to_last_order_processed_ms = {}
        self.position_uuid_to_cache = defaultdict(FeeCache)

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
                bt.logging.error(traceback.format_exc()[1000:])
                time.sleep(30)
            time.sleep(1)

    def get_historical_position(self, position:Position, timestamp_ms:int):
        hk = position.miner_hotkey  # noqa: F841

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
        # Handle position that was forced closed due to realtime data (liquidated)
        if len(new_orders) == len(position.orders) and position.return_at_close == 0:
            temp_pos_to_pop.return_at_close = 0
            temp_pos_to_pop.close_out_position(position.close_ms)

        return temp_pos, temp_pos_to_pop

    def generate_order_timeline(self, positions, now_ms) -> list[tuple]:
        # order to understand timestamps needing checking, position to understand returns per timestamp (will be adjusted)
        # (order, position)
        time_sorted_orders = []
        for p in positions:
            if p.is_closed_position and len(p.orders) < 2:
                bt.logging.info(f"perf ledger generate_order_timeline. Skipping closed position with < 2 orders: {p}")
                continue
            for o in p.orders:
                if o.processed_ms <= now_ms:
                    time_sorted_orders.append((o, p))
        # sort
        time_sorted_orders.sort(key=lambda x: x[0].processed_ms)
        return time_sorted_orders

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
        #stats = self.hk_to_dd_stats[miner_hotkey]
        #stats['mrpv'] = max_cuml_return_so_far
        #stats['n_closed_pos'] = n_closed_positions
        return max_cuml_return_so_far

    def _can_shortcut(self, tp_to_historical_positions: dict[str: Position], end_time_ms:int,
                      realtime_position_to_pop: Position | None) -> (bool, float, float, float):
        portfolio_value = 1.0
        portfolio_spread_fee = 1.0
        portfolio_carry_fee = 1.0
        n_open_positions = 0
        if end_time_ms < TimeUtil.now_in_millis() - TARGET_LEDGER_WINDOW_MS:
            for tp, historical_positions in tp_to_historical_positions.items():
                for i, historical_position in enumerate(historical_positions):
                    if tp == realtime_position_to_pop.trade_pair.trade_pair and i == len(historical_positions) - 1 and realtime_position_to_pop:
                        historical_position = realtime_position_to_pop
                    # fee timeskip OK since we just need to update for the portfolio value.
                    portfolio_spread_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                    portfolio_carry_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(end_time_ms, historical_position)
                    portfolio_value *= historical_position.return_at_close
                    n_open_positions += historical_position.is_open_position
            # Since this window would be purged anyways, we can skip by using the instantaneous portfolio return.
            #print(f'skipping with {n_open_positions} open positions')
            assert portfolio_carry_fee > 0, (portfolio_carry_fee, portfolio_spread_fee)
            return True, portfolio_value, portfolio_spread_fee, portfolio_carry_fee

        n_positions = 0
        n_closed_positions = 0
        n_positions_newly_opened = 0
        for tp, historical_positions in tp_to_historical_positions.items():
            for historical_position in historical_positions:
                n_positions += 1
                if historical_position.is_closed_position:
                    # fee timeskip OK since all positions would be closed
                    portfolio_value *= historical_position.return_at_close
                    portfolio_spread_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                    portfolio_carry_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(end_time_ms, historical_position)

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
        return ans, portfolio_value, portfolio_spread_fee, portfolio_carry_fee



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
        print(f'Fetching candles for tp {tp.trade_pair} for window {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to {TimeUtil.millis_to_formatted_date_str(end_time_ms)}')
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

    def positions_to_portfolio_return(self, tp_to_historical_positions_dense: dict[str: Position], t_ms, miner_hotkey, end_time_ms, portfolio_return, portfolio_spread_fee, portfolio_carry_fee):
        # Answers "What is the portfolio return at this time t_ms?"
        t_s = t_ms // 1000
        any_open = False
        #if miner_hotkey.endswith('9osUx') and abs(t_ms - end_time_ms) < 1000:
        #    print('------------------')

        for tp, historical_positions in tp_to_historical_positions_dense.items():  # TODO: multithread over trade pairs?
            for historical_position in historical_positions:
                #if miner_hotkey.endswith('9osUx') and abs(t_ms - end_time_ms) < 1000:
                #    print(f"time {TimeUtil.millis_to_formatted_date_str(t_ms)} hk {miner_hotkey[-5:]} {historical_position.trade_pair.trade_pair} n_orders {len(historical_position.orders)} return {historical_position.current_return} return_at_close {historical_position.return_at_close} closed@{'NA' if historical_position.is_open_position else TimeUtil.millis_to_formatted_date_str(historical_position.orders[-1].processed_ms)}")
                #    if historical_position.trade_pair == TradePair.USDCAD:
                #        for i, order in enumerate(historical_position.orders):
                #            print(f"    {historical_position.trade_pair.trade_pair} order price {order.price} leverage {order.leverage}")
                if self.shutdown_dict:
                    return portfolio_return, any_open, portfolio_spread_fee, portfolio_carry_fee
                #if len(historical_position.orders) == 0:  # Just opened an order. We will revisit this on the next event as there is no history to replay
                #    continue

                position_spread_fee = self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                position_carry_fee = self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(t_ms, historical_position)
                portfolio_spread_fee *= position_spread_fee
                portfolio_carry_fee *= position_carry_fee

                #if t_ms > 1720749609000:
                #    man = historical_position.get_carry_fee(t_ms)
                #    cache_valid_til_time = self.position_uuid_to_cache[historical_position.position_uuid].carry_fee_next_increase_time_ms
                #    cache_valid_til_time_formatted = TimeUtil.millis_to_formatted_date_str(cache_valid_til_time)
                #    fee_accural_time = TimeUtil.millis_to_formatted_date_str(historical_position.start_carry_fee_accrual_ms)
                #    position_open_ms = historical_position.open_ms
                #    assert position_carry_fee == man[0], \
                #        f"@@@@ cache value/expiry: {position_carry_fee}/{cache_valid_til_time_formatted} manual value: {man}, t_ms: {t_ms}/{TimeUtil.millis_to_formatted_date_str(t_ms)}). Fee accrural time {fee_accural_time}. open_ms {position_open_ms}"

                #if historical_position.is_closed_position:  # We want to process just-closed positions. wont be closed if we are on the corresponding event
                #    continue

                if not self.market_calendar.is_market_open(historical_position.trade_pair, t_ms):
                    portfolio_return *= historical_position.return_at_close
                    continue

                any_open = True
                self.refresh_price_info(t_ms, end_time_ms, historical_position.trade_pair)
                price_at_t_s = self.trade_pair_to_price_info[tp][t_s] if t_s in self.trade_pair_to_price_info[tp] else None
                if price_at_t_s is not None:
                    self.tp_to_last_price[tp] = price_at_t_s
                    # We are retoractively updating the last order's price if it is the same as the candle price. This is a retro fix for forex bid price propagation as well as nondeterministic failed price filling.
                    if t_ms == historical_position.orders[-1].processed_ms and historical_position.orders[-1].price != price_at_t_s:
                        self.n_price_corrections += 1
                        #percent_change = (price_at_t_s - historical_position.orders[-1].price) / historical_position.orders[-1].price
                        #bt.logging.warning(f"Price at t_s {TimeUtil.millis_to_formatted_date_str(t_ms)} {historical_position.trade_pair.trade_pair} is the same as the last order processed time. Changing price from {historical_position.orders[-1].price} to {price_at_t_s}. percent change {percent_change}")
                        historical_position.orders[-1].price = price_at_t_s
                        historical_position.rebuild_position_with_updated_orders()
                    historical_position.set_returns(price_at_t_s, time_ms=t_ms, total_fees=position_spread_fee * position_carry_fee)
                    self.trade_pair_to_position_ret[tp] = historical_position.return_at_close

                portfolio_return *= historical_position.return_at_close
                #assert portfolio_return > 0, f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_s}. opr {opr} rtp {price_at_t_s}, historical position {historical_position}"
        return portfolio_return, any_open, portfolio_spread_fee, portfolio_carry_fee

    def update_mdd(self, miner_hotkey, portfolio_return, perf_ledger):
        perf_ledger.max_return = max(perf_ledger.max_return, portfolio_return)
        dd = self.calculate_drawdown(portfolio_return, perf_ledger.max_return)
        return dd
        #stats = self.hk_to_dd_stats[miner_hotkey]
        #if dd < stats['worst_dd']:
        #    stats['worst_dd'] = dd
        #stats['last_dd'] = dd
        #stats['n_checks'] += 1
        #stats['current_portfolio_return'] = portfolio_return

    def check_liquidated(self, miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions, perf_ledger):
        if portfolio_return == 0:
            bt.logging.warning(f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_ms}. Eliminating miner.")
            elimination_row = self.generate_elimination_row(miner_hotkey, 0.0, 'LIQUIDATED', t_ms=t_ms,
                                        price_info=self.tp_to_last_price,
                                        return_info={'dd_stats':{}, 'returns': self.trade_pair_to_position_ret})
            self.elimination_rows.append(elimination_row)
            #self.hk_to_dd_stats[miner_hotkey]['eliminated'] = True
            for _, v in tp_to_historical_positions.items():
                for pos in v:
                    print(
                        f"    time {TimeUtil.millis_to_formatted_date_str(t_ms)} hk {miner_hotkey[-5:]} {pos.trade_pair.trade_pair} return {pos.current_return} return_at_close {pos.return_at_close} closed@{'NA' if pos.is_open_position else TimeUtil.millis_to_formatted_date_str(pos.orders[-1].processed_ms)}")
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

    def condense_positions(self, tp_to_historical_positions: dict[str: Position]) -> (float, float, float, dict[str: Position]):
        portfolio_return = 1.0
        portfolio_spread_fee = 1.0
        portfolio_carry_fee = 1.0
        tp_to_historical_positions_dense = {}
        for tp, historical_positions in tp_to_historical_positions.items():
            dense_positions = []
            for historical_position in historical_positions:
                if historical_position.is_closed_position:
                    portfolio_return *= historical_position.return_at_close
                    portfolio_spread_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                    portfolio_carry_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(historical_position.orders[-1].processed_ms, historical_position)
                elif len(historical_position.orders) == 0:
                    continue
                else:
                    dense_positions.append(historical_position)
            tp_to_historical_positions_dense[tp] = dense_positions
        return portfolio_return, portfolio_spread_fee, portfolio_carry_fee, tp_to_historical_positions_dense


    def build_perf_ledger(self, perf_ledger:PerfLedgerData, tp_to_historical_positions: dict[str: Position], start_time_ms, end_time_ms, miner_hotkey, realtime_position_to_pop) -> bool:
        #print(f"Building perf ledger for {miner_hotkey} from {start_time_ms} to {end_time_ms} ({(end_time_ms - start_time_ms) // 1000} s) order {order}")
        if len(perf_ledger.cps) == 0:
            perf_ledger.init_with_first_order(end_time_ms, point_in_time_dd=1.0)
            return False

        min_start_time_ms = self.now_ms - TARGET_LEDGER_WINDOW_MS
        start_time_ms = max(start_time_ms, min_start_time_ms)
        end_time_ms = max(start_time_ms, end_time_ms)
        if start_time_ms == end_time_ms:
            return False

        # "Shortcut" All positions closed and one newly open position OR all closed positions (all orders accounted for).
        can_shortcut, portfolio_return, portfolio_spread_fee, portfolio_carry_fee = \
            self._can_shortcut(tp_to_historical_positions, end_time_ms, realtime_position_to_pop)
        if can_shortcut:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, False, None, portfolio_spread_fee, portfolio_carry_fee)
            return False

        any_update = any_open = False
        last_dd = None
        self.init_tp_to_last_price(tp_to_historical_positions)
        initial_portfolio_return, initial_portfolio_spread_fee, initial_portfolio_carry_fee, tp_to_historical_positions_dense = self.condense_positions(tp_to_historical_positions)
        print(f'Updating perf ledger for {miner_hotkey} from {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to {TimeUtil.millis_to_formatted_date_str(end_time_ms)}')
        for t_ms in range(start_time_ms, end_time_ms, 1000):
            if self.shutdown_dict:
                return False
            assert t_ms >= perf_ledger.last_update_ms, f"t_ms: {t_ms}, last_update_ms: {perf_ledger.last_update_ms}, delta_s: {(t_ms - perf_ledger.last_update_ms) // 1000} s. perf ledger {perf_ledger}"
            portfolio_return, any_open, portfolio_spread_fee, portfolio_carry_fee = self.positions_to_portfolio_return(tp_to_historical_positions_dense, t_ms, miner_hotkey, end_time_ms, initial_portfolio_return, initial_portfolio_spread_fee, initial_portfolio_carry_fee)
            if portfolio_return == 0 and self.check_liquidated(miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions, perf_ledger):
                return True
            perf_ledger.max_return = max(perf_ledger.max_return, portfolio_return)
            last_dd = self.calculate_drawdown(portfolio_return, perf_ledger.max_return)

            #last_dd = self.update_mdd(miner_hotkey, portfolio_return, perf_ledger)
            #assert portfolio_return > 0, f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_ms // 1000}. perf ledger {perf_ledger}"
            #last_dd = self.hk_to_dd_stats[miner_hotkey]['last_dd']
            perf_ledger.update(portfolio_return, t_ms, miner_hotkey, any_open, last_dd, portfolio_spread_fee, portfolio_carry_fee)
            any_update = True

        # Get last sliver of time
        if any_update and perf_ledger.last_update_ms != end_time_ms:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, any_open, last_dd, portfolio_spread_fee, portfolio_carry_fee)
        return False

    def update_one_perf_ledger(self, hotkey_i, n_hotkeys, hotkey, positions, now_ms, existing_perf_ledgers) -> None:
        # if hotkey != '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
        #    continue

        eliminated = False
        self.n_api_calls = 0

        t0 = time.time()
        perf_ledger = existing_perf_ledgers.get(hotkey)
        if perf_ledger is None:
            perf_ledger = PerfLedgerData()
            verbose = True
        else:
            verbose = False
        existing_perf_ledgers[hotkey] = perf_ledger
        perf_ledger.init_max_portfolio_value()

        self.trade_pair_to_position_ret = {}
        #if hotkey in self.hk_to_dd_stats:
        #    del self.hk_to_dd_stats[hotkey]
        self.n_price_corrections = 0

        tp_to_historical_positions = defaultdict(list)
        sorted_timeline = self.generate_order_timeline(positions, now_ms)  # Enforces our "now_ms" constraint
        last_event_time = sorted_timeline[-1][0].processed_ms if sorted_timeline else 0
        self.hk_to_last_order_processed_ms[hotkey] = last_event_time

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
                if realtime_position_to_pop.return_at_close == 0:  # liquidated
                    self.check_liquidated(hotkey, 0.0, realtime_position_to_pop.close_ms, tp_to_historical_positions,
                                          perf_ledger)
                    eliminated = True
                    break

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
            assert n_open_positions == 0 or n_open_positions == 1, (
            n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])
            if n_open_positions == 1:
                assert tp_to_historical_positions[symbol][-1].is_open_position, (
                n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])

            # Perf ledger is already built, we just need to run the above loop to build tp_to_historical_positions
            if not building_from_new_orders:
                continue

            # Already processed this order. Skip until we get to the new order(s)
            if order.processed_ms < perf_ledger.last_update_ms:
                continue
            # Need to catch up from perf_ledger.last_update_ms to order.processed_ms
            eliminated = self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms,
                                                order.processed_ms, hotkey, realtime_position_to_pop)
            if eliminated:
                break
            # print(f"Done processing order {order}. perf ledger {perf_ledger}")

        if eliminated:
            return
        # We have processed all orders. Need to catch up to now_ms
        if realtime_position_to_pop:
            symbol = realtime_position_to_pop.trade_pair.trade_pair
            tp_to_historical_positions[symbol][-1] = realtime_position_to_pop
        if now_ms > perf_ledger.last_update_ms:
            self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms, now_ms, hotkey,
                                   None)

        lag = (TimeUtil.now_in_millis() - perf_ledger.last_update_ms) // 1000
        total_product = perf_ledger.get_total_product()
        last_portfolio_value = perf_ledger.prev_portfolio_ret
        if verbose:
            bt.logging.info(
                f"Done updating perf ledger for {hotkey} {hotkey_i + 1}/{n_hotkeys} in {time.time() - t0} "
                f"(s). Lag: {lag} (s). Total product: {total_product}. Last portfolio value: {last_portfolio_value}."
                f" n_api_calls: {self.n_api_calls} dd stats {None}. n_price_corrections {self.n_price_corrections}"
                f" last cp {perf_ledger.cps[-1]}. perf_ledger_mpv {perf_ledger.max_return}")

    def update_all_perf_ledgers(self, hotkey_to_positions: dict[str, List[Position]], existing_perf_ledgers: dict[str, PerfLedgerData], now_ms: int, return_dict=False) -> None | dict[str, PerfLedgerData]:
        t_init = time.time()
        self.now_ms = now_ms
        self.elimination_rows = []
        n_hotkeys = len(hotkey_to_positions)
        for hotkey_i, (hotkey, positions) in enumerate(hotkey_to_positions.items()):
            try:
                self.update_one_perf_ledger(hotkey_i, n_hotkeys, hotkey, positions, now_ms, existing_perf_ledgers)
            except Exception as e:
                bt.logging.error(f"Error updating perf ledger for {hotkey}: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
                continue

        bt.logging.info(f"Done updating perf ledger for all hotkeys in {time.time() - t_init} s")
        self.write_perf_ledger_eliminations_to_disk(self.elimination_rows)

        if self.shutdown_dict:
            return

        if return_dict:
            return existing_perf_ledgers
        else:
            PerfLedgerManager.save_perf_ledgers_to_disk(existing_perf_ledgers)


    @staticmethod
    @retry(tries=10, delay=1, backoff=1)
    def load_perf_ledgers_from_disk(read_as_pydantic=True) -> dict[str, PerfLedgerData]:
        file_path = ValiBkpUtils.get_perf_ledgers_path()
        # If the file doesn't exist, return a blank dictionary
        if not os.path.exists(file_path):
            return {}

        with open(file_path, 'r') as file:
            data = json.load(file)

        perf_ledgers = {}

        # Convert the dictionary back to PerfLedger objects
        for key, ledger_data in data.items():
            if read_as_pydantic:
                ledger_data['cps'] = [PerfCheckpoint(**cp) for cp in ledger_data['cps']]
                perf_ledgers[key] = PerfLedger(**ledger_data)
            else:
                # Assuming 'cps' field needs to be parsed as a list of PerfCheckpoint
                ledger_data['cps'] = [PerfCheckpointData(**cp) for cp in ledger_data['cps']]
                perf_ledgers[key] = PerfLedgerData(**ledger_data)

        return perf_ledgers

    def get_positions_perf_ledger(self, testing_one_hotkey=None):
        """
        Since we are running in our own thread, we need to retry in case positions are being written to simultaneously.
        """
        #testing_one_hotkey = '5GzYKUYSD5d7TJfK4jsawtmS2bZDgFuUYw8kdLdnEDxSykTU'
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

    def generate_perf_ledgers_for_analysis(self, hotkey_to_positions: dict[str, List[Position]], t_ms: int = None) -> dict[str, PerfLedger]:
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()  # Time to build the perf ledgers up to. Goes back 30 days from this time.
        existing_perf_ledgers = {}
        ans_data = self.update_all_perf_ledgers(hotkey_to_positions, existing_perf_ledgers, t_ms, return_dict=True)
        ans_pydantic = {k: PerfLedger.from_data(v) for k, v in ans_data.items()}
        return ans_pydantic

    def update(self, testing_one_hotkey=None, regenerate_all_ledgers=False):
        perf_ledgers = PerfLedgerManager.load_perf_ledgers_from_disk(read_as_pydantic=False)
        self._refresh_eliminations_in_memory()
        t_ms = TimeUtil.now_in_millis() - self.UPDATE_LOOKBACK_MS
        #if t_ms < 1720763350000 + 1000 * 60 * 60 * 1:  # Rebuild after bug fix
        #    for ledger in perf_ledgers.values():
        #        ledger.trim_checkpoints(1720665175000)  #  Wednesday, July 10, 2024 10:32:55 PM ET
        hotkey_to_positions = self.get_positions_perf_ledger(testing_one_hotkey=testing_one_hotkey)

        def sort_key(x):
            # Highest priority. Want to rebuild this hotkey first in case it has an incorrect dd from a Polygon bug
            #if x == "5Et6DsfKyfe2PBziKo48XNsTCWst92q8xWLdcFy6hig427qH":
            #    return float('inf')
            # Otherwise, sort by the last trade time
            return hotkey_to_positions[x][-1].orders[-1].processed_ms

        # Sort the keys with the custom sort key
        hotkeys_ordered_by_last_trade = sorted(hotkey_to_positions.keys(), key=sort_key, reverse=True)
        eliminated_hotkeys = self.get_eliminated_hotkeys()

        # Remove keys from perf ledgers if they aren't in the metagraph anymore
        metagraph_hotkeys = set(self.metagraph.hotkeys)
        hotkeys_to_delete = set()
        rss_modified = False

        # Determine which hotkeys to remove from the perf ledger
        hotkeys_to_iterate = [x for x in hotkeys_ordered_by_last_trade if x in perf_ledgers]

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

        # Regenerate checkpoints if a hotkey was modified during position sync
        attempting_invalidations = bool(self.position_syncer) and bool(self.position_syncer.perf_ledger_hks_to_invalidate)
        if attempting_invalidations:
            for hk, t in self.position_syncer.perf_ledger_hks_to_invalidate.items():
                hotkeys_to_delete.add(hk)

        perf_ledgers = {k: v for k, v in perf_ledgers.items() if k not in hotkeys_to_delete}
        #hk_to_last_update_date = {k: TimeUtil.millis_to_formatted_date_str(v.last_update_ms)
        #                            if v.last_update_ms else 'N/A' for k, v in perf_ledgers.items()}

        bt.logging.info(f"perf ledger PLM hotkeys to delete: {hotkeys_to_delete}. rss: {self.random_security_screenings}")

        if regenerate_all_ledgers or testing_one_hotkey:
            bt.logging.info("Regenerating all perf ledgers")
            perf_ledgers = {}

        self.trim_ledgers(perf_ledgers, hotkey_to_positions)

        # Time in the past to start updating the perf ledgers
        self.update_all_perf_ledgers(hotkey_to_positions, perf_ledgers, t_ms, return_dict=bool(testing_one_hotkey))

        # Clear invalidations after successful update. Prevent race condition by only clearing if we attempted invalidations.
        if attempting_invalidations:
            self.position_syncer.perf_ledger_hks_to_invalidate = {}

        if testing_one_hotkey:
            ledger = perf_ledgers[testing_one_hotkey]
            for x in ledger.cps:
                last_update_formated = TimeUtil.millis_to_timestamp(x.last_update_ms)
                print(x, last_update_formated)

    @staticmethod
    def save_perf_ledgers_to_disk(perf_ledgers: dict[str, PerfLedgerData] | dict[str, dict], raw_json=False):
        # Convert to PerfLedger (pydantic validation)
        if raw_json:
            pydantic_perf_ledgers = {key: PerfLedger.from_dict(value) for key, value in perf_ledgers.items()}
        else:
            pydantic_perf_ledgers = {key: PerfLedger.from_data(value) for key, value in perf_ledgers.items()}
        file_path = ValiBkpUtils.get_perf_ledgers_path()
        ValiBkpUtils.write_to_dir(file_path, pydantic_perf_ledgers)

    def print_perf_ledgers_on_disk(self):
        perf_ledgers = self.load_perf_ledgers_from_disk()
        for hotkey, perf_ledger in perf_ledgers.items():
            print(f"perf ledger for {hotkey}")
            print('    total gain product', perf_ledger.get_product_of_gains())
            print('    total loss product', perf_ledger.get_product_of_loss())
            print('    total product', perf_ledger.get_total_product())

    def trim_ledgers(self, perf_ledgers, hotkey_to_positions):
        """
        Trim checkpoints subject to race condition. Perf ledger fully update loop can take 30 min.
        An order can come in during update.
        """
        for hk, ledger in perf_ledgers.items():
            last_acked_order_time_ms = self.hk_to_last_order_processed_ms.get(hk)
            if not last_acked_order_time_ms:
                continue
            ledger_last_update_time = ledger.last_update_ms
            positions = hotkey_to_positions.get(hk)
            if positions is None:
                continue
            for p in positions:
                for o in p.orders:
                    # An order came in while the perf ledger was being updated. Trim the checkpoints to avoid a race condition.
                    if last_acked_order_time_ms < o.processed_ms < ledger_last_update_time:
                        order_time_str = TimeUtil.millis_to_formatted_date_str(o.processed_ms)
                        last_acked_time_str = TimeUtil.millis_to_formatted_date_str(last_acked_order_time_ms)
                        ledger_last_update_time_str = TimeUtil.millis_to_formatted_date_str(ledger_last_update_time)
                        bt.logging.info(f"Trimming checkpoints for {hk}. Order came in at {order_time_str} after last acked time {last_acked_time_str} but before perf ledger update time {ledger_last_update_time_str}")
                        ledger.trim_checkpoints(o.processed_ms)


class MockMetagraph():
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys

if __name__ == "__main__":
    bt.logging.enable_default()
    all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    all_hotkeys_on_disk = CacheController.get_directory_names(all_miners_dir)
    mmg = MockMetagraph(hotkeys=all_hotkeys_on_disk)
    perf_ledger_manager = PerfLedgerManager(metagraph=mmg, running_unit_tests=False)
    perf_ledger_manager.update(testing_one_hotkey='5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X')
    #perf_ledger_manager.update(regenerate_all_ledgers=True)

