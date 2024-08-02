# perf_ledger.pyx
from vali_config import ValiConfig, TradePair
from vali_objects.position import Position
import json
import math
import os
import time
import traceback
from copy import deepcopy
import bittensor as bt
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS

from shared_objects.cache_controller import CacheController
from shared_objects.retry import retry, periodic_heartbeat
from time_util.time_util import TimeUtil, UnifiedMarketCalendar

from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

TARGET_CHECKPOINT_DURATION_MS = 21600000  # 6 hours
TARGET_LEDGER_WINDOW_MS = 2592000000  # 30 days

cdef class FeeCache:
    cdef float spread_fee
    cdef long spread_fee_last_order_processed_ms
    cdef float carry_fee
    cdef long carry_fee_next_increase_time_ms

    def __init__(self):
        self.spread_fee = 1.0
        self.spread_fee_last_order_processed_ms = 0

        self.carry_fee = 1.0  # product of all individual interval fees.
        self.carry_fee_next_increase_time_ms = 0  # Compute fees based off the prior interval

    cpdef float get_spread_fee(self, position: Position):
        if position.orders[-1].processed_ms == self.spread_fee_last_order_processed_ms:
            return self.spread_fee

        self.spread_fee = position.get_spread_fee()
        self.spread_fee_last_order_processed_ms = position.orders[-1].processed_ms
        return self.spread_fee

    cpdef float get_carry_fee(self, current_time_ms, position: Position):
        if position.is_closed_position:
            current_time_ms = min(current_time_ms, position.close_ms)
        if position.trade_pair.is_crypto:
            start_time_cache_hit = self.carry_fee_next_increase_time_ms - MS_IN_8_HOURS
        elif position.trade_pair.is_forex or position.trade_pair.is_indices:
            start_time_cache_hit = self.carry_fee_next_increase_time_ms - MS_IN_24_HOURS
        else:
            raise Exception(f"Unknown trade pair type: {position.trade_pair}")
        if start_time_cache_hit <= current_time_ms < self.carry_fee_next_increase_time_ms:
            return self.carry_fee

        carry_fee, next_update_time_ms = position.get_carry_fee(current_time_ms)
        assert next_update_time_ms > current_time_ms, [TimeUtil.millis_to_verbose_formatted_date_str(x) for x in (self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms)] + [carry_fee, position] + [self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms]

        assert carry_fee >= 0, (carry_fee, next_update_time_ms, position)
        self.carry_fee = carry_fee
        self.carry_fee_next_increase_time_ms = next_update_time_ms
        return self.carry_fee

cdef class PerfCheckpointData:
    cdef long _last_update_ms
    cdef float _prev_portfolio_ret
    cdef float _prev_portfolio_spread_fee
    cdef float _prev_portfolio_carry_fee
    cdef long _accum_ms
    cdef long _open_ms
    cdef long _n_updates
    cdef float _gain
    cdef float _loss
    cdef float _spread_fee_loss
    cdef float _carry_fee_loss
    cdef float _mdd
    cdef float _mpv

    def __init__(self, long last_update_ms, float prev_portfolio_ret, float prev_portfolio_spread_fee=1.0, float prev_portfolio_carry_fee=1.0,
                 long accum_ms=0, long open_ms=0, long n_updates=0, float gain=0.0, float loss=0.0, float spread_fee_loss=0.0, float carry_fee_loss=0.0,
                 float mdd=1.0, float mpv=0.0):
        self._last_update_ms = last_update_ms
        self._prev_portfolio_ret = prev_portfolio_ret
        self._prev_portfolio_spread_fee = prev_portfolio_spread_fee
        self._prev_portfolio_carry_fee = prev_portfolio_carry_fee
        self._accum_ms = accum_ms
        self._open_ms = open_ms
        self._n_updates = n_updates
        self._gain = gain
        self._loss = loss
        self._spread_fee_loss = spread_fee_loss
        self._carry_fee_loss = carry_fee_loss
        self._mdd = mdd
        self._mpv = mpv

    def __str__(self):
        return str(self.to_dict())


    cpdef dict to_dict(self):
        return {
            'last_update_ms': self._last_update_ms,
            'prev_portfolio_ret': self._prev_portfolio_ret,
            'prev_portfolio_spread_fee': self._prev_portfolio_spread_fee,
            'prev_portfolio_carry_fee': self._prev_portfolio_carry_fee,
            'accum_ms': self._accum_ms,
            'open_ms': self._open_ms,
            'n_updates': self._n_updates,
            'gain': self._gain,
            'loss': self._loss,
            'spread_fee_loss': self._spread_fee_loss,
            'carry_fee_loss': self._carry_fee_loss,
            'mdd': self._mdd,
            'mpv': self._mpv
        }

    cpdef long time_created_ms(self):
        return self._last_update_ms - self._accum_ms

    #cpdef long get_last_update_ms(self):
    #    return self.last_update_ms
    property last_update_ms:
        def __get__(self):
            return self._last_update_ms
        def __set__(self, value):
            self._last_update_ms = value

    property prev_portfolio_ret:
        def __get__(self):
            return self._prev_portfolio_ret
        def __set__(self, value):
            self._prev_portfolio_ret = value

    property prev_portfolio_spread_fee:
        def __get__(self):
            return self._prev_portfolio_spread_fee
        def __set__(self, value):
            self._prev_portfolio_spread_fee = value

    property prev_portfolio_carry_fee:
        def __get__(self):
            return self._prev_portfolio_carry_fee
        def __set__(self, value):
            self._prev_portfolio_carry_fee = value

    property accum_ms:
        def __get__(self):
            return self._accum_ms
        def __set__(self, value):
            self._accum_ms = value

    property open_ms:
        def __get__(self):
            return self._open_ms
        def __set__(self, value):
            self._open_ms = value

    property n_updates:
        def __get__(self):
            return self._n_updates
        def __set__(self, value):
            self._n_updates = value

    property gain:
        def __get__(self):
            return self._gain
        def __set__(self, value):
            self._gain = value

    property loss:
        def __get__(self):
            return self._loss
        def __set__(self, value):
            self._loss = value

    property spread_fee_loss:
        def __get__(self):
            return self._spread_fee_loss
        def __set__(self, value):
            self._spread_fee_loss = value

    property carry_fee_loss:
        def __get__(self):
            return self._carry_fee_loss
        def __set__(self, value):
            self._carry_fee_loss = value

    property mdd:
        def __get__(self):
            return self._mdd
        def __set__(self, value):
            self._mdd = value

    property mpv:
        def __get__(self):
            return self._mpv
        def __set__(self, value):
            self._mpv = value


cdef class PerfLedgerData:
    cdef float max_return
    cdef long target_cp_duration_ms
    cdef long target_ledger_window_ms
    cdef list[object] _cps

    def __init__(self, float max_return=1.0, long target_cp_duration_ms=TARGET_CHECKPOINT_DURATION_MS,
                 long target_ledger_window_ms=TARGET_LEDGER_WINDOW_MS, list[object] cps=None):
        if cps is None:
            cps = []
        self.max_return = max_return
        self.target_cp_duration_ms = target_cp_duration_ms
        self.target_ledger_window_ms = target_ledger_window_ms
        self._cps = cps

    property cps:
        def __get__(self):
            return self._cps
        def __set__(self, value):
            self._cps = value

    cpdef dict to_dict(self):
        return {
            "max_return": self.max_return,
            "target_cp_duration_ms": self.target_cp_duration_ms,
            "target_ledger_window_ms": self.target_ledger_window_ms,
            "cps": [cp.to_dict() for cp in self.cps]
        }

    cpdef long last_update_ms(self):
        if len(self.cps) == 0:
            return 0

        return self.cps[-1].last_update_ms


    cpdef float prev_portfolio_ret(self):
        if len(self.cps) == 0:
            return 1.0
        return self.cps[-1].prev_portfolio_ret

    cpdef long start_time_ms(self):
        if len(self.cps) == 0:
            return 0
        return self.cps[0].last_update_ms - self.cps[0].accum_ms

    def init_max_portfolio_value(self):
        if self.cps:
            use_max_v2 = all(x.mpv != 0 for x in self.cps)
            if use_max_v2:
                self.max_return = max(x.mpv for x in self.cps)
        self.max_return = max(self.max_return, 1.0)

    cpdef void create_cps_to_fill_void(self, long time_since_last_update_ms, long now_ms, float point_in_time_dd):
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

    cpdef void init_with_first_order(self, long order_processed_ms, float point_in_time_dd):
        new_cp = PerfCheckpointData(last_update_ms=order_processed_ms, prev_portfolio_ret=1.0,
                                    mdd=point_in_time_dd, prev_portfolio_spread_fee=1.0, prev_portfolio_carry_fee=1.0, mpv=1.0)
        self.cps.append(new_cp)

    cpdef PerfCheckpointData get_or_create_latest_cp_with_mdd(self, long now_ms, float point_in_time_dd):
        if point_in_time_dd == -1:
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

    cpdef void update_accumulated_time(self, PerfCheckpointData cp, long now_ms, str miner_hotkey, bint any_open):
        accumulated_time = now_ms - cp.last_update_ms
        if accumulated_time < 0:
            bt.logging.error(f"Negative accumulated time: {accumulated_time} for miner {miner_hotkey}."
                             f" start_time_ms: {self.start_time_ms}, now_ms: {now_ms}")
            accumulated_time = 0
        cp.accum_ms += accumulated_time
        cp.last_update_ms = now_ms
        if any_open:
            cp.open_ms += accumulated_time

    cpdef float compute_delta_between_ticks(self, float cur, float prev):
        return math.log(cur / prev)

    cpdef void update_gains_losses(self, PerfCheckpointData current_cp, float current_portfolio_value,
                                   float current_portfolio_fee_spread, float current_portfolio_carry, str miner_hotkey):
        if current_portfolio_value == current_cp.prev_portfolio_ret:
            n_new_updates = 0
        else:
            n_new_updates = 1
            try:
                delta_return = self.compute_delta_between_ticks(current_portfolio_value, current_cp.prev_portfolio_ret)
            except Exception:
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

    cpdef void purge_old_cps(self):
        while self.get_total_ledger_duration_ms() > self.target_ledger_window_ms:
            bt.logging.trace(
                f"Purging old perf cps. Total ledger duration: {self.get_total_ledger_duration_ms()}. Target ledger window: {self.target_ledger_window_ms}")
            self.cps = self.cps[1:]

    cpdef void trim_checkpoints(self, long cutoff_ms):
        new_cps = []
        for cp in self.cps:
            if cp.time_created_ms + self.target_cp_duration_ms >= cutoff_ms:
                continue
            new_cps.append(cp)
        self.cps = new_cps

    cpdef void update(self, float current_portfolio_value, long now_ms, str miner_hotkey, bint any_open,
                      float point_in_time_dd, float current_portfolio_fee_spread, float current_portfolio_carry):
        current_cp = self.get_or_create_latest_cp_with_mdd(now_ms, point_in_time_dd)
        self.update_gains_losses(current_cp, current_portfolio_value, current_portfolio_fee_spread,
                                 current_portfolio_carry, miner_hotkey)
        self.update_accumulated_time(current_cp, now_ms, miner_hotkey, any_open)
        self.purge_old_cps()

    cpdef int count_events(self):
        return len(self.cps)

    def get_product_of_gains(self) -> float:
        cumulative_gains = sum(cp.gain for cp in self.cps)
        return math.exp(cumulative_gains)

    def get_product_of_loss(self) -> float:
        cumulative_loss = sum(cp.loss for cp in self.cps)
        return math.exp(cumulative_loss)

    def get_total_product(self) -> float:
        cumulative_gains = sum(cp.gain for cp in self.cps)
        cumulative_loss = sum(cp.loss for cp in self.cps)
        return math.exp(cumulative_gains + cumulative_loss)

    def get_total_ledger_duration_ms(self) -> int:
        return sum(cp.accum_ms for cp in self.cps)

cdef class PerfLedgerManager():
    cdef object metagraph
    cdef dict shutdown_dict
    cdef object position_syncer
    cdef object pds
    cdef dict trade_pair_to_price_info
    cdef dict trade_pair_to_position_ret
    cdef set random_security_screenings
    cdef object market_calendar
    cdef int n_api_calls
    cdef int POLYGON_MAX_CANDLE_LIMIT
    cdef long UPDATE_LOOKBACK_MS
    cdef int UPDATE_LOOKBACK_S
    cdef long now_ms
    cdef int n_price_corrections
    cdef list elimination_rows
    cdef dict hk_to_last_order_processed_ms
    cdef dict position_uuid_to_cache
    cdef list eliminations
    cdef object cc
    cdef dict tp_to_last_price

    def __init__(self, metagraph, live_price_fetcher=None, running_unit_tests=False, shutdown_dict=None, position_syncer=None):
        self.metagraph = metagraph
        self.shutdown_dict = shutdown_dict
        self.position_syncer = position_syncer
        if live_price_fetcher is None:
            secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
            live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
            self.pds = live_price_fetcher.polygon_data_service
        else:
            self.pds = live_price_fetcher.polygon_data_service
        self.trade_pair_to_price_info = {}
        self.trade_pair_to_position_ret = {}

        self.random_security_screenings = set()
        self.market_calendar = UnifiedMarketCalendar()
        self.n_api_calls = 0
        self.POLYGON_MAX_CANDLE_LIMIT = 49999
        self.UPDATE_LOOKBACK_MS = 600000  # 10 minutes ago. Want to give Polygon time to create candles on the backend.
        self.UPDATE_LOOKBACK_S = self.UPDATE_LOOKBACK_MS // 1000
        self.now_ms = 0
        self.n_price_corrections = 0
        self.elimination_rows = []
        self.hk_to_last_order_processed_ms = {}
        self.position_uuid_to_cache = {}
        self.cc = CacheController(metagraph=metagraph)

    def refresh_allowed(self, refresh_interval_ms):
        return TimeUtil.now_in_millis() - self._last_update_time_ms > refresh_interval_ms

    def set_last_update_time(self, skip_message=False):
        # Log that the class has finished updating and the time it finished updating
        if not skip_message:
            bt.logging.success(f"Finished updating class {self.__class__.__name__}")
        self._last_update_time_ms = TimeUtil.now_in_millis()

    @periodic_heartbeat(interval=600, message="perf ledger run_update_loop still running...")
    def run_update_loop(self):
        while not self.shutdown_dict:
            try:
                if self.refresh_allowed(ValiConfig.PERF_LEDGER_REFRESH_TIME_MS):
                    self.update()
                    self.set_last_update_time(skip_message=True)

            except Exception as e:
                bt.logging.error(f"Error during perf ledger update: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc()[1000:])
                time.sleep(30)
            time.sleep(1)

    cpdef tuple get_historical_position(self, object position, long timestamp_ms):
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
        if len(new_orders) == len(position.orders) and position.return_at_close == 0:
            temp_pos_to_pop.return_at_close = 0
            temp_pos_to_pop.close_out_position(position.close_ms)

        return temp_pos, temp_pos_to_pop

    def generate_order_timeline(self, positions: list, now_ms: int) -> list[tuple] :
        time_sorted_orders = []
        for p in positions:
            if p.is_closed_position and len(p.orders) < 2:
                bt.logging.info(f"perf ledger generate_order_timeline. Skipping closed position with < 2 orders: {p}")
                continue
            for o in p.orders:
                if o.processed_ms <= now_ms:
                    time_sorted_orders.append((o, p))
        time_sorted_orders.sort(key=lambda x: x[0].processed_ms)
        return time_sorted_orders

    cpdef bint replay_all_closed_positions(self, str miner_hotkey, dict tp_to_historical_positions):
        max_cuml_return_so_far = 1.0
        cuml_return = 1.0
        n_closed_positions = 0

        for _, positions in tp_to_historical_positions.items():
            for position in positions:
                if position.is_open_position:
                    continue
                cuml_return *= position.return_at_close
                n_closed_positions += 1
                max_cuml_return_so_far = max(cuml_return, max_cuml_return_so_far)

        return max_cuml_return_so_far

    cpdef tuple _can_shortcut(self, dict tp_to_historical_positions, long end_time_ms, object realtime_position_to_pop):
        portfolio_value = 1.0
        portfolio_spread_fee = 1.0
        portfolio_carry_fee = 1.0
        n_open_positions = 0
        if end_time_ms < TimeUtil.now_in_millis() - TARGET_LEDGER_WINDOW_MS:
            for tp, historical_positions in tp_to_historical_positions.items():
                for i, historical_position in enumerate(historical_positions):
                    if tp == realtime_position_to_pop.trade_pair.trade_pair and i == len(historical_positions) - 1 and realtime_position_to_pop:
                        historical_position = realtime_position_to_pop
                    if historical_position.position_uuid not in self.position_uuid_to_cache:
                        self.position_uuid_to_cache[historical_position.position_uuid] = FeeCache()
                    portfolio_spread_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                    portfolio_carry_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(end_time_ms, historical_position)
                    portfolio_value *= historical_position.return_at_close
                    n_open_positions += historical_position.is_open_position
            assert portfolio_carry_fee > 0, (portfolio_carry_fee, portfolio_spread_fee)
            return True, portfolio_value, portfolio_spread_fee, portfolio_carry_fee

        n_positions = 0
        n_closed_positions = 0
        n_positions_newly_opened = 0
        for tp, historical_positions in tp_to_historical_positions.items():
            for historical_position in historical_positions:
                n_positions += 1
                if historical_position.is_closed_position:
                    portfolio_value *= historical_position.return_at_close
                    if historical_position.position_uuid not in self.position_uuid_to_cache:
                        self.position_uuid_to_cache[historical_position.position_uuid] = FeeCache()
                    portfolio_spread_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                    portfolio_carry_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(end_time_ms, historical_position)
                    n_closed_positions += 1
                elif len(historical_position.orders) == 0:
                    n_positions_newly_opened += 1
                n_open_positions += historical_position.is_open_position

        ans = (n_positions == n_closed_positions + n_positions_newly_opened) and (n_positions_newly_opened == 1)
        ans |= (n_positions == n_closed_positions)
        return ans, portfolio_value, portfolio_spread_fee, portfolio_carry_fee

    cpdef bint new_window_intersects_old_window(self, long start_time_s, long end_time_s, long existing_lb_s, long existing_ub_s):
        return start_time_s <= existing_ub_s and end_time_s >= existing_lb_s

    cpdef void refresh_price_info(self, long t_ms, long end_time_ms, object tp):
        t_s = t_ms // 1000
        existing_lb_s = None
        existing_ub_s = None
        existing_window_s = None
        if tp.trade_pair in self.trade_pair_to_price_info:
            price_info = self.trade_pair_to_price_info[tp.trade_pair]
            existing_ub_s = price_info['ub_s']
            existing_lb_s = price_info['lb_s']
            existing_window_s = existing_ub_s - existing_lb_s
            if existing_lb_s <= t_s <= existing_ub_s:
                return

        start_time_s = t_s

        end_time_s = end_time_ms // 1000
        requested_seconds = end_time_s - start_time_s
        if requested_seconds > self.POLYGON_MAX_CANDLE_LIMIT:
            end_time_s = start_time_s + self.POLYGON_MAX_CANDLE_LIMIT
        elif requested_seconds < 3600:
            end_time_s = start_time_s + 3600

        end_time_s = min(self.now_ms // 1000, end_time_s)

        start_time_ms = start_time_s * 1000
        end_time_ms = end_time_s * 1000

        price_info, lb_ms, ub_ms = self.pds.get_candles_for_trade_pair_simple(
            trade_pair=tp, start_timestamp_ms=start_time_ms, end_timestamp_ms=end_time_ms)
        self.n_api_calls += 1

        assert lb_ms >= start_time_ms, (lb_ms, start_time_ms)
        assert ub_ms <= end_time_ms, (ub_ms, end_time_ms)
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

    cpdef tuple positions_to_portfolio_return(self, dict tp_to_historical_positions_dense, long t_ms, str miner_hotkey, long end_time_ms, float portfolio_return, float portfolio_spread_fee, float portfolio_carry_fee):
        t_s = t_ms // 1000
        any_open = False

        for tp, historical_positions in tp_to_historical_positions_dense.items():
            for historical_position in historical_positions:
                if self.shutdown_dict:
                    return portfolio_return, any_open, portfolio_spread_fee, portfolio_carry_fee
                if historical_position.position_uuid not in self.position_uuid_to_cache:
                    self.position_uuid_to_cache[historical_position.position_uuid] = FeeCache()
                position_spread_fee = self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                position_carry_fee = self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(t_ms, historical_position)
                portfolio_spread_fee *= position_spread_fee
                portfolio_carry_fee *= position_carry_fee

                if not self.market_calendar.is_market_open(historical_position.trade_pair, t_ms):
                    portfolio_return *= historical_position.return_at_close
                    continue

                any_open = True
                self.refresh_price_info(t_ms, end_time_ms, historical_position.trade_pair)
                price_at_t_s = self.trade_pair_to_price_info[tp][t_s] if t_s in self.trade_pair_to_price_info[tp] else None
                if price_at_t_s is not None:
                    self.tp_to_last_price[tp] = price_at_t_s
                    if t_ms == historical_position.orders[-1].processed_ms and historical_position.orders[-1].price != price_at_t_s:
                        self.n_price_corrections += 1
                        historical_position.orders[-1].price = price_at_t_s
                        historical_position.rebuild_position_with_updated_orders()
                    historical_position.set_returns(price_at_t_s, time_ms=t_ms, total_fees=position_spread_fee * position_carry_fee)
                    self.trade_pair_to_position_ret[tp] = historical_position.return_at_close

                portfolio_return *= historical_position.return_at_close
        return portfolio_return, any_open, portfolio_spread_fee, portfolio_carry_fee

    cpdef float update_mdd(self, str miner_hotkey, float portfolio_return, PerfLedgerData perf_ledger):
        perf_ledger.max_return = max(perf_ledger.max_return, portfolio_return)
        dd = 1.0 + (portfolio_return - perf_ledger.max_return) / perf_ledger.max_return
        return dd

    cpdef bint check_liquidated(self, str miner_hotkey, float portfolio_return, long t_ms, dict tp_to_historical_positions, PerfLedgerData perf_ledger):
        if portfolio_return == 0:
            bt.logging.warning(f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_ms}. Eliminating miner.")
            elimination_row = CacheController.generate_elimination_row(miner_hotkey, 0.0, 'LIQUIDATED', t_ms=t_ms,
                                        price_info=self.tp_to_last_price,
                                        return_info={'dd_stats':{}, 'returns': self.trade_pair_to_position_ret})
            self.elimination_rows.append(elimination_row)
            for _, v in tp_to_historical_positions.items():
                for pos in v:
                    print(
                        f"    time {TimeUtil.millis_to_formatted_date_str(t_ms)} hk {miner_hotkey[-5:]} {pos.trade_pair.trade_pair} return {pos.current_return} return_at_close {pos.return_at_close} closed@{'NA' if pos.is_open_position else TimeUtil.millis_to_formatted_date_str(pos.orders[-1].processed_ms)}")
            return True
        return False

    cpdef void init_tp_to_last_price(self, dict tp_to_historical_positions):
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

    cpdef tuple condense_positions(self, dict tp_to_historical_positions):
        portfolio_return = 1.0
        portfolio_spread_fee = 1.0
        portfolio_carry_fee = 1.0
        tp_to_historical_positions_dense = {}
        for tp, historical_positions in tp_to_historical_positions.items():
            dense_positions = []
            for historical_position in historical_positions:
                if historical_position.is_closed_position:
                    portfolio_return *= historical_position.return_at_close
                    if historical_position.position_uuid not in self.position_uuid_to_cache:
                        self.position_uuid_to_cache[historical_position.position_uuid] = FeeCache()
                    portfolio_spread_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                    portfolio_carry_fee *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(historical_position.orders[-1].processed_ms, historical_position)
                elif len(historical_position.orders) == 0:
                    continue
                else:
                    dense_positions.append(historical_position)
            tp_to_historical_positions_dense[tp] = dense_positions
        return portfolio_return, portfolio_spread_fee, portfolio_carry_fee, tp_to_historical_positions_dense

    cpdef bint build_perf_ledger(self, PerfLedgerData perf_ledger, dict tp_to_historical_positions, long start_time_ms, long end_time_ms, str miner_hotkey, object realtime_position_to_pop):
        if start_time_ms == 0:
            perf_ledger.init_with_first_order(end_time_ms, point_in_time_dd=1.0)
            return False
        if start_time_ms == end_time_ms:
            return False

        can_shortcut, portfolio_return, portfolio_spread_fee, portfolio_carry_fee = \
            self._can_shortcut(tp_to_historical_positions, end_time_ms, realtime_position_to_pop)
        if can_shortcut:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, False, -1, portfolio_spread_fee, portfolio_carry_fee)
            return False

        any_update = any_open = False
        last_dd = None
        self.init_tp_to_last_price(tp_to_historical_positions)
        initial_portfolio_return, initial_portfolio_spread_fee, initial_portfolio_carry_fee, tp_to_historical_positions_dense = self.condense_positions(tp_to_historical_positions)
        for t_ms in range(start_time_ms, end_time_ms, 1000):
            if self.shutdown_dict:
                return False
            assert t_ms >= perf_ledger.last_update_ms(), f"t_ms: {t_ms}, last_update_ms: {perf_ledger.last_update_ms()}, delta_s: {(t_ms - perf_ledger.last_update_ms()) // 1000} s. perf ledger {perf_ledger}"
            portfolio_return, any_open, portfolio_spread_fee, portfolio_carry_fee = self.positions_to_portfolio_return(tp_to_historical_positions_dense, t_ms, miner_hotkey, end_time_ms, initial_portfolio_return, initial_portfolio_spread_fee, initial_portfolio_carry_fee)
            if portfolio_return == 0 and self.check_liquidated(miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions, perf_ledger):
                return True
            perf_ledger.max_return = max(perf_ledger.max_return, portfolio_return)
            last_dd = 1.0 + (portfolio_return - perf_ledger.max_return) / perf_ledger.max_return

            perf_ledger.update(portfolio_return, t_ms, miner_hotkey, any_open, last_dd, portfolio_spread_fee, portfolio_carry_fee)
            any_update = True

        if any_update and perf_ledger.last_update_ms() != end_time_ms:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, any_open, last_dd, portfolio_spread_fee, portfolio_carry_fee)
        return False

    cpdef void update_one_perf_ledger(self, int hotkey_i, int n_hotkeys, str hotkey, list positions, long now_ms, dict existing_perf_ledgers):
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
        self.n_price_corrections = 0

        tp_to_historical_positions = {}
        sorted_timeline = self.generate_order_timeline(positions, now_ms)
        last_event_time = sorted_timeline[-1][0].processed_ms if sorted_timeline else 0
        self.hk_to_last_order_processed_ms[hotkey] = last_event_time

        building_from_new_orders = True

        pl_last_update_ms = perf_ledger.last_update_ms()
        if last_event_time <= pl_last_update_ms:
            building_from_new_orders = False

        realtime_position_to_pop = None
        for event in sorted_timeline:
            if realtime_position_to_pop:
                symbol = realtime_position_to_pop.trade_pair.trade_pair
                tp_to_historical_positions[symbol][-1] = realtime_position_to_pop
                if realtime_position_to_pop.return_at_close == 0:
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
                if symbol not in tp_to_historical_positions:
                    tp_to_historical_positions[symbol] = []
                tp_to_historical_positions[symbol].append(pos)

            n_open_positions = 0
            n_closed_positions = 0

            for p in tp_to_historical_positions[symbol]:
                if p.is_open_position:
                    n_open_positions += 1
                else:
                    n_closed_positions += 1
            assert n_open_positions == 0 or n_open_positions == 1, (
            n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])
            if n_open_positions == 1:
                assert tp_to_historical_positions[symbol][-1].is_open_position, (
                n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])

            if not building_from_new_orders:
                continue

            if order.processed_ms < perf_ledger.last_update_ms():
                continue
            eliminated = self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms(),
                                                order.processed_ms, hotkey, realtime_position_to_pop)
            if eliminated:
                break

        if eliminated:
            return
        if realtime_position_to_pop:
            symbol = realtime_position_to_pop.trade_pair.trade_pair
            tp_to_historical_positions[symbol][-1] = realtime_position_to_pop
        if now_ms > perf_ledger.last_update_ms():
            self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms(), now_ms, hotkey,
                                   None)

        lag = (TimeUtil.now_in_millis() - perf_ledger.last_update_ms()) // 1000
        total_product = perf_ledger.get_total_product()
        last_portfolio_value = perf_ledger.prev_portfolio_ret
        if verbose:
            bt.logging.info(
                f"Done updating perf ledger for {hotkey} {hotkey_i + 1}/{n_hotkeys} in {time.time() - t0} "
                f"(s). Lag: {lag} (s). Total product: {total_product}. Last portfolio value: {last_portfolio_value}."
                f" n_api_calls: {self.n_api_calls} dd stats {None}. n_price_corrections {self.n_price_corrections}"
                f" last cp {perf_ledger.cps[-1]}. perf_ledger_mpv {perf_ledger.max_return}")

    def write_perf_ledger_eliminations_to_disk(self, eliminations):
        bt.logging.trace(f"Writing [{len(eliminations)}] eliminations from memory to disk: {eliminations}")
        output_location = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=False)
        ValiBkpUtils.write_file(output_location, eliminations)

    cpdef dict update_all_perf_ledgers(self, dict hotkey_to_positions, dict existing_perf_ledgers, long now_ms, bint return_dict=False):
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
    def load_perf_ledgers_from_disk(read_as_pydantic=False):
        file_path = ValiBkpUtils.get_perf_ledgers_path()
        if not os.path.exists(file_path):
            return {}

        with open(file_path, 'r') as file:
            data = json.load(file)

        perf_ledgers = {}

        for key, ledger_data in data.items():
            #if read_as_pydantic:
            #    ledger_data['cps'] = [PerfCheckpoint(**cp) for cp in ledger_data['cps']]
            #    perf_ledgers[key] = PerfLedger(**ledger_data)
            #else:
            ledger_data['cps'] = [PerfCheckpointData(**cp) for cp in ledger_data['cps']]
            perf_ledgers[key] = PerfLedgerData(**ledger_data)

        return perf_ledgers

    def get_positions_perf_ledger(self, testing_one_hotkey=None, order_by_latest_trade=False) -> dict:
        if testing_one_hotkey:
            hotkey_to_positions = self.cc.get_all_miner_positions_by_hotkey(
                [testing_one_hotkey], sort_positions=True
            )
        else:
            hotkey_to_positions = self.cc.get_all_miner_positions_by_hotkey(
                self.metagraph.hotkeys, sort_positions=True,
                eliminations=self.eliminations
            )
            hotkeys_with_no_positions = set()
            for k, positions in hotkey_to_positions.items():
                if len(positions) == 0:
                    hotkeys_with_no_positions.add(k)
            for k in hotkeys_with_no_positions:
                del hotkey_to_positions[k]

        if order_by_latest_trade:
            def sort_key(x):
                return hotkey_to_positions[x][-1].orders[-1].processed_ms

            keys = sorted(hotkey_to_positions.keys(), key=sort_key, reverse=True)
            hotkey_to_positions = {k: hotkey_to_positions[k] for k in keys}
        return hotkey_to_positions

    cpdef dict generate_perf_ledgers_for_analysis(self, dict hotkey_to_positions, long t_ms=0):
        if not t_ms:
            t_ms = TimeUtil.now_in_millis()
        existing_perf_ledgers = {}
        ans_data = self.update_all_perf_ledgers(hotkey_to_positions, existing_perf_ledgers, t_ms, return_dict=True)
        return ans_data

    def get_eliminations_from_disk(self):
        location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=False)
        cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
        bt.logging.trace(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
        return cached_eliminations


    def get_filtered_eliminations_from_disk(self):
        # Filters out miners that have already been deregistered. (Not in the metagraph)
        # This allows the miner to participate again once they re-register
        cached_eliminations = self.cc.get_eliminations_from_disk()
        updated_eliminations = [elimination for elimination in cached_eliminations if
                                elimination['hotkey'] in self.metagraph.hotkeys]
        if len(updated_eliminations) != len(cached_eliminations):
            bt.logging.info(f"Filtered [{len(cached_eliminations) - len(updated_eliminations)}] / "
                            f"{len(cached_eliminations)} eliminations from disk due to not being in the metagraph")
        return updated_eliminations

    def get_eliminated_hotkeys(self):
        return set([x['hotkey'] for x in self.eliminations]) if self.eliminations else set()

    cpdef void update(self, str testing_one_hotkey=None, bint regenerate_all_ledgers=False):
        perf_ledgers = PerfLedgerManager.load_perf_ledgers_from_disk(read_as_pydantic=False)
        self.cc._refresh_eliminations_in_memory()
        self.eliminations = self.cc.eliminations
        t_ms = TimeUtil.now_in_millis() - self.UPDATE_LOOKBACK_MS
        hotkey_to_positions = self.get_positions_perf_ledger(testing_one_hotkey=testing_one_hotkey, order_by_latest_trade=True)

        eliminated_hotkeys = self.get_eliminated_hotkeys()

        metagraph_hotkeys = set(self.metagraph.hotkeys)
        hotkeys_to_delete = set()
        rss_modified = False

        for hotkey in hotkey_to_positions.keys():
            if hotkey not in metagraph_hotkeys:
                hotkeys_to_delete.add(hotkey)
            elif hotkey in eliminated_hotkeys:
                pass
            elif not len(hotkey_to_positions.get(hotkey, [])):
                hotkeys_to_delete.add(hotkey)
            elif not rss_modified and hotkey not in self.random_security_screenings:
                rss_modified = True
                self.random_security_screenings.add(hotkey)
                hotkeys_to_delete.add(hotkey)

        if not rss_modified:
            self.random_security_screenings = set()

        attempting_invalidations = bool(self.position_syncer) and bool(self.position_syncer.perf_ledger_hks_to_invalidate)
        if attempting_invalidations:
            for hk, t in self.position_syncer.perf_ledger_hks_to_invalidate.items():
                hotkeys_to_delete.add(hk)

        perf_ledgers = {k: v for k, v in perf_ledgers.items() if k not in hotkeys_to_delete}

        bt.logging.info(f"perf ledger PLM hotkeys to delete: {hotkeys_to_delete}. rss: {self.random_security_screenings}")

        if regenerate_all_ledgers or testing_one_hotkey:
            bt.logging.info("Regenerating all perf ledgers")
            perf_ledgers = {}

        self.trim_ledgers(perf_ledgers, hotkey_to_positions)

        self.update_all_perf_ledgers(hotkey_to_positions, perf_ledgers, t_ms, return_dict=bool(testing_one_hotkey))

        if attempting_invalidations:
            self.position_syncer.perf_ledger_hks_to_invalidate = {}

        if testing_one_hotkey:
            ledger = perf_ledgers[testing_one_hotkey]
            for x in ledger.cps:
                last_update_formated = TimeUtil.millis_to_timestamp(x.last_update_ms)
                print(x, last_update_formated)

    @staticmethod
    def save_perf_ledgers_to_disk(perf_ledgers:dict, raw_json=False):
        perf_ledgers = {key: value.to_dict() for key, value in perf_ledgers.items()}
        file_path = ValiBkpUtils.get_perf_ledgers_path()
        ValiBkpUtils.write_to_dir(file_path, perf_ledgers)

    cpdef void print_perf_ledgers_on_disk(self):
        perf_ledgers = self.load_perf_ledgers_from_disk()
        for hotkey, perf_ledger in perf_ledgers.items():
            print(f"perf ledger for {hotkey}")
            print('    total gain product', perf_ledger.get_product_of_gains())
            print('    total loss product', perf_ledger.get_product_of_loss())
            print('    total product', perf_ledger.get_total_product())

    cpdef void trim_ledgers(self, dict perf_ledgers, dict hotkey_to_positions):
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
                    if last_acked_order_time_ms < o.processed_ms < ledger_last_update_time:
                        order_time_str = TimeUtil.millis_to_formatted_date_str(o.processed_ms)
                        last_acked_time_str = TimeUtil.millis_to_formatted_date_str(last_acked_order_time_ms)
                        ledger_last_update_time_str = TimeUtil.millis_to_formatted_date_str(ledger_last_update_time)
                        bt.logging.info(f"Trimming checkpoints for {hk}. Order came in at {order_time_str} after last acked time {last_acked_time_str} but before perf ledger update time {ledger_last_update_time_str}")
                        ledger.trim_checkpoints(o.processed_ms)

class MockMetagraph():
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys

def main():
    bt.logging.enable_default()
    all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    all_hotkeys_on_disk = CacheController.get_directory_names(all_miners_dir)
    mmg = MockMetagraph(hotkeys=all_hotkeys_on_disk)
    perf_ledger_manager = PerfLedgerManager(metagraph=mmg, running_unit_tests=False)
    #perf_ledger_manager.update(testing_one_hotkey='5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X')
    perf_ledger_manager.update(regenerate_all_ledgers=True)

if __name__ == "__main__":
    main()