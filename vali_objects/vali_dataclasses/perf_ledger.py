import json
import math
import os
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import List
import bittensor as bt
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS

from pydantic import BaseModel

from shared_objects.cache_controller import CacheController
from shared_objects.retry import periodic_heartbeat
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_objects.vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

TARGET_CHECKPOINT_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
TARGET_LEDGER_WINDOW_MS = ValiConfig.TARGET_LEDGER_WINDOW_MS

TP_ID_PORTFOLIO = 'portfolio'
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
        elif position.trade_pair.is_forex or position.trade_pair.is_indices or position.trade_pair.is_equities:
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

# Enum class TradePairReturnStatus with 3 options 1. TP_MARKET_NOT_OPEN, TP_MARKET_OPEN_NO_PRICE_CHANGE, TP_MARKET_OPEN_PRICE_CHANGE
class TradePairReturnStatus(Enum):
    TP_NO_OPEN_POSITIONS = 0
    TP_MARKET_NOT_OPEN = 1
    TP_MARKET_OPEN_NO_PRICE_CHANGE = 2
    TP_MARKET_OPEN_PRICE_CHANGE = 3

    # Define greater than oeprator for TradePairReturnStatus
    def __gt__(self, other):
        return self.value > other.value

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
    def lowerbound_time_created_ms(self):
        # accum_ms boundary alignment makes this a lowerbound for the first cp.
        return self.last_update_ms - self.accum_ms

class PerfLedgerData:
    def __init__(self, initialization_time_ms: int=0, max_return=1.0, target_cp_duration_ms=TARGET_CHECKPOINT_DURATION_MS, target_ledger_window_ms=TARGET_LEDGER_WINDOW_MS, cps=None):
        if cps is None:
            cps = []
        self.max_return = max_return
        self.target_cp_duration_ms = target_cp_duration_ms
        self.target_ledger_window_ms = target_ledger_window_ms
        self.initialization_time_ms = initialization_time_ms
        self.cps = cps

    def to_dict(self):
        return {
            "initialization_time_ms": self.initialization_time_ms,
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
        elif self.initialization_time_ms != 0:  # 0 default value for old ledgers that haven't rebuilt as of this update.
            return self.initialization_time_ms
        else:
            return self.cps[0].lowerbound_time_created_ms  # legacy calculation that will stop being used in ~24 hrs

    def init_max_portfolio_value(self):
        if self.cps:
            self.max_return = max(x.mpv for x in self.cps)
        # Initial portfolio value is 1.0
        self.max_return = max(self.max_return, 1.0)

    def create_cps_to_fill_void(self, time_since_last_update_ms: int, now_ms: int, point_in_time_dd: float, any_open: TradePairReturnStatus):
        original_accum_time = self.cps[-1].accum_ms
        delta_accum_time_ms = self.target_cp_duration_ms - original_accum_time
        self.cps[-1].accum_ms += delta_accum_time_ms
        self.cps[-1].last_update_ms += delta_accum_time_ms
        if any_open > TradePairReturnStatus.TP_MARKET_NOT_OPEN:
            self.cps[-1].open_ms += delta_accum_time_ms
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

    def init_with_first_order(self, order_processed_ms: int, point_in_time_dd: float, current_portfolio_value: float,  current_portfolio_fee_spread:float, current_portfolio_carry:float):
        # figure out how many ms we want to initalize the checkpoint with so that once self.target_cp_duration_ms is
        # reached, the CP ends at 00:00:00 UTC or 12:00:00 UTC (12 hr cp case). This may change based on self.target_cp_duration_ms
        # |----x------midday-----------| -> accum_ms_for_utc_alignment = (distance between start of day and x) = x - start_of_day_ms
        # |-----------midday-----x-----| -> accum_ms_for_utc_alignment = (distance between midday and x) = x - midday_ms
        # By calculating the initial accum_ms this way, the co will always end at middday or 00:00:00 the next day.


        datetime_representation = TimeUtil.millis_to_datetime(order_processed_ms)
        assert self.target_cp_duration_ms == 43200000, f'self.target_cp_duration_ms is not 12 hours {self.target_cp_duration_ms}'
        midday = datetime_representation.replace(hour=12, minute=0, second=0, microsecond=0)
        midday_ms = int(midday.timestamp() * 1000)
        if order_processed_ms < midday_ms:
            start_of_day = datetime_representation.replace(hour=0, minute=0, second=0, microsecond=0)
            start_of_day_ms = int(start_of_day.timestamp() * 1000)
            accum_ms_for_utc_alignment = order_processed_ms - start_of_day_ms
        else:
            accum_ms_for_utc_alignment = order_processed_ms - midday_ms

        new_cp = PerfCheckpointData(last_update_ms=order_processed_ms, prev_portfolio_ret=current_portfolio_value,
                                    mdd=point_in_time_dd, prev_portfolio_spread_fee=current_portfolio_fee_spread,
                                    prev_portfolio_carry_fee=current_portfolio_carry, accum_ms=accum_ms_for_utc_alignment, mpv=1.0)
        self.cps.append(new_cp)

    def get_or_create_latest_cp_with_mdd(self, now_ms: int, current_portfolio_value:float, current_portfolio_fee_spread:float, current_portfolio_carry:float, any_open: TradePairReturnStatus) -> PerfCheckpointData:
        point_in_time_dd = CacheController.calculate_drawdown(current_portfolio_value, self.max_return)

        assert point_in_time_dd, point_in_time_dd

        if len(self.cps) == 0:
            self.init_with_first_order(now_ms, point_in_time_dd, current_portfolio_value, current_portfolio_fee_spread, current_portfolio_carry)
            return self.cps[-1]

        time_since_last_update_ms = now_ms - self.cps[-1].last_update_ms
        assert time_since_last_update_ms >= 0, self.cps
        if time_since_last_update_ms + self.cps[-1].accum_ms > self.target_cp_duration_ms:
            self.create_cps_to_fill_void(time_since_last_update_ms, now_ms, point_in_time_dd, any_open)
        else:
            self.cps[-1].mdd = min(self.cps[-1].mdd, point_in_time_dd)

        return self.cps[-1]

    def update_accumulated_time(self, cp: PerfCheckpointData, now_ms: int, miner_hotkey: str, any_open: TradePairReturnStatus, tp_debug):
        accumulated_time = now_ms - cp.last_update_ms
        if accumulated_time < 0:
            bt.logging.error(f"Negative accumulated time: {accumulated_time} for miner {miner_hotkey}."
                             f" start_time_ms: {self.start_time_ms}, now_ms: {now_ms}")
            accumulated_time = 0
        cp.accum_ms += accumulated_time
        cp.last_update_ms = now_ms
        if any_open == TradePairReturnStatus.TP_NO_OPEN_POSITIONS or any_open == TradePairReturnStatus.TP_MARKET_NOT_OPEN:
            pass
            #print(f' {any_open} Blocked {accumulated_time} ms of open time for miner {miner_hotkey}. Time {TimeUtil.millis_to_verbose_formatted_date_str(now_ms)} tp_debug {tp_debug}')
        else:
            cp.open_ms += accumulated_time

    def compute_delta_between_ticks(self, cur: float, prev: float):
        return math.log(cur / prev)

    def update_gains_losses(self, current_cp: PerfCheckpointData, current_portfolio_value: float,
                            current_portfolio_fee_spread: float, current_portfolio_carry: float, miner_hotkey: str, any_open: TradePairReturnStatus):
        # TODO: leave as is?
        # current_portfolio_value = current_portfolio_value * current_portfolio_carry  # spread fee already applied
        if any_open == TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE:
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
        else:
            n_new_updates = 0

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
                f"Purging old perf cp {self.cps[0]}. Total ledger duration: {self.get_total_ledger_duration_ms()}. Target ledger window: {self.target_ledger_window_ms}")
            self.cps = self.cps[1:]  # Drop the first cp (oldest)

    def trim_checkpoints(self, cutoff_ms: int):
        new_cps = []
        for cp in self.cps:
            if cp.lowerbound_time_created_ms + self.target_cp_duration_ms >= cutoff_ms:
                continue
            new_cps.append(cp)
        self.cps = new_cps

    def update_pl(self, current_portfolio_value: float, now_ms: int, miner_hotkey: str, any_open: TradePairReturnStatus,
              current_portfolio_fee_spread: float, current_portfolio_carry: float, tp_debug=None):
        if len(self.cps) ==  0:
            self.init_with_first_order(now_ms, point_in_time_dd=1.0, current_portfolio_value=1.0,
                                           current_portfolio_fee_spread=1.0, current_portfolio_carry=1.0)
        self.max_return = max(self.max_return, current_portfolio_value)
        current_cp = self.get_or_create_latest_cp_with_mdd(now_ms, current_portfolio_value, current_portfolio_fee_spread, current_portfolio_carry, any_open)
        self.update_gains_losses(current_cp, current_portfolio_value, current_portfolio_fee_spread,
                                 current_portfolio_carry, miner_hotkey, any_open)
        self.update_accumulated_time(current_cp, now_ms, miner_hotkey, any_open, tp_debug)


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
    initialization_time_ms: int = 0
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
                   target_ledger_window_ms=data.target_ledger_window_ms, cps=cps,
                   initialization_time_ms=data.initialization_time_ms)

    @classmethod
    def from_dict(self, data: dict):
        return PerfLedger(max_return=data['max_return'], target_cp_duration_ms=data['target_cp_duration_ms'],
                          target_ledger_window_ms=data['target_ledger_window_ms'], cps=data['cps'],
                          initialization_time_ms=data.get('initialization_time_ms', 0))


class PerfLedgerManager(CacheController):
    def __init__(self, metagraph, live_price_fetcher=None, running_unit_tests=False, shutdown_dict=None, position_syncer=None):
        super().__init__(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.shutdown_dict = shutdown_dict
        self.position_syncer = position_syncer
        self.live_price_fetcher = live_price_fetcher
        self.running_unit_tests = running_unit_tests

        if live_price_fetcher is None:
            secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
            live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
            self.pds = live_price_fetcher.polygon_data_service
        else:
            self.pds = live_price_fetcher.polygon_data_service
        # Every update, pick a hotkey to rebuild in case polygon 1s candle data changed.
        self.trade_pair_to_price_info = {'second':{}, 'minute':{}}
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
        self.hotkey_to_checkpointed_ledger = {}


    def load_perf_ledgers_from_disk(self, read_as_pydantic=True) -> dict[str, dict[str, PerfLedgerData]]:
        file_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)
        if not os.path.exists(file_path):
            return {}

        with open(file_path, 'r') as file:
            data = json.load(file)

        perf_ledgers = {}
        for hk, ledger_data in data.items():
            perf_ledgers[hk] = {}
            if 'initialization_time_ms' in ledger_data:
                # V1 Perf ledger skip.
                continue
            for key, ledger in ledger_data.items():
                if read_as_pydantic:
                    perf_ledgers[hk][key] = PerfLedger(**ledger)
                else:
                    perf_ledgers[hk][key] = PerfLedgerData(**ledger)

        return perf_ledgers

    def write_perf_ledgers_to_disk(self, perf_ledgers: dict[str, PerfLedgerData], write_as_pydantic=True):
        file_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)

        data = {}
        for key, ledger in perf_ledgers.items():
            ledger_data = ledger.to_dict() if write_as_pydantic else ledger.to_dict()
            ledger_data['cps'] = [cp.dict() if write_as_pydantic else cp.to_dict() for cp in ledger_data['cps']]
            data[key] = ledger_data

        ValiBkpUtils.write_file(file_path, data)

    def clear_perf_ledgers_from_disk(self):
        file_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)
        if os.path.exists(file_path):
            ValiBkpUtils.write_file(file_path, {})

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
                time.sleep(30)
            time.sleep(1)

    def get_historical_position(self, position: Position, timestamp_ms: int):
        hk = position.miner_hotkey  # noqa: F841

        new_orders = []
        position_at_start_timestamp = deepcopy(position)
        position_at_end_timestamp = deepcopy(position)
        for o in position.orders:
            if o.processed_ms <= timestamp_ms:
                new_orders.append(o)

        position_at_start_timestamp.orders = new_orders[:-1]
        position_at_start_timestamp.rebuild_position_with_updated_orders()
        position_at_end_timestamp.orders = new_orders
        position_at_end_timestamp.rebuild_position_with_updated_orders()
        # Handle position that was forced closed due to realtime data (liquidated)
        if len(new_orders) == len(position.orders) and position.return_at_close == 0:
            position_at_end_timestamp.return_at_close = 0
            position_at_end_timestamp.close_out_position(position.close_ms)

        return position_at_start_timestamp, position_at_end_timestamp

    def generate_order_timeline(self, positions: list[Position], now_ms: int, hk: str) -> (list[tuple], int):
        # order to understand timestamps needing checking, position to understand returns per timestamp (will be adjusted)
        # (order, position)
        time_sorted_orders = []
        last_event_time_ms = 0

        for p in positions:
            last_event_time_ms = max(p.orders[-1].processed_ms, last_event_time_ms)

            if p.is_closed_position and len(p.orders) < 2:
                bt.logging.info(f"perf ledger generate_order_timeline. Skipping closed position for hk {hk} with < 2 orders: {p}")
                continue
            for o in p.orders:
                if o.processed_ms <= now_ms:
                    time_sorted_orders.append((o, p))
        # sort
        time_sorted_orders.sort(key=lambda x: x[0].processed_ms)
        return time_sorted_orders, last_event_time_ms


    def _can_shortcut(self, tp_to_historical_positions: dict[str: Position], end_time_ms: int,
                      realtime_position_to_pop: Position | None, start_time_ms: int, perf_ledger_bundle: dict[str, PerfLedgerData]) -> (bool, float, float, float):

        tp_to_return = defaultdict(lambda: 1.0)
        tp_to_spread_fee = defaultdict(lambda: 1.0)
        tp_to_carry_fee = defaultdict(lambda: 1.0)
        n_open_positions = 0
        now_ms = TimeUtil.now_in_millis()
        ledger_cutoff_ms = now_ms - perf_ledger_bundle[TP_ID_PORTFOLIO].target_ledger_window_ms

        n_positions = 0
        n_closed_positions = 0
        n_positions_newly_opened = 0

        for tp_id, historical_positions in tp_to_historical_positions.items():
            for i, historical_position in enumerate(historical_positions):
                n_positions += 1
                if len(historical_position.orders) == 0:
                    n_positions_newly_opened += 1
                if realtime_position_to_pop and tp_id == realtime_position_to_pop.trade_pair.trade_pair_id and i == len(historical_positions) - 1:
                    historical_position = realtime_position_to_pop

                for tp_id in [TP_ID_PORTFOLIO, tp_id]:
                    tp_to_spread_fee[tp_id] *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                    tp_to_carry_fee[tp_id] *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(end_time_ms, historical_position)
                    tp_to_return[tp_id] *= historical_position.return_at_close
                n_open_positions += historical_position.is_open_position
                n_closed_positions += historical_position.is_closed_position
                self.trade_pair_to_position_ret[tp_id] = historical_position.return_at_close
        assert tp_to_carry_fee[TP_ID_PORTFOLIO] > 0, (tp_to_carry_fee[TP_ID_PORTFOLIO], tp_to_spread_fee[TP_ID_PORTFOLIO])

        reason = ''
        ans = False
        # When building from orders, we will always have at least one open position. When opening a position after a
        # period of all closed positions, we can shortcut by identifying that the new position is the only open position
        # and all other positions are closed. The time before this period, we have only closed positions.
        # Alternatively, we can be attempting to build the ledger after all orders have been accounted for. In this
        # case, we simply need to check if all positions are closed.
        if (n_positions == n_closed_positions + n_positions_newly_opened) and (n_positions_newly_opened == 1):
            reason += 'New position opened. '
            ans = True

        # This window would be dropped anyway
        if (end_time_ms < ledger_cutoff_ms):
            reason += 'Ledger cutoff. '
            ans = True

        if 0 and ans:
            for tp, historical_positions in tp_to_historical_positions.items():
                positions = []
                for i, historical_position in enumerate(historical_positions):
                    if realtime_position_to_pop and tp == realtime_position_to_pop.trade_pair.trade_pair and i == len(
                            historical_positions) - 1:
                        historical_position = realtime_position_to_pop
                        foo = True
                    else:
                        foo = False
                    positions.append((historical_position.position_uuid, [x.price for x in historical_position.orders],
                                      historical_position.return_at_close, foo, historical_position.is_open_position))
                print(f'{tp}: {positions}')

            final_cp = None
            if perf_ledger_bundle and TP_ID_PORTFOLIO in perf_ledger_bundle and perf_ledger_bundle[TP_ID_PORTFOLIO].cps:
                final_cp = perf_ledger_bundle[TP_ID_PORTFOLIO].cps[-1]
            print('---------------------------------------------------------------------')
            print(f' Skipping ({reason}) with n_positions: {n_positions} n_open_positions: {n_open_positions} n_closed_positions: '
                  f'{n_closed_positions}, n_positions_newly_opened: {n_positions_newly_opened}, '
                  f'start_time_ms: {TimeUtil.millis_to_formatted_date_str(start_time_ms)} ({start_time_ms}) , '
                  f'end_time_ms: {TimeUtil.millis_to_formatted_date_str(end_time_ms)} ({end_time_ms}) , '
                  f'portfolio_value: {tp_to_return[TP_ID_PORTFOLIO]} '
                  f'ledger_cutoff_ms: {TimeUtil.millis_to_formatted_date_str(ledger_cutoff_ms)}, '
                  f'realtime_position_to_pop.trade_pair.trade_pair: {realtime_position_to_pop.trade_pair.trade_pair if realtime_position_to_pop else None}, '
                  f'trade_pair_to_position_ret: {self.trade_pair_to_position_ret} '
                  f'final portfolio cp {final_cp}')
            print('---------------------------------------------------------------------')

        return ans, tp_to_return, tp_to_spread_fee, tp_to_carry_fee, start_time_ms, end_time_ms


    def new_window_intersects_old_window(self, start_time_ms, end_time_ms, existing_lb_ms, existing_ub_ms):
        # Check if new window intersects with the old window
        # An intersection occurs if the start of the new window is before the end of the old window,
        # and the end of the new window is after the start of the old window
        return start_time_ms <= existing_ub_ms and end_time_ms >= existing_lb_ms

    def align_t_ms_to_mode(self, t_ms, mode):
        if mode == 'second':
            return t_ms - (t_ms % 1000)
        elif mode == 'minute':
            return t_ms - (t_ms % 60000)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def refresh_price_info(self, t_ms, end_time_ms, tp, mode, min_candles_per_request=3600):
        t_ms = self.align_t_ms_to_mode(t_ms, mode)

        existing_lb_ms = None
        existing_ub_ms = None
        existing_window_ms = None
        if tp.trade_pair_id in self.trade_pair_to_price_info[mode]:
            price_info = self.trade_pair_to_price_info[mode][tp.trade_pair_id]
            existing_ub_ms = price_info['ub_ms']
            existing_lb_ms = price_info['lb_ms']
            existing_window_ms = existing_ub_ms - existing_lb_ms
            if existing_lb_ms <= t_ms <= existing_ub_ms:  # No refresh needed
                return
        #else:
        #    print('11111', tp.trade_pair, trade_pair_to_price_info.keys())

        start_time_ms = t_ms
        requested_milliseconds = end_time_ms - start_time_ms
        n_candles_requested = requested_milliseconds // 1000 if mode == 'second' else requested_milliseconds // 60000
        if n_candles_requested > self.POLYGON_MAX_CANDLE_LIMIT:  # Polygon limit
            end_time_ms = start_time_ms + self.POLYGON_MAX_CANDLE_LIMIT * 1000 if mode == 'second' else start_time_ms + self.POLYGON_MAX_CANDLE_LIMIT * 60000
        elif n_candles_requested < min_candles_per_request:  # Get a batch of candles to minimize number of fetches
            offset = min_candles_per_request * 1000 if mode == 'second' else min_candles_per_request * 60000
            end_time_ms = start_time_ms + offset

        end_time_ms = min(int(self.now_ms * 1000), end_time_ms)  # Don't fetch candles beyond check time or will fill in null.

        #t0 = time.time()
        #print(f"Starting #{requested_seconds} candle fetch for {tp.trade_pair}")
        price_info_raw = self.pds.get_candles_for_trade_pair_simple(
            trade_pair=tp, start_timestamp_ms=start_time_ms, end_timestamp_ms=end_time_ms, timespan=mode)
        self.n_api_calls += 1
        #print(f'Fetched candles for tp {tp.trade_pair} for window {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to {TimeUtil.millis_to_formatted_date_str(end_time_ms)}')
        #print(f'Got {len(price_info)} candles after request of {requested_seconds} candles for tp {tp.trade_pair} in {time.time() - t0}s')

        #assert lb_ms >= start_time_ms, (lb_ms, start_time_ms)
        #assert ub_ms <= end_time_ms, (ub_ms, end_time_ms)
        # Can we build on top of existing data or should we wipe?
        perform_wipe = True
        if tp.trade_pair_id in self.trade_pair_to_price_info:
            new_window_size_ms = end_time_ms - start_time_ms
            candidate_window_size = new_window_size_ms + existing_window_ms
            candidate_n_candles_in_memory = candidate_window_size // 1000 if mode == 'second' else candidate_window_size // 60000
            if candidate_n_candles_in_memory < self.POLYGON_MAX_CANDLE_LIMIT and \
                    self.new_window_intersects_old_window(start_time_ms, end_time_ms, existing_lb_ms, existing_ub_ms):
                perform_wipe = False

        attr = 'close' if mode == 'second' else 'vwap' # TODO: @@@@@ use low if within x% of wvap. else use vwap. Do a study on optimal x.
        if perform_wipe:
            price_info = {self.align_t_ms_to_mode(a.timestamp, mode): getattr(a, attr) for a in price_info_raw}
            #for a in price_info_raw:
            #    price_info[a.timestamp // 1000] = a.close
            self.trade_pair_to_price_info[mode][tp.trade_pair_id] = price_info
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['lb_ms'] = start_time_ms
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['ub_ms'] = end_time_ms
        else:
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['ub_ms'] = max(existing_ub_ms, end_time_ms)
            self.trade_pair_to_price_info[mode][tp.trade_pair_id]['lb_ms'] = min(existing_lb_ms, start_time_ms)
            for a in price_info_raw:
                self.trade_pair_to_price_info[mode][tp.trade_pair_id][self.align_t_ms_to_mode(a.timestamp, mode)] = getattr(a, attr)

        #print(f'Fetched {requested_seconds} s of candles for tp {tp.trade_pair} in {time.time() - t0}s')
        #print('22222', tp.trade_pair, trade_pair_to_price_info.keys())

    def positions_to_portfolio_return(self, tp_to_historical_positions_dense: dict[str: Position], t_ms, mode, end_time_ms, tp_to_initial_return, tp_to_initial_spread_fee, tp_to_initial_carry_fee):
        # Answers "What is the portfolio return at this time t_ms?"
        all_tp_ids = list(tp_to_historical_positions_dense.keys()) + [TP_ID_PORTFOLIO]
        tp_to_any_open = {x: TradePairReturnStatus.TP_NO_OPEN_POSITIONS for x in all_tp_ids}
        tp_to_return = deepcopy(tp_to_initial_return)
        tp_to_spread_fee = deepcopy(tp_to_initial_spread_fee)
        tp_to_carry_fee = deepcopy(tp_to_initial_carry_fee)
        t_ms = self.align_t_ms_to_mode(t_ms, mode)
        for tp_id, historical_positions in tp_to_historical_positions_dense.items():  # TODO: multithread over trade pairs?
            for historical_position in historical_positions:
                if self.shutdown_dict:
                    return tp_to_return, tp_to_any_open, tp_to_spread_fee, tp_to_carry_fee

                position_spread_fee = self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                position_carry_fee = self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(t_ms, historical_position)
                tp_to_spread_fee[tp_id] *= position_spread_fee
                tp_to_carry_fee[tp_id] *= position_carry_fee

                if not self.market_calendar.is_market_open(historical_position.trade_pair, t_ms):
                    tp_to_return[tp_id] *= historical_position.return_at_close
                    tp_to_return[TP_ID_PORTFOLIO] *= historical_position.return_at_close
                    tp_to_any_open[tp_id] = TradePairReturnStatus.TP_MARKET_NOT_OPEN
                    tp_to_any_open[TP_ID_PORTFOLIO] = max(TradePairReturnStatus.TP_MARKET_NOT_OPEN, tp_to_any_open[TP_ID_PORTFOLIO])
                    continue

                tp_to_any_open[tp_id] = TradePairReturnStatus.TP_MARKET_OPEN_NO_PRICE_CHANGE
                tp_to_any_open[TP_ID_PORTFOLIO] = max(TradePairReturnStatus.TP_MARKET_OPEN_NO_PRICE_CHANGE, tp_to_any_open[TP_ID_PORTFOLIO])
                self.refresh_price_info(t_ms, end_time_ms, historical_position.trade_pair, mode)
                price_at_t_ms = self.trade_pair_to_price_info[mode][tp_id][t_ms] if t_ms in self.trade_pair_to_price_info[mode][tp_id] else None
                if price_at_t_ms is not None:
                    price_changed = price_at_t_ms != self.tp_to_last_price.get(tp_id, None)
                    if price_changed:
                        tp_to_any_open[tp_id] = TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE
                        tp_to_any_open[TP_ID_PORTFOLIO] = TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE
                    self.tp_to_last_price[tp_id] = price_at_t_ms
                    # We are retoractively updating the last order's price if it is the same as the candle price. This is needed to get a smoothed PNL curve for price data across potentially various data providers and data availability.
                    if t_ms == historical_position.orders[-1].processed_ms and historical_position.orders[-1].price != price_at_t_ms:
                        self.n_price_corrections += 1
                        #percent_change = (price_at_t_s - historical_position.orders[-1].price) / historical_position.orders[-1].price
                        #bt.logging.warning(f"Price at t_s {TimeUtil.millis_to_formatted_date_str(t_ms)} {historical_position.trade_pair.trade_pair} is the same as the last order processed time. Changing price from {historical_position.orders[-1].price} to {price_at_t_s}. percent change {percent_change}")
                        historical_position.orders[-1].price = price_at_t_ms
                        historical_position.rebuild_position_with_updated_orders()
                    historical_position.set_returns(price_at_t_ms, time_ms=t_ms, total_fees=position_spread_fee * position_carry_fee)
                    self.trade_pair_to_position_ret[tp_id] = historical_position.return_at_close

                tp_to_return[tp_id] *= historical_position.return_at_close
                tp_to_return[TP_ID_PORTFOLIO] *= historical_position.return_at_close
                #assert portfolio_return > 0, f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_s}. opr {opr} rtp {price_at_t_s}, historical position {historical_position}"

        return tp_to_return, tp_to_any_open, tp_to_spread_fee, tp_to_carry_fee



    def check_liquidated(self, miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions):
        if portfolio_return == 0:
            bt.logging.warning(f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_ms}. Eliminating miner.")
            elimination_row = self.generate_elimination_row(miner_hotkey, 0.0, 'LIQUIDATED', t_ms=t_ms, price_info=self.tp_to_last_price, return_info={'dd_stats': {}, 'returns': self.trade_pair_to_position_ret})
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
        all_tp_ids = list(tp_to_historical_positions.keys()) + [TP_ID_PORTFOLIO]
        tp_to_initial_return = {x: 1.0 for x in all_tp_ids}
        tp_to_initial_spread_fee = {x: 1.0 for x in all_tp_ids}
        tp_to_initial_carry_fee = {x: 1.0 for x in all_tp_ids}
        tp_to_historical_positions_dense = {}
        for tp_id, historical_positions in tp_to_historical_positions.items():
            dense_positions = []
            for historical_position in historical_positions:
                if historical_position.is_closed_position:
                    for x in [TP_ID_PORTFOLIO, tp_id]:
                        tp_to_initial_return[x] *= historical_position.return_at_close
                        tp_to_initial_spread_fee[x] *= self.position_uuid_to_cache[historical_position.position_uuid].get_spread_fee(historical_position)
                        tp_to_initial_carry_fee[x] *= self.position_uuid_to_cache[historical_position.position_uuid].get_carry_fee(historical_position.orders[-1].processed_ms, historical_position)
                elif len(historical_position.orders) == 0:
                    continue
                else:
                    dense_positions.append(historical_position)
            tp_to_historical_positions_dense[tp_id] = dense_positions
        return tp_to_initial_return, tp_to_initial_spread_fee, tp_to_initial_carry_fee, tp_to_historical_positions_dense

    def build_perf_ledger(self, perf_ledger_bundle: dict[str:dict[str, PerfLedgerData]], tp_to_historical_positions: dict[str: Position], start_time_ms, end_time_ms, miner_hotkey, realtime_position_to_pop) -> bool:
        #print(f"Building perf ledger for {miner_hotkey} from {start_time_ms} to {end_time_ms} ({(end_time_ms - start_time_ms) // 1000} s)")
        portfolio_pl = perf_ledger_bundle[TP_ID_PORTFOLIO]
        if len(portfolio_pl.cps) == 0:
            portfolio_pl.init_with_first_order(end_time_ms, point_in_time_dd=1.0, current_portfolio_value=1.0,
                                              current_portfolio_fee_spread=1.0, current_portfolio_carry=1.0)

        # Init per-trade-pair perf ledgers
        tp_ids_to_build = [TP_ID_PORTFOLIO]
        for i, (tp_id, positions) in enumerate(tp_to_historical_positions.items()):
            if tp_id in perf_ledger_bundle:
                # Can only build perf ledger between orders or after all orders have passed.
                tp_ids_to_build.append(tp_id)
            else:
                assert len(positions) == 1
                assert len(positions[0].orders) == 0
                assert realtime_position_to_pop and tp_id == realtime_position_to_pop.trade_pair.trade_pair_id
                relevant_position = realtime_position_to_pop
                initialization_time_ms = relevant_position.open_ms
                perf_ledger_bundle[tp_id] = PerfLedgerData(initialization_time_ms=initialization_time_ms)
                perf_ledger_bundle[tp_id].init_with_first_order(end_time_ms, point_in_time_dd=1.0, current_portfolio_value=1.0,
                                                   current_portfolio_fee_spread=1.0, current_portfolio_carry=1.0)

        if portfolio_pl.initialization_time_ms == end_time_ms:
            return False  # Can only build perf ledger between orders or after all orders have passed.

        # "Shortcut" All positions closed and one newly open position OR before the ledger lookback window.
        can_shortcut, tp_to_return, tp_to_spread_fee, tp_to_carry_fee, start_time_ms, end_time_ms = \
            self._can_shortcut(tp_to_historical_positions, end_time_ms, realtime_position_to_pop, start_time_ms, perf_ledger_bundle)
        if can_shortcut:
            for tp_id in tp_ids_to_build:
                perf_ledger = perf_ledger_bundle[tp_id]
                tp_return = tp_to_return[tp_id]
                tp_spread_fee = tp_to_spread_fee[tp_id]
                tp_carry_fee = tp_to_carry_fee[tp_id]
                perf_ledger.update_pl(tp_return, end_time_ms, miner_hotkey, TradePairReturnStatus.TP_MARKET_NOT_OPEN, tp_spread_fee, tp_carry_fee, tp_debug=tp_id + '_shortcut')
                perf_ledger.purge_old_cps()
            return False

        self.init_tp_to_last_price(tp_to_historical_positions)
        tp_to_initial_return, tp_to_initial_spread_fee, tp_to_initial_carry_fee, tp_to_historical_positions_dense = self.condense_positions(tp_to_historical_positions)
        # We avoided a shortcut. Any trade pairs from open positions (tp_to_historical_positions_dense) need to be in the ledgers bundle.

        # Default mode becomes minute if there are at least 7 minutes between start and end time
        if (end_time_ms - start_time_ms) > 420000:
            default_mode = 'minute'
        else:
            default_mode = 'second'

        #time_list = list(range(start_time_ms, end_time_ms, step_ms))
        accumulated_time_ms = 0
        #mode_to_ticks = {'second': 0, 'minute': 0}
        while start_time_ms + accumulated_time_ms < end_time_ms:
            # Need high resolution at the start and end of the time window
            mode = default_mode
            if default_mode == 'minute':
                candidate_t_ms = int((start_time_ms + accumulated_time_ms) // 1000) * 1000
                ms_from_minute_boundary = candidate_t_ms % 60000
                if ms_from_minute_boundary != 0:
                    mode = 'second'
                elif candidate_t_ms + 60000 >= end_time_ms:
                    mode = 'second'

            #mode_to_ticks[mode] += 1
            t_ms = start_time_ms + accumulated_time_ms

            if self.shutdown_dict:
                return False
            assert t_ms >= portfolio_pl.last_update_ms, (f"t_ms: {t_ms}, "
                                                         f"last_update_ms: {TimeUtil.millis_to_formatted_date_str(portfolio_pl.last_update_ms)},"
                                                         f"mode: {mode},"
                                                         f" delta_ms: {(t_ms - portfolio_pl.last_update_ms)} s. perf ledger {portfolio_pl}")

            tp_to_current_return, tp_to_any_open, tp_to_current_spread_fee, tp_to_current_carry_fee = self.positions_to_portfolio_return(tp_to_historical_positions_dense, t_ms, mode, end_time_ms, tp_to_initial_return, tp_to_initial_spread_fee, tp_to_initial_carry_fee)
            portfolio_return = tp_to_current_return[TP_ID_PORTFOLIO]
            if portfolio_return == 0 and self.check_liquidated(miner_hotkey, portfolio_return, t_ms, tp_to_historical_positions):
                return True

            if portfolio_return < perf_ledger_bundle[TP_ID_PORTFOLIO].cps[-1].prev_portfolio_ret * 0.98:
                time_since_last_update = t_ms - perf_ledger_bundle[TP_ID_PORTFOLIO].cps[-1].last_update_ms
                print(f'perf ledger for hk {miner_hotkey} significant return drop from {perf_ledger_bundle[TP_ID_PORTFOLIO].cps[-1].prev_portfolio_ret} to {portfolio_return} over {time_since_last_update} ms ({t_ms})', perf_ledger_bundle[TP_ID_PORTFOLIO].cps[-1].to_dict(), self.trade_pair_to_position_ret)
                for tp, historical_positions in tp_to_historical_positions.items():
                    positions = []
                    for historical_position in historical_positions:
                        positions.append((historical_position.position_uuid, [x.price for x in historical_position.orders], historical_position.return_at_close, historical_position.is_open_position))
                    print(f'    tp {tp} positions {positions}')

            for tp_id in tp_ids_to_build:
                perf_ledger = perf_ledger_bundle[tp_id]
                perf_ledger.update_pl(tp_to_current_return[tp_id], t_ms, miner_hotkey, tp_to_any_open[tp_id], tp_to_spread_fee[tp_id], tp_to_current_carry_fee[tp_id], tp_debug=tp_id)

            accumulated_time_ms += 1000 if mode == 'second' else 60000

        # Get last sliver of time
        for tp_id in tp_ids_to_build:
           perf_ledger = perf_ledger_bundle[tp_id]
           if perf_ledger.last_update_ms != end_time_ms:
               assert perf_ledger.last_update_ms < end_time_ms, (perf_ledger.last_update_ms, end_time_ms)
               perf_ledger.update_pl(tp_to_current_return[tp_id], end_time_ms, miner_hotkey, tp_to_any_open[tp_id], tp_to_spread_fee[tp_id], tp_to_current_carry_fee[tp_id])

        for tp_id in tp_ids_to_build:
            perf_ledger = perf_ledger_bundle[tp_id]
            perf_ledger.purge_old_cps()

        #n_minutes_between_intervals = (end_time_ms - start_time_ms) // 60000
        #print(f'Updated between {TimeUtil.millis_to_formatted_date_str(start_time_ms)} and {TimeUtil.millis_to_formatted_date_str(end_time_ms)} ({n_minutes_between_intervals} min). mode_to_ticks {mode_to_ticks}. Default mode {default_mode}')
        return False

    def update_one_perf_ledger_bundle(self, hotkey_i: int, n_hotkeys: int, hotkey: str, positions: List[Position], now_ms:int,
                                      existing_perf_ledger_bundles: dict[str, dict[str, PerfLedgerData]]) -> None:

        eliminated = False
        self.n_api_calls = 0

        t0 = time.time()
        perf_ledger_bundle_candidate = existing_perf_ledger_bundles.get(hotkey)
        if perf_ledger_bundle_candidate is None:
            first_order_time_ms = float('inf')
            for p in positions:
                first_order_time_ms = min(first_order_time_ms, p.orders[0].processed_ms)
            perf_ledger_bundle_candidate = {TP_ID_PORTFOLIO:
                PerfLedgerData(initialization_time_ms=first_order_time_ms if first_order_time_ms != float('inf') else 0)}
            verbose = True
        else:
            perf_ledger_bundle_candidate = deepcopy(perf_ledger_bundle_candidate)
            verbose = False

        for tp_id, perf_ledger in perf_ledger_bundle_candidate.items():
            perf_ledger.init_max_portfolio_value()

        self.trade_pair_to_position_ret = {}
        #if hotkey in self.hk_to_dd_stats:
        #    del self.hk_to_dd_stats[hotkey]
        self.n_price_corrections = 0

        tp_to_historical_positions = defaultdict(list)
        sorted_timeline, last_event_time_ms = self.generate_order_timeline(positions, now_ms, hotkey)  # Enforces our "now_ms" constraint
        self.hk_to_last_order_processed_ms[hotkey] = last_event_time_ms
        # There hasn't been a new order since the last update time. Just need to update for open positions
        building_from_new_orders = True
        if last_event_time_ms <= perf_ledger_bundle_candidate[TP_ID_PORTFOLIO].last_update_ms:
            building_from_new_orders = False
            # Preserve returns from realtime positions
            sorted_timeline = []
            tp_to_historical_positions = {}
            for p in positions:
                if p.trade_pair.trade_pair in tp_to_historical_positions:
                    tp_to_historical_positions[p.trade_pair.trade_pair].append(p)
                else:
                    tp_to_historical_positions[p.trade_pair.trade_pair] = [p]
            

        # Building for scratch or there have been order(s) since the last update time
        realtime_position_to_pop = None
        for event_idx, event in enumerate(sorted_timeline):
            if realtime_position_to_pop:
                symbol = realtime_position_to_pop.trade_pair.trade_pair_id
                tp_to_historical_positions[symbol][-1] = realtime_position_to_pop
                if realtime_position_to_pop.return_at_close == 0:  # liquidated
                    self.check_liquidated(hotkey, 0.0, realtime_position_to_pop.close_ms, tp_to_historical_positions)
                    eliminated = True
                    break

            order, position = event
            symbol = position.trade_pair.trade_pair_id
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

            assert n_open_positions == 0 or n_open_positions == 1, (n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])
            if n_open_positions == 1:
                assert tp_to_historical_positions[symbol][-1].is_open_position, (n_open_positions, n_closed_positions, [p for p in tp_to_historical_positions[symbol] if p.is_open_position])

            # Perf ledger is already built, we just need to run the above loop to build tp_to_historical_positions
            if not building_from_new_orders:
                continue

            # Building from a checkpoint ledger. Skip until we get to the new order(s). We are only running this to build up tp_to_historical_positions.
            portfolio_last_update_ms = perf_ledger_bundle_candidate[TP_ID_PORTFOLIO].last_update_ms
            if order.processed_ms < portfolio_last_update_ms:
                continue

            # Need to catch up from perf_ledger.last_update_ms to order.processed_ms
            eliminated = self.build_perf_ledger(perf_ledger_bundle_candidate, tp_to_historical_positions, portfolio_last_update_ms, order.processed_ms, hotkey, realtime_position_to_pop)
            if event_idx == len(sorted_timeline) - 1:
                self.hotkey_to_checkpointed_ledger[hotkey] = deepcopy(perf_ledger_bundle_candidate)

            if eliminated:
                break
            # print(f"Done processing order {order}. perf ledger {perf_ledger}")

        if eliminated:
            return
        # We have processed all orders. Need to catch up to now_ms
        if realtime_position_to_pop:
            symbol = realtime_position_to_pop.trade_pair.trade_pair_id
            tp_to_historical_positions[symbol][-1] = realtime_position_to_pop

        portfolio_perf_ledger = perf_ledger_bundle_candidate[TP_ID_PORTFOLIO]
        if now_ms > portfolio_perf_ledger.last_update_ms:
            portfolio_last_update_ms = portfolio_perf_ledger.last_update_ms
            self.build_perf_ledger(perf_ledger_bundle_candidate, tp_to_historical_positions,
                                   portfolio_last_update_ms, now_ms, hotkey,None)

        lag = (TimeUtil.now_in_millis() - portfolio_perf_ledger.last_update_ms) // 1000
        total_product = portfolio_perf_ledger.get_total_product()
        last_portfolio_value = portfolio_perf_ledger.prev_portfolio_ret
        if verbose:
            bt.logging.info(
                f"Done updating perf ledger for {hotkey} {hotkey_i + 1}/{n_hotkeys} in {time.time() - t0} "
                f"(s). Lag: {lag} (s). Total product: {total_product}. Last portfolio value: {last_portfolio_value}."
                f" n_api_calls: {self.n_api_calls} dd stats {None}. n_price_corrections {self.n_price_corrections}"
                f" last cp {portfolio_perf_ledger.cps[-1] if portfolio_perf_ledger.cps else None}. perf_ledger_mpv {portfolio_perf_ledger.max_return} "
                f"perf_ledger_initialization_time {TimeUtil.millis_to_formatted_date_str(portfolio_perf_ledger.initialization_time_ms)}")

        # Write candidate at the very end in case an exception leads to a partial update
        existing_perf_ledger_bundles[hotkey] = perf_ledger_bundle_candidate

    def update_all_perf_ledgers(self, hotkey_to_positions: dict[str, List[Position]],
                                existing_perf_ledgers: dict[str, dict[str, PerfLedgerData]],
                                now_ms: int, return_dict=False) -> None | dict[str, dict[str, PerfLedgerData]]:
        t_init = time.time()
        self.now_ms = now_ms
        self.elimination_rows = []
        n_hotkeys = len(hotkey_to_positions)
        for hotkey_i, (hotkey, positions) in enumerate(hotkey_to_positions.items()):
            try:
                self.update_one_perf_ledger_bundle(hotkey_i, n_hotkeys, hotkey, positions, now_ms, existing_perf_ledgers)
            except Exception as e:
                bt.logging.error(f"Error updating perf ledger for {hotkey}: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
                continue

        n_perf_ledgers = len(existing_perf_ledgers) if existing_perf_ledgers else 0
        n_hotkeys_with_positions = len(hotkey_to_positions) if hotkey_to_positions else 0
        bt.logging.info(f"Done updating perf ledger for all hotkeys in {time.time() - t_init} s. n_perf_ledgers {n_perf_ledgers}. n_hotkeys_with_positions {n_hotkeys_with_positions}")
        self.write_perf_ledger_eliminations_to_disk(self.elimination_rows)

        if self.shutdown_dict:
            return

        self.save_perf_ledgers_to_disk(existing_perf_ledgers)
        if return_dict:
            return existing_perf_ledgers

    def get_positions_perf_ledger(self, testing_one_hotkey=None):
        """
        Since we are running in our own thread, we need to retry in case positions are being written to simultaneously.
        """
        #testing_one_hotkey = '5GzYKUYSD5d7TJfK4jsawtmS2bZDgFuUYw8kdLdnEDxSykTU'
        hotkeys_with_no_positions = set()
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
            for k, positions in hotkey_to_positions.items():
                if len(positions) == 0:
                    hotkeys_with_no_positions.add(k)
            for k in hotkeys_with_no_positions:
                del hotkey_to_positions[k]

        return hotkey_to_positions, hotkeys_with_no_positions

    def generate_perf_ledgers_for_analysis(self, hotkey_to_positions: dict[str, List[Position]], t_ms: int = None) -> dict[str, dict[str, PerfLedger]]:
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()  # Time to build the perf ledgers up to. Goes back 30 days from this time.
        existing_perf_ledgers = {}
        dat = self.update_all_perf_ledgers(hotkey_to_positions, existing_perf_ledgers, t_ms, return_dict=True)
        ans_pydantic = {}
        for k, v in dat.items():
            ans_pydantic[k] = {}
            for tp_id, pl in v.items():
                ans_pydantic[k][tp_id] = PerfLedger.from_data(pl)
        return ans_pydantic

    def update(self, testing_one_hotkey=None, regenerate_all_ledgers=False):
        perf_ledger_bundles = self.load_perf_ledgers_from_disk(read_as_pydantic=False)
        self._refresh_eliminations_in_memory()
        t_ms = TimeUtil.now_in_millis() - self.UPDATE_LOOKBACK_MS
        """
        tt = 1734279788000
        if t_ms < tt + 1000 * 60 * 60 * 1:  # Rebuild after bug fix
            for ledger in perf_ledgers.values():
                try:
                    # 12 hrs ago
                    ledger.trim_checkpoints(tt - 12 * 60 * 60 * 1000)
                    print('trim successful', ledger)
                except Exception as e:
                    print('trim failed', e, ledger)
                    raise
        """
        hotkey_to_positions, hotkeys_with_no_positions = self.get_positions_perf_ledger(testing_one_hotkey=testing_one_hotkey)

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
        hotkeys_to_delete = set([x for x in hotkeys_with_no_positions if x in perf_ledger_bundles])
        rss_modified = False

        # Determine which hotkeys to remove from the perf ledger
        hotkeys_to_iterate = [x for x in hotkeys_ordered_by_last_trade if x in perf_ledger_bundles]
        for k in perf_ledger_bundles.keys():  # Some hotkeys may not be in the positions (old, bugged, etc.)
            if k not in hotkeys_to_iterate:
                hotkeys_to_iterate.append(k)

        for hotkey in hotkeys_to_iterate:
            if hotkey not in metagraph_hotkeys:
                hotkeys_to_delete.add(hotkey)
            elif hotkey in eliminated_hotkeys:  # eliminated hotkeys won't be in positions so they will stop updating. We will keep them in perf ledger for visualizing metrics in the dashboard.
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
                bt.logging.info(f"perf ledger invalidated for hk {hk} due to position sync at time {t}")

        perf_ledgers = {k: v for k, v in perf_ledger_bundles.items() if k not in hotkeys_to_delete}
        self.hk_to_last_order_processed_ms = {k: v for k, v in self.hk_to_last_order_processed_ms.items() if k not in hotkeys_to_delete}

        #hk_to_last_update_date = {k: TimeUtil.millis_to_formatted_date_str(v.last_update_ms)
        #                            if v.last_update_ms else 'N/A' for k, v in perf_ledgers.items()}

        bt.logging.info(f"perf ledger PLM hotkeys to delete: {hotkeys_to_delete}. rss: {self.random_security_screenings}")

        if regenerate_all_ledgers or testing_one_hotkey:
            bt.logging.info("Regenerating all perf ledgers")
            perf_ledger_bundles = {}

        self.restore_out_of_sync_ledgers(perf_ledgers, hotkey_to_positions)

        # Time in the past to start updating the perf ledgers
        self.update_all_perf_ledgers(hotkey_to_positions, perf_ledger_bundles, t_ms, return_dict=bool(testing_one_hotkey))

        # Clear invalidations after successful update. Prevent race condition by only clearing if we attempted invalidations.
        if attempting_invalidations:
            self.position_syncer.perf_ledger_hks_to_invalidate = {}

        if testing_one_hotkey:
            portfolio_ledger = perf_ledger_bundles[testing_one_hotkey][TP_ID_PORTFOLIO]
            # print all attributes except cps: Note ledger is an object
            print(f'Portfolio ledger attributes: initialization_time_ms {portfolio_ledger.initialization_time_ms},'
                    f' max_return {portfolio_ledger.max_return}')
            returns = []
            returns_muled = []
            times = []
            n_contributing_tps = []
            for i, x in enumerate(portfolio_ledger.cps):
                returns.append(x.prev_portfolio_ret)
                foo = 1.0
                n_contributing = 0
                for tp_id, ledger in perf_ledger_bundles[testing_one_hotkey].items():
                    if tp_id == TP_ID_PORTFOLIO:
                        continue
                    rele_cp = None
                    for y in ledger.cps:
                        if y.last_update_ms == x.last_update_ms:
                            rele_cp = y
                            break
                    if rele_cp:
                        n_contributing += 1
                        foo *= rele_cp.prev_portfolio_ret
                returns_muled.append(foo)
                n_contributing_tps.append(n_contributing)
                times.append(TimeUtil.millis_to_timestamp(x.last_update_ms))

                last_update_formated = TimeUtil.millis_to_timestamp(x.last_update_ms)
                # assert the checkpoint ends on a 12 hour boundary
                if i != len(portfolio_ledger.cps) - 1:
                    assert x.last_update_ms % portfolio_ledger.target_cp_duration_ms == 0, x.last_update_ms
                print(x, last_update_formated)
            # Plot time vs return using matplotlib as well as time vs dd. use a legend.
            import matplotlib.pyplot as plt
            # Make the plot bigger
            plt.figure(figsize=(10, 5))
            plt.plot(times, returns, color='red', label='Return')
            plt.plot(times, returns_muled, color='blue', label='Return_Mulled')
            # Labels
            plt.xlabel('Time')
            plt.title(f'Return vs Time for HK {testing_one_hotkey}')
            plt.legend(['Return', 'Return_Mulled'])
            plt.show()

            for tp_id, pl in perf_ledger_bundles[testing_one_hotkey].items():
                print(f"perf ledger for {tp_id} last cp {pl.cps[-1]}")
                print('    total gain product', pl.get_product_of_gains())
                print('    total loss product', pl.get_product_of_loss())
                print('    total product', pl.get_total_product())

            print('validating returns:')
            for z in zip(returns, returns_muled, n_contributing_tps):
                print(z, z[0] - z[1])

    def save_perf_ledgers_to_disk(self, perf_ledgers: dict[str, dict[str, PerfLedgerData]] | dict[str, dict], raw_json=False):
        # Convert to PerfLedger (pydantic validation)
        pydantic_perf_ledgers = {}
        for hk, dat in perf_ledgers.items():
            pydantic_perf_ledgers[hk] = {}
            for tp_id, pl in dat.items():
                pydantic_perf_ledgers[hk][tp_id] = PerfLedger.from_dict(pl) if raw_json else PerfLedger.from_data(pl)

        file_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)
        ValiBkpUtils.write_to_dir(file_path, pydantic_perf_ledgers)

    def print_perf_ledgers_on_disk(self):
        perf_ledgers = self.load_perf_ledgers_from_disk()
        for hotkey, perf_ledger in perf_ledgers.items():
            print(f"perf ledger for {hotkey}")
            print('    total gain product', perf_ledger.get_product_of_gains())
            print('    total loss product', perf_ledger.get_product_of_loss())
            print('    total product', perf_ledger.get_total_product())

    def restore_out_of_sync_ledgers(self, existing_perf_ledgers, hotkey_to_positions):
        """
        REstore ledgers subject to race condition. Perf ledger fully update loop can take 30 min.
        An order can come in during update.

        We can only build perf ledgers between orders or after all orders
        """
        hotkeys_needing_recovery = []
        for hk, ledger in existing_perf_ledgers.items():
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
                        bt.logging.info(f"Recovering checkpoints for {hk}. Order came in at {order_time_str} after last acked time {last_acked_time_str} but before perf ledger update time {ledger_last_update_time_str}")
                        hotkeys_needing_recovery.append(hk)


        n_hotkeys_recovered = 0
        for hk in hotkeys_needing_recovery:
            if hk in self.hotkey_to_checkpointed_ledger:
                existing_perf_ledgers[hk] = self.hotkey_to_checkpointed_ledger[hk]
                n_hotkeys_recovered += 1
            else:
                existing_perf_ledgers.pop(hk)  # Full regeneration needed
        if hotkeys_needing_recovery:
            bt.logging.info(f"Recovered {n_hotkeys_recovered} / {len(hotkeys_needing_recovery)} perf ledgers for out of sync hotkeys")


class MockMetagraph():
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys


if __name__ == "__main__":
    bt.logging.enable_info()
    all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    all_hotkeys_on_disk = CacheController.get_directory_names(all_miners_dir)
    mmg = MockMetagraph(hotkeys=all_hotkeys_on_disk)
    perf_ledger_manager = PerfLedgerManager(metagraph=mmg, running_unit_tests=False)
    #perf_ledger_manager.update(testing_one_hotkey='5Ct5amT9YmnfaksGbcZepFnL95N8D59gWStybSvcXGR3RLmv')

    perf_ledger_manager.update(regenerate_all_ledgers=True)
