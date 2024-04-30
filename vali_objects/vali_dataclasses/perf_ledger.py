import json
import math
import os
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from json import JSONDecodeError
from typing import List

import bittensor as bt

from pydantic import BaseModel

from shared_objects.cache_controller import CacheController
from shared_objects.retry import retry
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
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

    def __str__(self):
        return self.to_json_string()

    def to_json_string(self) -> str:
        # Using pydantic's json method with built-in validation
        json_str = self.json()
        # Unfortunately, we can't tell pydantic v1 to strip certain fields so we do that here
        json_loaded = json.loads(json_str)
        return json.dumps(json_loaded)


class PerfLedger(BaseModel):
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

    def create_cps_to_fill_void(self, time_since_last_update_ms: int, now_ms:int):
        original_accum_time = self.cps[-1].accum_ms
        delta_accum_time_ms = self.target_cp_duration_ms - original_accum_time
        self.cps[-1].accum_ms += delta_accum_time_ms
        self.cps[-1].last_update_ms += delta_accum_time_ms
        time_since_last_update_ms -= delta_accum_time_ms
        assert time_since_last_update_ms >= 0, (self.cps, time_since_last_update_ms)
        while time_since_last_update_ms > self.target_cp_duration_ms:
            new_cp = PerfCheckpoint(last_update_ms=self.cps[-1].last_update_ms + self.target_cp_duration_ms,
                                    prev_portfolio_ret=self.cps[-1].prev_portfolio_ret,
                                    accum_ms=self.target_cp_duration_ms)
            assert new_cp.last_update_ms < now_ms, (self.cps, (now_ms - new_cp.last_update_ms))
            self.cps.append(new_cp)
            time_since_last_update_ms -= self.target_cp_duration_ms

        assert time_since_last_update_ms >= 0
        new_cp = PerfCheckpoint(last_update_ms=self.cps[-1].last_update_ms,
                                prev_portfolio_ret=self.cps[-1].prev_portfolio_ret)
        assert new_cp.last_update_ms <= now_ms, self.cps
        self.cps.append(new_cp)

    def init_with_first_order(self, order_processed_ms: int):
        new_cp = PerfCheckpoint(last_update_ms=order_processed_ms, prev_portfolio_ret=1.0)
        self.cps.append(new_cp)

    def get_or_create_latest_cp(self, now_ms: int, miner_hotkey:str):
        if len(self.cps) == 0:
            self.init_with_first_order(now_ms)
            return self.cps[-1]

        time_since_last_update_ms = now_ms - self.cps[-1].last_update_ms
        assert time_since_last_update_ms >= 0, self.cps
        if time_since_last_update_ms + self.cps[-1].accum_ms >= self.target_cp_duration_ms:
            self.create_cps_to_fill_void(time_since_last_update_ms, now_ms)

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

    def update(self, current_portfolio_value:float, now_ms:int, miner_hotkey:str, any_open:bool):
        current_cp = self.get_or_create_latest_cp(now_ms, miner_hotkey)
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
    def __init__(self, metagraph, live_price_fetcher=None, running_unit_tests=False):
        super().__init__(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.stop_requested = False
        if live_price_fetcher is None:
            secrets = ValiUtils.get_secrets()
            live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
            self.pds = live_price_fetcher.polygon_data_service
        else:
            self.pds = live_price_fetcher.polygon_data_service

    def run_update_loop(self):
        while not self.stop_requested:
            try:
                if self.refresh_allowed(ValiConfig.PERF_LEDGER_REFRESH_TIME_MS):
                    self.update()
                    self.set_last_update_time(skip_message=True)

            except Exception as e:
                # Handle exceptions or log errors
                bt.logging.error(f"Error during perf ledger update: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
            time.sleep(1)

    def stop_update_loop(self):
        self.stop_requested = True

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

    def _can_shortcut(self, tp_to_historical_positions: dict[str: Position], start_time_ms, end_time_ms):
        n_positions = 0
        n_closed_positions = 0
        n_positions_newly_opened = 0
        portfolio_value = 1.0
        for tp, historical_positions in tp_to_historical_positions.items():
            for historical_position in historical_positions:
                n_positions += 1
                if historical_position.is_closed_position:
                    portfolio_value *= historical_position.return_at_close
                    n_closed_positions += 1
                elif len(historical_position.orders) == 0:
                    n_positions_newly_opened += 1

        ans = (n_positions == n_closed_positions + n_positions_newly_opened) and (n_positions_newly_opened == 1)
        #if ans:
        #    window_s = (end_time_ms - start_time_ms) // 1000
        #    #print(f"Shortcutting. n_positions: {n_positions}, n_closed_positions: {n_closed_positions}, n_positions_newly_opened: {n_positions_newly_opened}, window_s: {window_s}")
        return ans, portfolio_value



    def refresh_price_info(self, trade_pair_to_price_info, t_ms, end_time_ms, tp):
        t_s = t_ms // 1000
        if tp.trade_pair in trade_pair_to_price_info:
            price_info = trade_pair_to_price_info[tp.trade_pair]
            assert t_s >= price_info['lb_s'], (t_s, price_info['lb_s'])
            if t_s <= price_info['ub_s']:  # No refresh needed
                return
        #else:
        #    print('11111', tp.trade_pair, trade_pair_to_price_info.keys())

        start_time_s = t_s
        end_time_s = end_time_ms // 1000
        requested_seconds = end_time_s - start_time_s
        if requested_seconds > 50000:  # Polygon limit
            end_time_s = start_time_s + 50000

        start_time_ms = start_time_s * 1000
        end_time_ms = end_time_s * 1000

        #t0 = time.time()
        #print(f"Starting #{requested_seconds} candle fetch for {tp.trade_pair}")
        price_info, lb_ms, ub_ms = self.pds.get_candles_for_trade_pair_simple(
            trade_pair=tp, start_timestamp_ms=start_time_ms, end_timestamp_ms=end_time_ms)

        #print(f'Got {len(price_info)} candles after request of {requested_seconds} candles for tp {tp.trade_pair} in {time.time() - t0}s')

        assert lb_ms >= start_time_ms, (lb_ms, start_time_ms)
        assert ub_ms <= end_time_ms, (ub_ms, end_time_ms)

        trade_pair_to_price_info[tp.trade_pair] = price_info
        trade_pair_to_price_info[tp.trade_pair]['lb_s'] = start_time_s
        trade_pair_to_price_info[tp.trade_pair]['ub_s'] = end_time_s
        #print(f'Fetched {requested_seconds} s of candles for tp {tp.trade_pair} in {time.time() - t0}s')
        #print('22222', tp.trade_pair, trade_pair_to_price_info.keys())

    def positions_to_portfolio_return(self, tp_to_historical_positions: dict[str: Position], t_ms, trade_pair_to_price_info, miner_hotkey, end_time_ms):
        # What is the portfolio return at this time t_ms?
        portfolio_return = 1.0
        t_s = t_ms // 1000
        any_open = False
        for tp, historical_positions in tp_to_historical_positions.items():  # TODO: multithread over trade pairs?
            for historical_position in historical_positions:
                if len(historical_position.orders) == 0:  # Just opened an order. We will revisit this on the next event as there is no history to replay
                    continue
                if historical_position.is_closed_position:  # We want to process just-closed positions. wont be closed if we are on the corresponding event
                    portfolio_return *= historical_position.return_at_close
                    continue
                any_open = True
                self.refresh_price_info(trade_pair_to_price_info, t_ms, end_time_ms, historical_position.trade_pair)
                price_at_t_s = trade_pair_to_price_info[tp].get(t_s)
                if price_at_t_s is None:
                    # Use the last portfolio return
                    opr = historical_position.return_at_close
                else:
                    opr = historical_position.get_open_position_return_with_fees(price_at_t_s, t_ms)
                    historical_position.return_at_close = opr
                portfolio_return *= opr
                #assert portfolio_return > 0, f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_s}. opr {opr} rtp {price_at_t_s}, historical position {historical_position}"
        return portfolio_return, any_open

    def build_perf_ledger(self, perf_ledger:PerfLedger, tp_to_historical_positions: dict[str: Position], start_time_ms, end_time_ms, miner_hotkey):
        #print(f"Building perf ledger for {miner_hotkey} from {start_time_ms} to {end_time_ms} ({(end_time_ms - start_time_ms) // 1000} s) order {order}")
        if start_time_ms == 0:  # This ledger is being initialized. First order received.
            perf_ledger.init_with_first_order(end_time_ms)
            return
        if start_time_ms == end_time_ms:  # No new orders since last update. Shouldn't happen
            return
        # "Shortcut" All positions closed and one newly open position.
        can_shortcut, portfolio_return = self._can_shortcut(tp_to_historical_positions, start_time_ms, end_time_ms)
        if can_shortcut:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, False)
            return

        trade_pair_to_price_info = {}
        any_update = any_open = False
        for t_ms in range(start_time_ms, end_time_ms, 1000):
            assert t_ms >= perf_ledger.last_update_ms, f"t_ms: {t_ms}, last_update_ms: {perf_ledger.last_update_ms}, delta_s: {(t_ms - perf_ledger.last_update_ms) // 1000} s. perf ledger {perf_ledger}"
            portfolio_return, any_open = self.positions_to_portfolio_return(tp_to_historical_positions, t_ms, trade_pair_to_price_info, miner_hotkey, end_time_ms)
            assert portfolio_return > 0, f"Portfolio value is {portfolio_return} for miner {miner_hotkey} at {t_ms // 1000}. perf ledger {perf_ledger}"
            perf_ledger.update(portfolio_return, t_ms, miner_hotkey, any_open)
            any_update = True

        # Get last sliver of time
        if any_update and perf_ledger.last_update_ms != end_time_ms:
            perf_ledger.update(portfolio_return, end_time_ms, miner_hotkey, any_open)

    def update_all_perf_ledgers(self, hotkey_to_positions: dict[str, List[Position]], existing_perf_ledgers: dict[str, PerfLedger], now_ms: int):
        t_init = time.time()
        for hotkey_i, (hotkey, positions) in enumerate(hotkey_to_positions.items()):
            t0 = time.time()
            perf_ledger = existing_perf_ledgers.get(hotkey, PerfLedger())
            existing_perf_ledgers[hotkey] = perf_ledger
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
                #print(f"Processing order {order}")
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
                self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms, order.processed_ms, hotkey)
                #print(f"Done processing order {order}. perf ledger {perf_ledger}")

            # We have processed all orders. Need to catch up to now_ms
            if now_ms > perf_ledger.last_update_ms:
                self.build_perf_ledger(perf_ledger, tp_to_historical_positions, perf_ledger.last_update_ms, now_ms, hotkey)

            PerfLedgerManager.save_perf_ledgers_to_disk(existing_perf_ledgers)
            lag = (TimeUtil.now_in_millis() - perf_ledger.last_update_ms) // 1000
            bt.logging.info(
                f"Done updating perf ledger for {hotkey} {hotkey_i}/ {len(hotkey_to_positions)} in {time.time() - t0} (s). Lag: {lag} (s)")

        bt.logging.info(f"Done updating perf ledger for all hotkeys in {time.time() - t_init} s")



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

    @retry(tries=10, delay=1, backoff=1)
    def get_positions_with_retry(self, testing_one_hotkey=None):
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
        t_ms = TimeUtil.now_in_millis() - 600000  # 10 minutes ago
        if t_ms < 1714438182000 + 1000 * 60 * 60 * 1:  # Rebuild after bug fix
            perf_ledgers = {}
        hotkey_to_positions = self.get_positions_with_retry(testing_one_hotkey=testing_one_hotkey)
        # Time in the past to start updating the perf ledgers
        self.update_all_perf_ledgers(hotkey_to_positions, perf_ledgers, t_ms)

    @staticmethod
    def save_perf_ledgers_to_disk(perf_ledgers: dict[str, PerfLedger]):
        file_path = ValiBkpUtils.get_perf_ledgers_path()
        with open(file_path, 'w') as f:
            json.dump(perf_ledgers, f, cls=CustomEncoder)

    def print_perf_ledgers_on_disk(self):
        perf_ledgers = self.load_perf_ledgers_from_disk()
        for hotkey, perf_ledger in perf_ledgers.items():
            print(f"perf ledger for {hotkey}")
            print('    total gain product', perf_ledger.get_product_of_gains())
            print('    total loss product', perf_ledger.get_product_of_loss())
            print('    total product', perf_ledger.get_total_product())


if __name__ == "__main__":
    perf_ledger_manager = PerfLedgerManager(metagraph=None, running_unit_tests=False)
