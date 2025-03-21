# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import shutil
import time
import traceback
from collections import defaultdict
from pickle import UnpicklingError
from typing import List, Dict
import bittensor as bt
from pathlib import Path

from copy import deepcopy
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil, timeme
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.positions_to_snap import positions_to_snap
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import OrderStatus, ORDER_SRC_DEPRECATION_FLAT, Order
from vali_objects.utils.position_filtering import PositionFiltering

TARGET_MS = 1742577204000 + (1000 * 60 * 60 * 3)  # + 3 hours


class PositionManager(CacheController):
    def __init__(self, metagraph=None, running_unit_tests=False,
                 perform_order_corrections=False,
                 perform_compaction=False,
                 is_mothership=False, perf_ledger_manager=None,
                 challengeperiod_manager=None,
                 elimination_manager=None,
                 secrets=None,
                 ipc_manager=None,
                 live_price_fetcher=None,
                 is_backtesting=False):

        super().__init__(metagraph=metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        # Populate memory with positions

        self.perf_ledger_manager = perf_ledger_manager
        self.challengeperiod_manager = challengeperiod_manager
        self.elimination_manager = elimination_manager

        self.recalibrated_position_uuids = set()

        self.is_mothership = is_mothership
        self.perform_compaction = perform_compaction
        self.perform_order_corrections = perform_order_corrections
        if ipc_manager:
            self.hotkey_to_positions = ipc_manager.dict()
        else:
            self.hotkey_to_positions = {}
        self.secrets = secrets
        self._populate_memory_positions_for_first_time()
        self.live_price_fetcher = live_price_fetcher

    @timeme
    def _populate_memory_positions_for_first_time(self):
        if self.is_backtesting:
            return

        initial_hk_to_positions = self.get_positions_for_all_miners(from_disk=True)
        for hk, positions in initial_hk_to_positions.items():
            if positions:  # Only populate if there are no positions in the miner dir
                self.hotkey_to_positions[hk] = positions

    def filtered_positions_for_scoring(
            self,
            hotkeys: List[str] = None
    ) -> (Dict[str, List[Position]], Dict[str, int]):
        """
        Filter the positions for a set of hotkeys.
        """
        if hotkeys is None:
            hotkeys = self.get_miner_hotkeys_with_at_least_one_position()

        hk_to_first_order_time = {}
        filtered_positions = {}
        for hotkey, miner_positions in self.get_positions_for_hotkeys(hotkeys, sort_positions=True).items():
            if miner_positions:
                hk_to_first_order_time[hotkey] = min([p.orders[0].processed_ms for p in miner_positions])
                filtered_positions[hotkey] = PositionFiltering.filter_positions_for_duration(miner_positions)

        return filtered_positions, hk_to_first_order_time

    def pre_run_setup(self):
        """
        Run this outside of init so that cross object dependencies can be set first. See validator.py
        """
        if self.perform_compaction:
            try:
                self.compact_price_sources()
            except Exception as e:
                bt.logging.error(f"Error performing compaction: {e}")
                traceback.print_exc()

        if self.perform_order_corrections:
            try:
                self.apply_order_corrections()
                #time_now_ms = TimeUtil.now_in_millis()
                #if time_now_ms < TARGET_MS:
                #    self.close_open_orders_for_suspended_trade_pairs()
            except Exception as e:
                bt.logging.error(f"Error applying order corrections: {e}")
                traceback.print_exc()

    def give_erronously_eliminated_miners_another_shot(self, hotkey_to_positions):
        time_now_ms = TimeUtil.now_in_millis()
        if time_now_ms > TARGET_MS:
            return
        # The MDD Checker will immediately eliminate miners if they exceed the maximum drawdown
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        eliminations_to_delete = set()
        for e in eliminations:
            if e['hotkey'] in ('5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X',
                               '5E9Ppyn5DzHGaPQmsHVnkNJDjGd7DstqjHWZpQhWPMbqzNex',
                               '5DoCFr2EoW1CGuYCEXhsuQdWRsgiUMuxGwNt4Xqb5TCptcBW',
                               '5EHpm2UK3CyhH1zZiJmM6erGrzkmVAF9EnT1QLSPhMzQaQHG',
                               '5GzYKUYSD5d7TJfK4jsawtmS2bZDgFuUYw8kdLdnEDxSykTU',
                               '5CALivVcJBTjYJFMsAkqhppQgq5U2PYW4HejCajHMvTMUgkC',
                               '5FTR8y26ap56vvahaxbB4PYxSkTQFpkQDqZN32uTVcW9cKjy',
                               '5Et6DsfKyfe2PBziKo48XNsTCWst92q8xWLdcFy6hig427qH',
                               '5HYRAnpjhcT45f6udFAbfJXwUmqqeaNvte4sTjuQvDxTaQB3',
                               '5Cd9bVVja2KdgsTiR7rTAh7a4UKVfnAuYAW1bs8BiedUE9JN',
                               '5FmvpMPvurA896m1X19fZXnct3NRXFrY57XVRcQLupb4sNZs',
                               '5DXRG8rCuuF7Lkd46mMbkdDNq52kDdph5PbxrCLAhuKAwkdq',
                               '5CcsBjaLAVfrjsAh6FyaTK4rBikkfQVanEmespwVpDGcE7jP',
                               '5DqxA5rsR5FGCkoZQ2eDnpQu1dBrdqr6EU7ZFKqsnHQQvpVh',
                               '5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ'):
                bt.logging.warning('Removed elimination for hotkey ', e['hotkey'])
                positions = hotkey_to_positions.get(e['hotkey'])
                if positions:
                    self.reopen_force_closed_positions(positions)
                eliminations_to_delete.add(e)

        self.elimination_manager.delete_eliminations(eliminations_to_delete)

    def strip_old_price_sources(self, position: Position, time_now_ms: int) -> int:
        n_removed = 0
        one_week_ago_ms = time_now_ms - 1000 * 60 * 60 * 24 * 7
        for o in position.orders:
            if o.processed_ms < one_week_ago_ms:
                if o.price_sources:
                    o.price_sources = []
                    n_removed += 1
        return n_removed

    def correct_for_tp(self, positions: List[Position], idx, prices, tp, timestamp_ms=None, n_attempts=0,
                       n_corrections=0, unique_corrections=None, pos=None):
        n_attempts += 1
        i = -1

        if pos:
            i = idx
        else:
            for p in positions:
                if p.trade_pair == tp:
                    pos = p
                    i += 1
                    if i == idx:
                        break

            if i != idx:
                bt.logging.warning(f"Could not find position for trade pair {tp.trade_pair_id} at index {idx}. i {i}")
                return n_attempts, n_corrections

        if pos and timestamp_ms:
            # check if the timestamp_ms is outside of 5 minutes of the position's open_ms
            delta_time_min = abs(timestamp_ms - pos.open_ms) / 1000.0 / 60.0
            if delta_time_min > 5.0:
                bt.logging.warning(
                    f"Timestamp ms: {timestamp_ms} is more than 5 minutes away from position open ms: {pos.open_ms}. delta_time_min {delta_time_min}")
                return n_attempts, n_corrections

        if not prices:
            # del position
            if pos:
                self.delete_position(pos)
                unique_corrections.add(pos.position_uuid)
                n_corrections += 1
                return n_attempts, n_corrections

        elif i == idx and pos and len(prices) <= len(pos.orders):
            self.delete_position(pos)
            for i in range(len(prices)):
                pos.orders[i].price = prices[i]

            old_return = pos.return_at_close  # noqa: F841
            pos.rebuild_position_with_updated_orders()
            self.save_miner_position(pos, delete_open_position_if_exists=False)
            unique_corrections.add(pos.position_uuid)
            n_corrections += 1
            return n_attempts, n_corrections
        else:
            bt.logging.warning(
                f"Could not correct position for trade pair {tp.trade_pair_id}. i {i}, idx {idx}, len(prices) {len(prices)}, len(pos.orders) {len(pos.orders)}")
        return n_attempts, n_corrections

    def reopen_force_closed_positions(self, positions):
        for position in positions:
            if position.is_closed_position and abs(position.net_leverage) > 0:
                print('rac1:', position.return_at_close)
                print(
                    f"Deleting position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id} nl {position.net_leverage}")
                self.delete_position(position)
                position.reopen_position()
                position.rebuild_position_with_updated_orders()
                print('rac2:', position.return_at_close)
                self.save_miner_position(position, delete_open_position_if_exists=False)
                print(f"Reopened position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id}")

    @timeme
    def compact_price_sources(self):
        time_now = TimeUtil.now_in_millis()
        n_price_sources_removed = 0
        hotkey_to_positions = self.get_positions_for_all_miners(sort_positions=True)
        eliminated_miners = self.elimination_manager.get_eliminations_from_memory()
        eliminated_hotkeys = set([e['hotkey'] for e in eliminated_miners])
        for hotkey, positions in hotkey_to_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            for position in positions:
                n = self.strip_old_price_sources(position, time_now)
                if n:
                    n_price_sources_removed += n
                    self.save_miner_position(position, delete_open_position_if_exists=False)

        bt.logging.info(f'Removed {n_price_sources_removed} price sources from old data.')

    def dedupe_positions(self, positions, miner_hotkey):
        positions_by_trade_pair = defaultdict(list)
        n_positions_deleted = 0
        n_orders_deleted = 0
        n_positions_rebuilt_with_new_orders = 0
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(deepcopy(position))

        for trade_pair, positions in positions_by_trade_pair.items():
            position_uuid_to_dedupe = {}
            for p in positions:
                if p.position_uuid in position_uuid_to_dedupe:
                    # Replace if it has more orders
                    if len(p.orders) > len(position_uuid_to_dedupe[p.position_uuid].orders):
                        old_position = position_uuid_to_dedupe[p.position_uuid]
                        self.delete_position(old_position)
                        position_uuid_to_dedupe[p.position_uuid] = p
                        n_positions_deleted += 1
                    else:
                        self.delete_position(p)
                        n_positions_deleted += 1
                else:
                    position_uuid_to_dedupe[p.position_uuid] = p

            for position in position_uuid_to_dedupe.values():
                order_uuid_to_dedup = {}
                new_orders = []
                any_orders_deleted = False
                for order in position.orders:
                    if order.order_uuid in order_uuid_to_dedup:
                        n_orders_deleted += 1
                        any_orders_deleted = True
                    else:
                        new_orders.append(order)
                        order_uuid_to_dedup[order.order_uuid] = order
                if any_orders_deleted:
                    position.orders = new_orders
                    position.rebuild_position_with_updated_orders()
                    self.save_miner_position(position, delete_open_position_if_exists=False)
                    n_positions_rebuilt_with_new_orders += 1
        if n_positions_deleted or n_orders_deleted or n_positions_rebuilt_with_new_orders:
            bt.logging.warning(
                f"Hotkey {miner_hotkey}: Deleted {n_positions_deleted} duplicate positions and {n_orders_deleted} "
                f"duplicate orders across {n_positions_rebuilt_with_new_orders} positions.")

    @timeme
    def apply_order_corrections(self):
        """
        This is our mechanism for manually synchronizing validator orders in situations where a bug prevented an
        order from filling. We are working on a more robust automated synchronization/recovery system.

        11/4/2024 - Metagraph synchronization was set to 5 minutes preventing a new miner from having their orders
        processed by all validators. After verifying that this miner's order should have been sent to all validators,
        we increased the metagraph update frequency to 1 minute to prevent this from happening again. This override
        will correct the order status for this miner.

        4/13/2024 - Price recalibration incorrectly applied to orders made after TwelveData websocket prices were
        implemented. This regressed pricing since the websocket prices are more accurate.

        Errantly closed out open CADCHF positions during a recalibration. Delete these positions that adversely affected
        miners

        One miner was eliminated due to a faulty candle from polygon at the close. We are investigating a workaround
        and have several candidate solutions.

        miner couldn't close position due to temporary bug. deleted position completely.

        # 4/15/24 Verified high lag on order price using Twelve Data

        # 4/17/24 Verified duplicate order sent due to a miner.py script. deleting entire position.

        # 4/19/24 Verified bug on old version of miner.py that delayed order significantly. The PR to reduce miner lag went
         live April 14th and this trade was April 9th

         4/23/24 - position price source flipped from polygon to TD. Need to be consistent within a position.
          Fix coming in next update.

          4/26/24, 5/9/24 - extreme price parsing is giving outliers from bad websocket data. Patch the function and manually correct
          elimination.

          Bug in forex market close due to federal holiday logic 5/27/24. deleted position

          5/30/24 - duplicate order bug. miner.py script updated.

          5.31.24 - validator outage due to twelvedata thread error. add position if not exists.

        """
        hotkey_to_positions = self.get_positions_for_all_miners(sort_positions=True)
        #self.give_erronously_eliminated_miners_another_shot(hotkey_to_positions)
        n_corrections = 0
        n_attempts = 0
        unique_corrections = set()
        now_ms = TimeUtil.now_in_millis()
        # Wipe miners only once when dynamic challenge period launches
        miners_to_wipe = []
        miners_to_promote = []
        wipe_positions = False
        if now_ms < TARGET_MS:
            # All miners that wanted their challenge period restarted
            miners_to_wipe = []# All miners that should have been promoted
            miners_to_promote = []

            for p in positions_to_snap:
                try:
                    pos = Position(**p)
                    if pos.is_open_position:
                        self.delete_open_position_if_exists(pos)
                    self.save_miner_position(pos)
                    print(f"Added position {pos.position_uuid} for trade pair {pos.trade_pair.trade_pair_id} for hk {pos.miner_hotkey}")
                except Exception as e:
                    print(f"Error adding position {p} {e}")

        #Don't accidentally promote eliminated miners
        for e in self.elimination_manager.get_eliminations_from_memory():
            if e['hotkey'] in miners_to_promote:
                miners_to_promote.remove(e['hotkey'])

        # Promote miners that would have passed challenge period
        for miner in miners_to_promote:
            if miner in self.challengeperiod_manager.challengeperiod_testing:
                self.challengeperiod_manager.challengeperiod_testing.pop(miner)
            if miner not in self.challengeperiod_manager.challengeperiod_success:
                self.challengeperiod_manager.challengeperiod_success[miner] = now_ms
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        # Wipe miners_to_wipe below
        for k in miners_to_wipe:
            if k not in hotkey_to_positions:
                hotkey_to_positions[k] = []

        n_eliminations_before = len(self.elimination_manager.get_eliminations_from_memory())
        for e in self.elimination_manager.get_eliminations_from_memory():
            if e['hotkey'] in miners_to_wipe:
                self.elimination_manager.delete_eliminations([e['hotkey']])
                print(f"Removed elimination for hotkey {e['hotkey']}")
        n_eliminations_after = len(self.elimination_manager.get_eliminations_from_memory())
        print(f'    n_eliminations_before {n_eliminations_before} n_eliminations_after {n_eliminations_after}')
        for miner_hotkey, positions in hotkey_to_positions.items():
            n_attempts += 1
            self.dedupe_positions(positions, miner_hotkey)
            if miner_hotkey in miners_to_wipe: # and now_ms < TARGET_MS:
                bt.logging.info(f"Resetting hotkey {miner_hotkey}")
                n_corrections += 1
                unique_corrections.update([p.position_uuid for p in positions])
                for pos in positions:
                    if wipe_positions:
                        self.delete_position(pos)
                    else:
                        if any(o.src == 1 for o in pos.orders):
                            pos.orders = [o for o in pos.orders if o.src != 1]
                            pos.rebuild_position_with_updated_orders()
                            self.save_miner_position(pos)
                            print(f'Removed eliminated orders from position {pos}')
                if miner_hotkey in self.challengeperiod_manager.challengeperiod_testing:
                    self.challengeperiod_manager.challengeperiod_testing.pop(miner_hotkey)
                    print(f'Removed challengeperiod testing for {miner_hotkey}')
                if miner_hotkey in self.challengeperiod_manager.challengeperiod_success:
                    self.challengeperiod_manager.challengeperiod_success.pop(miner_hotkey)
                    print(f'Removed challengeperiod success for {miner_hotkey}')

                self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

                perf_ledgers = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
                print('n perf ledgers before:', len(perf_ledgers))
                perf_ledgers_new = {k:v for k,v in perf_ledgers.items() if k != miner_hotkey}
                print('n perf ledgers after:', len(perf_ledgers_new))
                self.perf_ledger_manager.save_perf_ledgers(perf_ledgers_new)


            """
            if miner_hotkey == '5Cd9bVVja2KdgsTiR7rTAh7a4UKVfnAuYAW1bs8BiedUE9JN' and now_ms < TARGET_MS:
                position_that_should_exist_raw = {"miner_hotkey": "5Cd9bVVja2KdgsTiR7rTAh7a4UKVfnAuYAW1bs8BiedUE9JN",
                                                  "position_uuid": "f5a54d87-26c4-4a73-91b3-d8607b898507", "open_ms": 1734077788550,
                                                  "trade_pair": TradePair.USDJPY, "orders":
                                                      [{"order_type": "LONG", "leverage": 0.25, "price": 152.865, "processed_ms": 1734077788550, "order_uuid": "f5a54d87-26c4-4a73-91b3-d8607b898507", "price_sources": [], "src": 0},
                                                       {"order_type": "LONG", "leverage": 0.25, "price": 153.846, "processed_ms": 1734424931078, "order_uuid": "a53bd995-ad81-4b98-8039-5991abc00374", "price_sources": [], "src": 0},
                                                       {"order_type": "FLAT", "leverage": 0.25, "price": 153.656, "processed_ms": 1734517608513, "order_uuid": "3572eabe-4a4c-4fa2-8262-bf2a8e8ea394", "price_sources": [], "src": 0}],
                                                  "current_return": 1.0009828934026757, "close_ms": 1734517608513, "return_at_close": 1.000926976973672,
                                                  "net_leverage": 0.0, "average_entry_price": 153.3555, "position_type": "FLAT", "is_closed_position": True}

                success = self.enforce_position_state(position_that_should_exist_raw, TradePair.USDJPY, miner_hotkey,
                                                      unique_corrections, overwrite=True)
                n_corrections += success
                n_attempts += 1

            if miner_hotkey == "5HYBzAsTcxDXxHNXBpUJAQ9ZwmaGTwTb24ZBGJpELpG7LPGf" and now_ms < TARGET_MS:
                position_that_should_exist_raw = \
                {"miner_hotkey": "5HYBzAsTcxDXxHNXBpUJAQ9ZwmaGTwTb24ZBGJpELpG7LPGf",
                 "position_uuid": "c1be3244-5125-4bd6-83b7-9f56c84b3387", "open_ms": 1736389802186,
                 "trade_pair": TradePair.BTCUSD, "orders": [
                    {"order_type": "SHORT", "leverage": -0.5, "price": 94432.48, "processed_ms": 1736389802186,
                     "order_uuid": "c1be3244-5125-4bd6-83b7-9f56c84b3387", "price_sources": [
                        {"source": "Polygon_ws", "timespan_ms": 0, "open": 94432.48, "close": 94432.48,
                         "vwap": 94432.48, "high": 94432.48, "low": 94432.48, "start_ms": 1736389802000,
                         "websocket": False, "lag_ms": 186, "volume": 0.04655431},
                        {"source": "Tiingo_gdax_rest", "timespan_ms": 0, "open": 94431.06, "close": 94431.06,
                         "vwap": 94431.06, "high": 94431.06, "low": 94431.06, "start_ms": 1736389800615,
                         "websocket": True, "lag_ms": 1571, "volume": None},
                        {"source": "Polygon_rest", "timespan_ms": 1000, "open": 94237.5, "close": 94200.0,
                         "vwap": 94243.0749, "high": 94246.12, "low": 94200.0, "start_ms": 1736390000000,
                         "websocket": False, "lag_ms": 197814, "volume": 0.01125985}], "src": 0},
                    {"order_type": "FLAT", "leverage": 0.5, "price": 93908.85, "processed_ms": 1736395887370,
                     "order_uuid": "da0075dd-b97a-4cb4-a7d2-8c4e074101c5", "price_sources": [
                        {"source": "Polygon_ws", "timespan_ms": 0, "open": 93908.85, "close": 93908.85,
                         "vwap": 93908.85, "high": 93908.85, "low": 93908.85, "start_ms": 1736395887000,
                         "websocket": True, "lag_ms": 370, "volume": 1.3e-05},
                        {"source": "Tiingo_gdax_rest", "timespan_ms": 0, "open": 93908.85, "close": 93908.85,
                         "vwap": 93908.85, "high": 93908.85, "low": 93908.85, "start_ms": 1736395886709,
                         "websocket": True, "lag_ms": 661, "volume": None}], "src": 0}],
                 "current_return": 1.0027725100516263, "close_ms": 1736395887370, "return_at_close": 1.0022180496021578,
                 "net_leverage": 0.0, "average_entry_price": 94432.48, "position_type": "FLAT",
                 "is_closed_position": True}
                success = self.enforce_position_state(position_that_should_exist_raw, TradePair.BTCUSD, miner_hotkey, unique_corrections)
                n_corrections += success
                n_attempts += 1


            
                    
            if miner_hotkey == '5DX8tSyGrx1QuoR1wL99TWDusvmmWgQW5su3ik2Sc8y8Mqu3':
                n_corrections += self.correct_for_tp(positions, 0, [151.83500671, 151.792], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                n_corrections += self.correct_for_tp(positions, 0, [0.66623, 0.66634], TradePair.CADCHF, unique_corrections)

            if miner_hotkey == '5D4zieKMoRVm477oUyMTZAWZ9orzpiJM8K6ufQQjryiXwpGU':
                n_corrections += self.correct_for_tp(positions, 0, [0.66634, 0.6665], TradePair.CADCHF, unique_corrections)

            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                n_corrections += self.correct_for_tp(positions, 0, None, TradePair.CADCHF, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 0, [151.841, 151.773], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 1, [151.8, 152.302], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5Ec93qtHkKprEaA5EWXrmPmWppMeMiwaY868bpxfkH5ocBxi':
                n_corrections += self.correct_for_tp(positions, 0, [151.808, 151.844], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 1, [151.817, 151.84], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 2, [151.839, 151.809], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 3, [151.772, 151.751], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 4, [151.77, 151.748], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5Ct1J2jNxb9zeHpsj547BR1nZk4ZD51Bb599tzEWnxyEr4WR':
                n_corrections += self.correct_for_tp(positions, 0, None, TradePair.CADCHF, unique_corrections)
                
            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                correct_for_tp(positions, 2, None, TradePair.EURCHF, timestamp_ms=1712950839925)
            if miner_hotkey == '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
                correct_for_tp(positions, 0, [0.66242, 0.66464], TradePair.CADCHF)
            if miner_hotkey == '5D4zieKMoRVm477oUyMTZAWZ9orzpiJM8K6ufQQjryiXwpGU':
                correct_for_tp(positions, 0, [111.947, 111.987], TradePair.CADJPY)
            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                correct_for_tp(positions, 0, [151.727, 151.858, 153.0370, 153.0560, 153.0720, 153.2400, 153.2280, 153.2400], TradePair.USDJPY)
            if miner_hotkey == '5DfhKZckZwjCqEcBUsW7jwzA5APCdj5SgZbfK6zzS9bMPuHn':
                correct_for_tp(positions, 0, [111.599, 111.55999756, 111.622], TradePair.CADJPY)
                
            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                correct_for_tp(positions, 0, [151.73, 151.862, 153.047, 153.051, 153.071, 153.241, 153.225, 153.235], TradePair.USDJPY)
            if miner_hotkey == '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx':
                correct_for_tp(positions, 0, None, TradePair.ETHUSD, timestamp_ms=1713102534971)
            
            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                correct_for_tp(positions, 1, [100.192, 100.711, 100.379], TradePair.AUDJPY)
                correct_for_tp(positions, 1, None, TradePair.GBPJPY, timestamp_ms=1712624748605)
                correct_for_tp(positions, 2, None, TradePair.AUDCAD, timestamp_ms=1712839053529)
                
            if miner_hotkey == '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
                n_attempts, n_corrections = self.correct_for_tp(positions, 1, None, TradePair.BTCUSD, timestamp_ms=1712671378202, n_attempts=n_attempts, n_corrections=n_corrections, unique_corrections=unique_corrections)

            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                n_attempts, n_corrections = self.correct_for_tp(positions, 3, [1.36936, 1.36975], TradePair.USDCAD, n_attempts=n_attempts,
                                                                n_corrections=n_corrections,
                                                                unique_corrections=unique_corrections)
                                                                
            if miner_hotkey == '5Dxqzduahnqw8q3XSUfTcEZGU7xmAsfJubhHZwvXVLN9fSjR':
                self.reopen_force_closed_positions(positions)
                n_corrections += 1
                n_attempts += 1

            if miner_hotkey == '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
                #with open(ValiBkpUtils.get_positions_override_dir() + miner_hotkey + '.json', 'w') as f:
                #    dat = [p.to_json_string() for p in positions]
                #    f.write(json.dumps(dat, cls=CustomEncoder))


                time_now_ms = TimeUtil.now_in_millis()
                if time_now_ms > TARGET_MS:
                    return
                n_attempts += 1
                self.restore_from_position_override(miner_hotkey)
                n_corrections += 1

            if miner_hotkey == "5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw":
                time_now_ms = TimeUtil.now_in_millis()
                if time_now_ms > TARGET_MS:
                    return
                position_to_delete = [x for x in positions if x.trade_pair == TradePair.NZDUSD][-1]
                n_attempts, n_corrections = self.correct_for_tp(positions, None, None, TradePair.NZDUSD,
                                                                timestamp_ms=1716906327000, n_attempts=n_attempts,
                                                                n_corrections=n_corrections,
                                                                unique_corrections=unique_corrections,
                                                                pos=position_to_delete)
                                                                
            if miner_hotkey == "5DWmX9m33Tu66Qh12pr41Wk87LWcVkdyM9ZSNJFsks3QritF":
                 time_now_ms = TimeUtil.now_in_millis()
                 if time_now_ms > TARGET_MS:
                     return
                 position_to_delete = sorted([x for x in positions if x.trade_pair == TradePair.SPX], key=lambda x: x.close_ms)[-1]
                 n_attempts, n_corrections = self.correct_for_tp(positions, None, None, TradePair.SPX,
                                                                 timestamp_ms=None, n_attempts=n_attempts,
                                                                 n_corrections=n_corrections,
                                                                 unique_corrections=unique_corrections,
                                                                 pos=position_to_delete)
        """


        #5DCzvCF22vTVhXLtGrd7dBy19iFKKJNxmdSp5uo4C4v6Xx6h
        bt.logging.warning(
            f"Applied {n_corrections} order corrections out of {n_attempts} attempts. unique positions corrected: {len(unique_corrections)}")


    def enforce_position_state(self, position_that_should_exist_raw, trade_pair, miner_hotkey, unique_corrections, overwrite=False):
        position_that_should_exist_raw['trade_pair'] = trade_pair
        for o in position_that_should_exist_raw['orders']:
            o['trade_pair'] = trade_pair
        position = Position.from_dict(position_that_should_exist_raw)
        # check if the position exists on the filesystem
        existing_disk_positions = self.get_positions_for_one_hotkey(miner_hotkey)
        position_exists = False
        for p in existing_disk_positions:
            if p.position_uuid == position.position_uuid:
                position_exists = True
                break
        if not position_exists or overwrite:
            self.save_miner_position(position, delete_open_position_if_exists=True)
            print(f"Added position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id}")
            unique_corrections.add(position.position_uuid)
            return True
        return False

    def close_open_orders_for_suspended_trade_pairs(self):
        if not self.live_price_fetcher:
            self.live_price_fetcher = LivePriceFetcher(secrets=self.secrets, disable_ws=True)
        tps_to_eliminate = [TradePair.SPX, TradePair.DJI, TradePair.NDX, TradePair.VIX]
        if not tps_to_eliminate:
            return
        all_positions = self.get_positions_for_all_miners(sort_positions=True)
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
        bt.logging.info(f"Found {len(eliminations)} eliminations on disk.")
        for hotkey, positions in all_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            # Closing all open positions for the specified trade pair
            for position in positions:
                if position.is_closed_position:
                    continue
                if position.trade_pair in tps_to_eliminate:
                    price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(position.trade_pair, TARGET_MS)
                    live_price = price_sources[0].parse_appropriate_price(TARGET_MS, position.trade_pair.is_forex, OrderType.FLAT, position)
                    flat_order = Order(price=live_price,
                                       price_sources=price_sources,
                                       processed_ms=TARGET_MS,
                                       order_uuid=position.position_uuid[::-1],
                                       # determinstic across validators. Won't mess with p2p sync
                                       trade_pair=position.trade_pair,
                                       order_type=OrderType.FLAT,
                                       leverage=0,
                                       src=ORDER_SRC_DEPRECATION_FLAT)

                    position.add_order(flat_order)
                    self.save_miner_position(position, delete_open_position_if_exists=True)
                    bt.logging.info(
                    f"Position {position.position_uuid} for hotkey {hotkey} and trade pair {position.trade_pair.trade_pair_id} has been closed. Added flat order {flat_order}")


    def get_return_per_closed_position(self, positions: List[Position]) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []
        for position in positions:
            if position.is_open_position:
                continue
            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms
            closed_position_returns.append(position.return_at_close)

        cumulative_return = 1
        per_position_return = []

        # calculate the return over time at each position close
        for value in closed_position_returns:
            cumulative_return *= value
            per_position_return.append(cumulative_return)
        return per_position_return

    def get_percent_profitable_positions(self, positions: List[Position]) -> float:
        if len(positions) == 0:
            return 0.0

        profitable_positions = 0
        n_closed_positions = 0

        for position in positions:
            if position.is_open_position:
                continue

            n_closed_positions += 1
            if position.return_at_close > 1.0:
                profitable_positions += 1

        if n_closed_positions == 0:
            return 0.0

        return profitable_positions / n_closed_positions

    @staticmethod
    def positions_are_the_same(position1: Position, position2: Position | dict) -> (bool, str):
        # Iterate through all the attributes of position1 and compare them to position2.
        # Get attributes programmatically.
        comparing_to_dict = isinstance(position2, dict)
        for attr in dir(position1):
            attr_is_property = isinstance(getattr(type(position1), attr, None), property)
            if attr.startswith("_") or callable(getattr(position1, attr)) or (comparing_to_dict and attr_is_property) \
                    or (attr in ('model_computed_fields', 'model_config', 'model_fields', 'model_fields_set', 'newest_order_age_ms')):
                continue

            value1 = getattr(position1, attr)
            # Check if position2 is a dict and access the value accordingly.
            if comparing_to_dict:
                # Use .get() to avoid KeyError if the attribute is missing in the dictionary.
                value2 = position2.get(attr)
            else:
                value2 = getattr(position2, attr, None)

            if value1 != value2:
                return False, f"{attr} is different. {value1} != {value2}"
        return True, ""

    def get_miner_position_by_uuid(self, hotkey:str, position_uuid: str) -> Position | None:
        if hotkey not in self.hotkey_to_positions:
            return None
        return self._position_from_list_of_position(hotkey, position_uuid)

    def get_recently_updated_miner_hotkeys(self):
        """
        Identifies and returns a list of directories that have been updated in the last 3 days.
        """
        # Define the path to the directory containing the directories to check
        query_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        # Get the current time
        current_time = time.time()
        # List of directories updated in the last 24 hours
        updated_directory_names = []
        # Get the names of all directories in query_dir
        directory_names = CacheController.get_directory_names(query_dir)
        # Loop through each directory name
        for item in directory_names:
            item_path = Path(query_dir) / item  # Construct the full path
            # Get the last modification time of the directory
            root_last_modified_time_s = self._get_file_mod_time_s(item_path)
            latest_modification_time_s = self._get_latest_file_modification_time_s(item_path, root_last_modified_time_s)
            # Check if the directory was updated in the last 3 days
            if current_time - latest_modification_time_s < 259200:  # 3 days in seconds
                updated_directory_names.append(item)

        return updated_directory_names

    def _get_latest_file_modification_time_s(self, dir_path, root_last_modified_time):
        """
        Recursively finds the max modification time of all files within a directory.
        """
        latest_mod_time_s = root_last_modified_time
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                mod_time = self._get_file_mod_time_s(file_path)
                latest_mod_time_s = max(latest_mod_time_s, mod_time)

        return latest_mod_time_s

    def _get_file_mod_time_s(self, file_path):
        try:
            return os.path.getmtime(file_path)
        except OSError:  # Handle the case where the file is inaccessible
            return 0

    def delete_open_position_if_exists(self, position: Position) -> None:
        # See if we need to delete the open position file
        open_position = self.get_open_position_for_a_miner_trade_pair(position.miner_hotkey,
                                                                      position.trade_pair.trade_pair_id)
        if open_position:
            self.delete_position(open_position)

    def verify_open_position_write(self, miner_dir, updated_position):
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
        # Print all files found for dir
        positions = [self._get_position_from_disk(file) for file in all_files]
        if len(positions) == 0:
            return  # First time open position is being saved
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(
                f"More than one open position for miner {updated_position.miner_hotkey} and trade_pair."
                f" {updated_position.trade_pair.trade_pair_id}. Please restore cache. Positions: {positions}")
        elif len(positions) == 1:
            if positions[0].position_uuid != updated_position.position_uuid:
                msg = (
                    f"Attempted to write open position {updated_position.position_uuid} for miner {updated_position.miner_hotkey} "
                    f"and trade_pair {updated_position.trade_pair.trade_pair_id} but found an existing open"
                    f" position with a different position_uuid {positions[0].position_uuid}.")
                raise ValiRecordsMisalignmentException(msg)

        # -------------------------------------------------------------------------------------
        # Make sure the memory positions match the disk positions. Only run this during test
        if not self.running_unit_tests:
            return

        cdf = miner_dir[:-5] + 'closed/'
        positions.extend([self._get_position_from_disk(file) for file in ValiBkpUtils.get_all_files_in_dir(cdf)])

        temp = self.hotkey_to_positions.get(updated_position.miner_hotkey, [])
        positions_memory_by_position_uuid = {}
        for position in temp:
            if position.trade_pair == updated_position.trade_pair:
                positions_memory_by_position_uuid[position.position_uuid] = position
        positions_disk_by_uuid = {p.position_uuid: p for p in positions}
        errors = []
        for position_uuid, position in positions_memory_by_position_uuid.items():
            if position_uuid not in positions_disk_by_uuid:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found in memory but not on disk.")
                continue
            disk_position = positions_disk_by_uuid[position_uuid]
            is_same, diff = self.positions_are_the_same(position, disk_position)
            if not is_same:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found in memory but does not match the position on disk. {diff}")

        for position_uuid, position in positions_disk_by_uuid.items():
            if position_uuid not in positions_memory_by_position_uuid:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found on disk but not in memory.")
                continue
            memory_position = positions_memory_by_position_uuid[position_uuid]
            is_same, diff = self.positions_are_the_same(memory_position, position)
            if not is_same:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found on disk but does not match the position in memory. {diff}")
        if errors:
            raise ValiRecordsMisalignmentException(
                f"Found errors in miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id}. Errors: {errors}."
                f" Disk positions: {positions_disk_by_uuid.keys()}. Memory positions: {positions_memory_by_position_uuid.keys()}. all files {all_files}")
        # -------------------------------------------------------------------------------------

    def _position_from_list_of_position(self, hotkey, position_uuid):
        for p in self.hotkey_to_positions.get(hotkey, []):
            if p.position_uuid == position_uuid:
                return deepcopy(p)  # for unit tests we deepcopy. ipc cache never returns a reference.
        return None

    def _save_miner_position_to_memory(self, position: Position):
        # Multiprocessing-safe
        hk = position.miner_hotkey
        if hk not in self.hotkey_to_positions:
            existing_positions = []
        else:
            existing_positions = self.hotkey_to_positions[hk]

        # Santiy check
        if position.miner_hotkey in self.hotkey_to_positions and position.position_uuid in self.hotkey_to_positions[
            position.miner_hotkey]:
            existing_pos = self._position_from_list_of_position(position.miner_hotkey, position.position_uuid)
            assert existing_pos.trade_pair == position.trade_pair, f"Trade pair mismatch for position {position.position_uuid}. Existing: {existing_pos.trade_pair}, New: {position.trade_pair}"

        new_positions = [p for p in existing_positions if p.position_uuid != position.position_uuid]
        new_positions.append(deepcopy(position))
        self.hotkey_to_positions[hk] = new_positions  # Trigger the update on the multiprocessing Manager


    def save_miner_position(self, position: Position, delete_open_position_if_exists=True) -> None:
        if not self.is_backtesting:
            miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                         position.trade_pair.trade_pair_id,
                                                                         order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                                                                         running_unit_tests=self.running_unit_tests)
            if position.is_closed_position and delete_open_position_if_exists:
                self.delete_open_position_if_exists(position)
            elif position.is_open_position:
                self.verify_open_position_write(miner_dir, position)

            #print(f'Saving position {position.position_uuid} for miner {position.miner_hotkey} and trade pair {position.trade_pair.trade_pair_id} is_open {position.is_open_position}')
            ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)
        self._save_miner_position_to_memory(position)

    def overwrite_position_on_disk(self, position: Position) -> None:
        # delete the position from disk. Try the open position dir and the closed position dir
        self.delete_position(position, check_open_and_closed_dirs=True)
        miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                     position.trade_pair.trade_pair_id,
                                                                     order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                                                                     running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)
        self._save_miner_position_to_memory(position)

    def clear_all_miner_positions(self, target_hotkey=None):
        self.hotkey_to_positions = {}
        # Clear all files and directories in the directory specified by dir
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            if target_hotkey and file != target_hotkey:
                continue
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def get_number_of_eliminations(self):
        return len(self.elimination_manager.eliminations)

    def get_number_of_miners_with_any_positions(self):
        ans = 0
        for k, v in self.hotkey_to_positions.items():
            if len(v) > 0:
                ans += 1
        return ans

    def get_extreme_position_order_processed_on_disk_ms(self):
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        min_time = float("inf")
        max_time = 0
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                continue
            hotkey = file
            # Read all positions in this directory
            positions = self.get_positions_for_one_hotkey(hotkey)
            for p in positions:
                for o in p.orders:
                    min_time = min(min_time, o.processed_ms)
                    max_time = max(max_time, o.processed_ms)
        return min_time, max_time

    def get_open_position_for_a_miner_trade_pair(self, hotkey: str, trade_pair_id: str) -> Position | None:
        temp = self.hotkey_to_positions.get(hotkey, [])
        positions = []
        for p in temp:
            if p.trade_pair.trade_pair_id == trade_pair_id and p.is_open_position:
                positions.append(p)
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {hotkey} and trade_pair."
                                                   f" {trade_pair_id}. Please restore cache. Positions: {positions}")
        return deepcopy(positions[0]) if len(positions) == 1 else None

    def get_filepath_for_position(self, hotkey, trade_pair_id, position_uuid, is_open):
        order_status = OrderStatus.CLOSED if not is_open else OrderStatus.OPEN
        return ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=order_status,
                                                                running_unit_tests=self.running_unit_tests) + position_uuid

    def delete_position(self, p: Position, check_open_and_closed_dirs=False):
        hotkey = p.miner_hotkey
        trade_pair_id = p.trade_pair.trade_pair_id
        position_uuid = p.position_uuid
        is_open = p.is_open_position
        if check_open_and_closed_dirs:
            file_paths = [self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, True),
                          self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, False)]
        else:
            file_paths = [self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, is_open)]
        for fp in file_paths:
            if not self.is_backtesting:
                if os.path.exists(fp):
                    os.remove(fp)
                    bt.logging.info(f"Deleted position from disk: {fp}")
            self._delete_position_from_memory(hotkey, position_uuid)

    def _delete_position_from_memory(self, hotkey, position_uuid):
        if hotkey in self.hotkey_to_positions:
            new_positions = [p for p in self.hotkey_to_positions[hotkey] if p.position_uuid != position_uuid]
            if new_positions:
                self.hotkey_to_positions[hotkey] = new_positions
            else:
                del self.hotkey_to_positions[hotkey]

    def calculate_net_portfolio_leverage(self, hotkey: str) -> float:
        """
        Calculate leverage across all open positions
        Normalize each asset class with a multiplier
        """
        positions = self.get_positions_for_one_hotkey(hotkey, only_open_positions=True)

        portfolio_leverage = 0.0
        for position in positions:
            portfolio_leverage += abs(position.get_net_leverage()) * position.trade_pair.leverage_multiplier

        return portfolio_leverage

    @timeme
    def get_positions_for_all_miners(self, from_disk=False, **args):
        if from_disk:
            all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(
                ValiBkpUtils.get_miner_dir(self.running_unit_tests)
            )
        else:
            all_miner_hotkeys = list(self.hotkey_to_positions.keys())
        return self.get_positions_for_hotkeys(all_miner_hotkeys, from_disk=from_disk, **args)

    def _get_position_from_disk(self, file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        file_string = None
        try:
            file_string = ValiBkpUtils.get_file(file)
            ans = Position.model_validate_json(file_string)
            if not ans.orders:
                bt.logging.warning(f"Anomalous position has no orders: {ans.to_dict()}")
            return ans
        except FileNotFoundError:
            raise ValiFileMissingException(f"Vali position file is missing {file}")
        except UnpicklingError as e:
            raise ValiBkpCorruptDataException(f"file_string is {file_string}, {e}")
        except UnicodeDecodeError as e:
            raise ValiBkpCorruptDataException(
                f" Error {e} for file {file} You may be running an old version of the software. Confirm with the team if you should delete your cache. file string {file_string[:2000] if file_string else None}")
        except Exception as e:
            raise ValiBkpCorruptDataException(f"Error {e} file_path {file} file_string: {file_string}")

    def sort_by_close_ms(self, _position):
        return (
            _position.close_ms if _position.is_closed_position else float("inf")
        )

    def exorcise_positions(self, positions, all_files) -> List[Position]:
        """
        1/7/24: Not needed anymore?
        Disk positions can be left in a bad state for a variety of reasons. Let's clean them up here.
        If a dup is encountered, deleted both and let position syncing add the correct one back.
        """
        filtered_positions = []
        position_uuid_to_count = defaultdict(int)
        order_uuid_to_count = defaultdict(int)
        order_uuids_to_purge = set()
        for position in positions:
            position_uuid_to_count[position.position_uuid] += 1
            for order in position.orders:
                order_uuid_to_count[order.order_uuid] += 1
                if order_uuid_to_count[order.order_uuid] > 1:
                    order_uuids_to_purge.add(order.order_uuid)

        for file_name, position in zip(all_files, positions):
            if position_uuid_to_count[position.position_uuid] > 1:
                bt.logging.info(f"Exorcising position from disk due to duplicate position uuid: {file_name} {position}")
                os.remove(file_name)
                continue

            elif not position.orders:
                bt.logging.info(f"Exorcising position from disk due to no orders: {file_name} {position.to_dict()}")
                os.remove(file_name)
                continue

            new_orders = [x for x in position.orders if order_uuid_to_count[x.order_uuid] == 1]
            if len(new_orders) != len(position.orders):
                bt.logging.info(f"Exorcising position from disk due to order mismatch: {file_name} {position}")
                os.remove(file_name)
            else:
                filtered_positions.append(position)
        return filtered_positions

    def get_positions_for_one_hotkey(self,
                                     miner_hotkey: str,
                                     only_open_positions: bool = False,
                                     sort_positions: bool = False,
                                     acceptable_position_end_ms: int = None,
                                     from_disk: bool = False
                                     ) -> List[Position]:

        if from_disk:
            miner_dir = ValiBkpUtils.get_miner_all_positions_dir(miner_hotkey,
                                                                 running_unit_tests=self.running_unit_tests)
            all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
            positions = [self._get_position_from_disk(file) for file in all_files]
        else:
            positions = self.hotkey_to_positions.get(miner_hotkey, [])

        if acceptable_position_end_ms is not None:
            positions = [
                position
                for position in positions
                if position.open_ms > acceptable_position_end_ms
            ]

        if only_open_positions:
            positions = [
                position for position in positions if position.is_open_position
            ]

        if sort_positions:
            positions = sorted(positions, key=self.sort_by_close_ms)

        return positions

    def get_positions_for_hotkeys(self, hotkeys: List[str], eliminations: List = None, **args) -> Dict[
        str, List[Position]]:
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()

        return {
            hotkey: self.get_positions_for_one_hotkey(hotkey, **args)
            for hotkey in hotkeys
            if hotkey not in eliminated_hotkeys
        }

    def get_miner_hotkeys_with_at_least_one_position(self) -> set[str]:
        return set(self.hotkey_to_positions.keys())

if __name__ == '__main__':
    from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
    from vali_objects.utils.elimination_manager import EliminationManager
    from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
    bt.logging.enable_info()

    plm = PerfLedgerManager(None)
    pm = PositionManager(perf_ledger_manager=plm)
    elimination_manager = EliminationManager(None, pm, None)
    cpm = ChallengePeriodManager(None, position_manager=pm)
    pm.challengeperiod_manager = cpm
    pm.elimination_manager = elimination_manager
    pm.apply_order_corrections()
