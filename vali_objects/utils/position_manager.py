# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
import os
import shutil
import time
import copy
import traceback
import uuid
import math
from collections import defaultdict
from pickle import UnpicklingError
from typing import List, Dict, Union
import bittensor as bt
from pathlib import Path

from copy import deepcopy
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_config import TradePair, ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.position import Position
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.vali_dataclasses.order import OrderStatus, Order
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedger
TARGET_MS = 1717185371000 + 1000 * 60 * 60 * 2
class PositionManager(CacheController):
    def __init__(self, config=None, metagraph=None, running_unit_tests=False, perform_price_adjustment=False,
                 live_price_fetcher=None, perform_order_corrections=False, perform_fee_structure_update=False,
                 generate_correction_templates=False, apply_corrections_template=False, perform_compaction=False):
        super().__init__(config=config, metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.init_cache_files()
        self.position_locks = PositionLocks()
        self.live_price_fetcher = live_price_fetcher
        self.recalibrated_position_uuids = set()
        if perform_compaction:
            try:
                self.perform_compaction()
            except Exception as e:
                bt.logging.error(f"Error performing compaction: {e}")
                traceback.print_exc()

        if perform_price_adjustment:
            self.perform_price_recalibration()
        if perform_order_corrections:
            try:
                self.apply_order_corrections()
            except Exception as e:
                bt.logging.error(f"Error applying order corrections: {e}")
                traceback.print_exc()
        if generate_correction_templates:
            self.generate_correction_templates()
        if apply_corrections_template:
            self.apply_corrections_from_price_audit()

        if perform_fee_structure_update:
            self.ensure_latest_fee_structure_applied()

    def give_erronously_eliminated_miners_another_shot(self, hotkey_to_positions):
        time_now_ms = TimeUtil.now_in_millis()
        if time_now_ms > TARGET_MS:
            return
        # The MDD Checker will immediately eliminate miners if they exceed the maximum drawdown
        eliminations = self.get_miner_eliminations_from_disk()
        new_eliminations = []
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
            else:
                new_eliminations.append(e)

        self.write_eliminations_to_disk(new_eliminations)

    def strip_old_price_sources(self, position: Position, time_now_ms: int) -> int:
        n_removed = 0
        one_week_ago_ms = time_now_ms - 1000 * 60 * 60 * 24 * 7
        for o in position.orders:
            if o.processed_ms < one_week_ago_ms:
                if o.price_sources:
                    o.price_sources = []
                    n_removed += 1
        return n_removed

    def correct_for_tp(self, positions: List[Position], idx, prices, tp, timestamp_ms=None, n_attempts=0, n_corrections=0, unique_corrections=None, pos=None):
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
                self.delete_position_from_disk(pos)
                unique_corrections.add(pos.position_uuid)
                n_corrections += 1
                return n_attempts, n_corrections

        elif i == idx and pos and len(prices) <= len(pos.orders):
            self.delete_position_from_disk(pos)
            for i in range(len(prices)):
                pos.orders[i].price = prices[i]

            old_return = pos.return_at_close
            pos.rebuild_position_with_updated_orders()
            self.save_miner_position_to_disk(pos, delete_open_position_if_exists=False)
            unique_corrections.add(pos.position_uuid)
            n_corrections += 1
            return n_attempts, n_corrections
        else:
            bt.logging.warning(f"Could not correct position for trade pair {tp.trade_pair_id}. i {i}, idx {idx}, len(prices) {len(prices)}, len(pos.orders) {len(pos.orders)}")
        return n_attempts, n_corrections

    def reopen_force_closed_positions(self, positions):
        for position in positions:
            if position.is_closed_position and abs(position.net_leverage) > 0:
                print('rac1:', position.return_at_close)
                print(f"Deleting position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id} nl {position.net_leverage}")
                self.delete_position_from_disk(position)
                position.reopen_position()
                position.rebuild_position_with_updated_orders()
                print('rac2:', position.return_at_close)
                self.save_miner_position_to_disk(position, delete_open_position_if_exists=False)
                print(f"Reopened position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id}")

    def perform_compaction(self):
        time_now = TimeUtil.now_in_millis()
        n_price_sources_removed = 0
        hotkey_to_positions = self.get_all_disk_positions_for_all_miners(only_open_positions=False, sort_positions=True)
        eliminated_miners = self.get_miner_eliminations_from_disk()
        eliminated_hotkeys = set([e['hotkey'] for e in eliminated_miners])
        for hotkey, positions in hotkey_to_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            for position in positions:
                n = self.strip_old_price_sources(position, time_now)
                if n:
                    n_price_sources_removed += n
                    self.save_miner_position_to_disk(position, delete_open_position_if_exists=False)

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
                        self.delete_position_from_disk(old_position)
                        position_uuid_to_dedupe[p.position_uuid] = p
                        n_positions_deleted += 1
                    else:
                        self.delete_position_from_disk(p)
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
                    self.save_miner_position_to_disk(position, delete_open_position_if_exists=False)
                    n_positions_rebuilt_with_new_orders += 1
        if n_positions_deleted or n_orders_deleted or n_positions_rebuilt_with_new_orders:
            bt.logging.warning(f"Hotkey {miner_hotkey}: Deleted {n_positions_deleted} duplicate positions and {n_orders_deleted} "
                           f"duplicate orders across {n_positions_rebuilt_with_new_orders} positions.")

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

          5.31.24 - validator outage due to twlevedata thread error. add position if not exists.

        """

        hotkey_to_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False, perform_exorcism=True)
        #self.give_erronously_eliminated_miners_another_shot(hotkey_to_positions)
        n_corrections = 0
        n_attempts = 0
        unique_corrections = set()
        for miner_hotkey, positions in hotkey_to_positions.items():
            self.dedupe_positions(positions, miner_hotkey)
            """
                    
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
        """



        bt.logging.warning(f"Applied {n_corrections} order corrections out of {n_attempts} attempts. unique positions corrected: {len(unique_corrections)}")


    def restore_from_position_override(self, miner_hotkey):
        self.clear_all_miner_positions_from_disk(target_hotkey=miner_hotkey)
        with open(ValiBkpUtils.get_positions_override_dir() + miner_hotkey + '.json', 'r') as f:
            positions = json.loads(f.read(), cls=GeneralizedJSONDecoder)
            for pos in positions:
                position = Position(**pos)
                ValiBkpUtils.make_dir(ValiBkpUtils.get_miner_all_positions_dir(miner_hotkey))
                self.save_miner_position_to_disk(position, delete_open_position_if_exists=False)

    def ensure_latest_fee_structure_applied(self):
        hotkey_to_positions = self.get_all_disk_positions_for_all_miners(only_open_positions=False, sort_positions=True)
        eliminated_miners = self.get_miner_eliminations_from_disk()
        eliminated_hotkeys = set([e['hotkey'] for e in eliminated_miners])
        n_positions_seen = 0
        n_positions_updated = 0
        n_positions_updated_significantly = 0
        n_positions_flipped_to_positive = 0
        n_positions_flipped_negative = 0
        n_positions_stayed_the_same = 0
        significant_deltas = []
        for hotkey, positions in hotkey_to_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            for position in positions:
                # skip open positions as their returns will be updated in the next MDD check.
                # Also once positions closes, it will make it past this check next validator boot
                if position.is_open_position:
                    continue
                if not position.trade_pair.is_crypto:
                    continue
                n_positions_seen += 1
                # Ensure this position is using the latest fee structure. If not, recalculate and persist the new return to disk
                old_return_at_close = position.return_at_close
                new_return_at_close = position.calculate_return_with_fees(position.current_return, timestamp_ms=position.close_ms)
                if old_return_at_close != new_return_at_close:
                    n_positions_updated += 1
                    if old_return_at_close < 1.0 and new_return_at_close > 1.0:
                        n_positions_flipped_to_positive += 1
                    elif old_return_at_close > 1.0 and new_return_at_close < 1.0:
                        n_positions_flipped_negative += 1
                    if abs(old_return_at_close - new_return_at_close) / old_return_at_close > 0.03:
                        n_positions_updated_significantly += 1
                        print(f"Updating return_at_close for position {position.position_uuid} trade pair "
                                        f"{position.trade_pair.trade_pair_id} from {old_return_at_close} to {new_return_at_close}")
                        #significant_deltas.append((old_return_at_close, new_return_at_close, position.to_json_string()))
                    position.return_at_close = new_return_at_close
                    self.save_miner_position_to_disk(position, delete_open_position_if_exists=False)
                else:
                    n_positions_stayed_the_same += 1
        bt.logging.info(f"Updated {n_positions_updated} positions. {n_positions_updated_significantly} positions "
                        f"were updated significantly. {n_positions_stayed_the_same} stayed the same. {n_positions_seen} "
                        f"positions were seen in total. Significant deltas: {significant_deltas} "
                        f"n_positions_flipped_to_positive: {n_positions_flipped_to_positive}"
                        f" n_positions_flipped_negative: {n_positions_flipped_negative}")

    def handle_eliminated_miner(self, hotkey: str,
                                trade_pair_to_price_source_used_for_elimination_check: Dict[TradePair, PriceSource],
                                open_position_trade_pairs=None):

        tps_to_iterate_over = open_position_trade_pairs if open_position_trade_pairs else TradePair
        for trade_pair in tps_to_iterate_over:
            with self.position_locks.get_lock(hotkey, trade_pair.trade_pair_id):
                open_position = self.get_open_position_for_a_miner_trade_pair(hotkey, trade_pair.trade_pair_id)
                source_for_elimination = trade_pair_to_price_source_used_for_elimination_check.get(trade_pair)
                if open_position:
                    bt.logging.info(
                        f"Closing open position for hotkey: {hotkey} and trade_pair: {trade_pair.trade_pair_id}. "
                        f"Source for elimination {source_for_elimination}")
                    open_position.close_out_position(TimeUtil.now_in_millis())
                    if source_for_elimination:
                        open_position.orders[-1].price_sources.append(source_for_elimination)
                    self.save_miner_position_to_disk(open_position)

    def recalculate_return_at_close_and_write_corrected_position_to_disk(self, position: Position, hotkey:str):
        # TODO LOCK and how to handle open positions?
        tp = position.trade_pair
        if tp not in (TradePair.CADJPY, TradePair.USDJPY, TradePair.CHFJPY):
            return position.return_at_close

        any_changes = False
        disposable_clone = deepcopy(position)
        deltas = []
        new_orders = []
        for o in disposable_clone.orders:
            timestamp_ms = o.processed_ms
            old_price = o.price
            new_price = self.live_price_fetcher.get_close_at_date(tp, timestamp_ms)

            if new_price is not None and new_price != old_price:
                # IF the order recent, don't recalibrate
                if o.processed_ms > 1712901063000:
                    time_lo = TimeUtil.timestamp_ms_to_eastern_time_str(o.processed_ms)
                    bt.logging.warning(
                        f"Skipping recalibration for trade pair {tp.trade_pair_id}."
                        f" Last order is recent. {time_lo}. Price would have changed from {old_price} to {new_price}.")
                else:
                    any_changes = True
                    deltas.append((old_price, new_price))
                    o.price = new_price
            new_orders.append(o)


        for i in range(len(position.orders)):
            orders = new_orders[:i+1]
            disposable_clone.orders = orders
            disposable_clone.position_type = None
            disposable_clone._update_position()
            if disposable_clone.orders[0].leverage < 0:
                order_type = OrderType.SHORT
            else:
                order_type = OrderType.LONG
            if 0:#len(orders) > 1:
                prev_order = orders[-2]
                o = orders[-1]
                window_start_ms = prev_order.processed_ms
                window_end_ms = o.processed_ms
                candles = self.live_price_fetcher.get_candles([tp], window_start_ms,
                                                              window_end_ms) if window_start_ms else None
                fs = TimeUtil.millis_to_formatted_date_str(window_start_ms)
                fe = TimeUtil.millis_to_formatted_date_str(window_end_ms)
                if tp in candles and candles[tp] and isinstance(candles[tp], list):
                    min_price_seen = min([x.low for x in candles[tp]])
                    max_price_seen = max([x.high for x in candles[tp]])
                    unrealized_return = disposable_clone.calculate_unrealized_pnl(min_price_seen if order_type == OrderType.LONG else max_price_seen)
                    unrealized_return_with_fees = disposable_clone.calculate_return_with_fees(unrealized_return, timestamp_ms=window_end_ms)
                    drawdown = self.calculate_drawdown(unrealized_return_with_fees, 1.0)
                    bt.logging.warning(f"Drawdown is {drawdown} for hotkey {hotkey} trade pair {tp.trade_pair_id}. Between time: {fs} and {fe}. p1: {prev_order.price} p2: {o.price}, min: {min_price_seen}.")
                    if drawdown < ValiConfig.MAX_TOTAL_DRAWDOWN:
                        bt.logging.error(f"Drawdown is {drawdown} for hotkey {hotkey} trade pair {tp.trade_pair_id}. Unrealized return: {unrealized_return_with_fees}.")


        assert len(disposable_clone.orders) == len(position.orders)
        if any_changes:
            with self.position_locks.get_lock(hotkey, tp.trade_pair_id):
                self.save_miner_position_to_disk(disposable_clone, delete_open_position_if_exists=False)
            bt.logging.info(f"Recalculated return_at_close for position {position.position_uuid}."
                            f" Trade pair {position.trade_pair.trade_pair_id} New value: "
                            f"{disposable_clone.return_at_close}. Original value: {position.return_at_close}")
            bt.logging.info(f"Corrected n={len(deltas)} prices for position {position.position_uuid} Trade pair {tp.trade_pair_id}. Deltas (before/after): {deltas}")

        return disposable_clone.return_at_close

    def close_open_orders_for_suspended_trade_pairs(self):
        tps_to_eliminate = []
        if not tps_to_eliminate:
            return
        all_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)
        eliminations = self.get_miner_eliminations_from_disk()
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
                    bt.logging.info(
                        f"Position {position.position_uuid} for hotkey {hotkey} and trade pair {position.trade_pair.trade_pair_id} has been closed")
            self.handle_eliminated_miner(hotkey, {}, tps_to_eliminate)

    def perform_price_recalibration(self, time_per_batch_s=90):
        try:
            t0 = time.time()
            if not self.live_price_fetcher.polygon_available:
                bt.logging.error("Polygon API not detected. Skipping price recalibration. Your validator will fall out of consensus.")
                return
            self.perform_price_recalibration_arap(time_per_batch_s)
            bt.logging.info(f"Price recalibration complete for {len(self.recalibrated_position_uuids)} positions in {time.time() - t0} seconds.")
        except Exception as e:
            bt.logging.error(f"Error performing price recalibration: {e}")
            bt.logging.error(traceback.format_exc())

    def perform_price_recalibration_arap(self, time_per_batch_s):
        t0 = time.time()
        all_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)
        eliminations = self.get_miner_eliminations_from_disk()
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
        hotkeys_eliminated_to_current_return = {}
        hotkeys_with_return_modified = set()
        skipped_eliminated = 0
        n_positions_total = 0
        n_positions_modified = 0
        RETURNS_MODIFIED = []
        for hotkey, positions in all_positions.items():
            max_portfolio_return = 1.0
            cur_portfolio_return = 1.0
            most_recent_elimination_idx = None
            n_positions_total += len(positions)

            if len(positions) == 0:
                continue

            if hotkey in eliminated_hotkeys:
                skipped_eliminated += len(positions)
                continue

            dd_to_log = None
            for i, position in enumerate(positions):
                if position.position_uuid in self.recalibrated_position_uuids:
                    continue
                if time.time() - t0 > time_per_batch_s:
                    return
                original_return = position.return_at_close
                new_return = self.recalculate_return_at_close_and_write_corrected_position_to_disk(position, hotkey)
                if original_return != new_return:
                    hotkeys_with_return_modified.add(hotkey)
                    n_positions_modified += 1
                    RETURNS_MODIFIED.append((original_return, new_return, position.trade_pair.trade_pair_id, hotkey))
                if position.is_open_position:
                    continue
                self.recalibrated_position_uuids.add(position.position_uuid)
                cur_portfolio_return *= new_return
                max_portfolio_return = max(max_portfolio_return, cur_portfolio_return)
                drawdown = self.calculate_drawdown(cur_portfolio_return, max_portfolio_return)
                mdd_failure = self.is_drawdown_beyond_mdd(drawdown,
                                                           time_now=TimeUtil.millis_to_datetime(position.close_ms))
                if mdd_failure:
                    dd_to_log = drawdown
                    most_recent_elimination_idx = i

            return_with_open_positions = cur_portfolio_return
            for p in positions:
                if p.is_open_position:
                    return_with_open_positions *= p.return_at_close

            drawdown = self.calculate_drawdown(return_with_open_positions, max_portfolio_return)
            mdd_failure = self.is_drawdown_beyond_mdd(drawdown)
            if mdd_failure:
                most_recent_elimination_idx = len(positions) - 1
                dd_to_log = drawdown

            if most_recent_elimination_idx is not None:
                return_as_shown_on_dash = 1.0
                for p in positions:
                    return_as_shown_on_dash *= p.return_at_close
                hotkeys_eliminated_to_current_return[hotkey] = return_as_shown_on_dash
                msg =(
                    f"MDD failure occurred at position {most_recent_elimination_idx} out of {len(positions)} positions for hotkey "
                    f"{hotkey}. Drawdown: {dd_to_log}. MDD failure: {mdd_failure}. Portfolio return: {return_with_open_positions}. ")
                bt.logging.warning(msg)



        bt.logging.info(f"Found n= {len(hotkeys_eliminated_to_current_return)} hotkeys to eliminate. After modifying returns for n = {len(hotkeys_with_return_modified)} hotkeys.")
        bt.logging.info(f"Total initial positions: {n_positions_total}, Modified {n_positions_modified}, Skipped {skipped_eliminated} eliminated.")
        bt.logging.warning(f"Position returns modified: {RETURNS_MODIFIED}")
        for rm in RETURNS_MODIFIED:
            if abs(rm[0] - rm[1]) / rm[0] > -1:
                bt.logging.warning(f"    Original return: {rm[0]}, New return: {rm[1]}, Trade pair: {rm[2]} hotkey: {rm[3]}")

        for k, v in sorted(hotkeys_eliminated_to_current_return.items(), key=lambda x: x[1]):
            bt.logging.info(f"hotkey: {k}. return as shown on dash: {v}")
        return hotkeys_eliminated_to_current_return

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
    def augment_perf_ledger(
        ledger: dict[str, PerfLedger],
        evaluation_time_ms: int,
        decay_coefficient: float = None,
        time_decay_coefficient: float = None,
        baseline_gain_rate: float = None
    ) -> dict[str, PerfLedger]:
        """
        Augments the entire perf ledger, augmented with historical decay.
        """
        if not ledger:
            return ledger
        
        if decay_coefficient is None:
            decay_coefficient = ValiConfig.HISTORICAL_GAIN_LOSS_COEFFICIENT

        if time_decay_coefficient is None:
            time_decay_coefficient = ValiConfig.HISTORICAL_DECAY_TIME_INTENSITY_COEFFICIENT

        if baseline_gain_rate is None:
            baseline_gain_rate = ValiConfig.BASELINE_ANNUAL_LOG_RETURN_MS
        
        augmented_ledger = copy.deepcopy(ledger)
        for miner, minerledger in augmented_ledger.items():
            augmented_ledger[miner].cps = PositionManager.augment_perf_checkpoint(
                minerledger.cps,
                evaluation_time_ms,
                gain_augmentation_coefficient=decay_coefficient,
                loss_augmentation_coefficient=decay_coefficient,
                time_decay_coefficient=time_decay_coefficient,
                baseline_gain_rate=baseline_gain_rate
            )

        return augmented_ledger
    
    @staticmethod
    def cumulative_returns(ledger: dict[str, PerfLedger]) -> dict[int, float]:
        """
        Returns the cumulative return of the ledger.
        """
        cumulative_returns = {}
        ledger_copy = copy.deepcopy(ledger)
        for miner, minerledger in ledger_copy.items():
            minerspecific_returns = []
            returnoverall = 1.0
            if len(minerledger.cps) == 0:
                continue

            for cp in minerledger.cps:
                returnvalue = math.exp(cp.gain + cp.loss)
                returnoverall *= returnvalue
                minerspecific_returns.append({
                    "last_update_ms": cp.last_update_ms,
                    "overall_returns": returnoverall
                })

            cumulative_returns[miner] = minerspecific_returns

        return cumulative_returns
    
    @staticmethod
    def augment_perf_checkpoint(
        cps: list[PerfCheckpoint],
        evaluation_time_ms: int,
        gain_augmentation_coefficient: float = None,
        loss_augmentation_coefficient: float = None,
        time_decay_coefficient: float = None,
        baseline_gain_rate: float = None
    ) -> list[PerfCheckpoint]:
        """
        Returns the return at each performance checkpoint, augmented with historical decay.
        """
        if len(cps) == 0:
            return []
        
        if gain_augmentation_coefficient is None:
            gain_augmentation_coefficient = ValiConfig.HISTORICAL_DECAY_GAIN_COEFFICIENT

        if loss_augmentation_coefficient is None:
            loss_augmentation_coefficient = ValiConfig.HISTORICAL_DECAY_LOSS_COEFFICIENT

        if baseline_gain_rate is None:
            baseline_gain_rate = ValiConfig.BASELINE_ANNUAL_LOG_RETURN_MS
        
        cps_augmented = []
        for cp in cps:
            cp_copy = copy.deepcopy(cp)
            baseline_gain = baseline_gain_rate * cp.accum_ms

            lookback_fraction = PositionUtils.compute_lookback_fraction(
                cp.last_update_ms,
                cp.last_update_ms,
                evaluation_time_ms
            )
            
            cp_copy.gain = PositionUtils.dampen_value(
                cp.gain,
                lookback_fraction,
                time_decay_coefficient
            )

            cp_copy.loss = PositionUtils.dampen_value(
                cp.loss - baseline_gain, # tbill augmentation
                lookback_fraction,
                time_decay_coefficient
            )

            cps_augmented.append(cp_copy)
        return cps_augmented
    
    def get_return_per_closed_position_augmented(
            self, 
            positions: List[Position], 
            evaluation_time_ms: Union[None, int] = None
        ) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []

        for position in positions:
            # if bool on for closed only
            if position.is_open_position:
                continue

            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms

            # this value will probably be around 0, or between -1 and 1
            logged_return_at_close = PositionUtils.log_transform(position.return_at_close)

            # this should be even closer to 0 with an absolute value less than the logged_return_at_close
            if evaluation_time_ms is None:
                raise ValueError("evaluation_time_ms must be provided if augmented is True")

            dampened_return_at_close = PositionUtils.dampen_return(
                logged_return_at_close,
                position.open_ms, 
                position.close_ms, 
                evaluation_time_ms
            )
            closed_position_returns.append(dampened_return_at_close)

        consistency_penalty = PositionUtils.compute_consistency_penalty(
            positions, 
            evaluation_time_ms
        )

        # cumulative_return_logged = 0
        # per_position_return_logged = []

        # # calculate the return over time at each position close
        # for value in closed_position_returns:
        #     cumulative_return_logged += value
        #     per_position_return_logged.append(value)

        return [ x * consistency_penalty for x in closed_position_returns ]

    @staticmethod
    def positions_are_the_same(position1: Position, position2: Position | dict) -> (bool, str):
        # Iterate through all the attributes of position1 and compare them to position2.
        # Get attributes programmatically.
        comparing_to_dict = isinstance(position2, dict)
        for attr in dir(position1):
            attr_is_property = isinstance(getattr(type(position1), attr, None), property)
            if attr.startswith("_") or callable(getattr(position1, attr)) or (comparing_to_dict and attr_is_property) \
                    or (attr in ('model_computed_fields', 'model_config', 'model_fields')):
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

    def get_miner_position_from_disk_using_position_in_memory(self, memory_position: Position) -> Position:
        # Position could have changed to closed
        fp = self.get_filepath_for_position(hotkey=memory_position.miner_hotkey,
                                            trade_pair_id=memory_position.trade_pair.trade_pair_id,
                                            position_uuid=memory_position.position_uuid,
                                            is_open=memory_position.is_open_position)
        try:
            return self.get_miner_position_from_disk(fp)
        except Exception as e:
            bt.logging.info(f"Error getting position from disk: {e}")
            if memory_position.is_open_position:
                bt.logging.warning(f"Attempting to get closed position from disk for memory position {memory_position}")
                fp = self.get_filepath_for_position(hotkey=memory_position.miner_hotkey,
                                                    trade_pair_id=memory_position.trade_pair.trade_pair_id,
                                                    position_uuid=memory_position.position_uuid,
                                                    is_open=False
                                                )
                return self.get_miner_position_from_disk(fp)
            else:
                raise e

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

    def get_latest_file_mod_time_s(self):
        """
        get the last file modification time from all directories
        """
        # Define the path to the directory containing the directories to check
        query_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        # Get the names of all directories in query_dir
        directory_names = CacheController.get_directory_names(query_dir)

        latest_modification_time_s = 0
        # Loop through each directory name
        for item in directory_names:
            item_path = Path(query_dir) / item  # Construct the full path
            latest_modification_time_s = max(latest_modification_time_s, self._get_latest_file_modification_time_s(item_path, 0))

        return latest_modification_time_s

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
            self.delete_position_from_disk(open_position)

    def verify_open_position_write(self, miner_dir, updated_position):
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
        # Print all files found for dir
        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions) == 0:
            return  # First time open position is being saved
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {updated_position.miner_hotkey} and trade_pair."
                                                   f" {updated_position.trade_pair.trade_pair_id}. Please restore cache. Positions: {positions}")
        elif len(positions) == 1:
            if positions[0].position_uuid != updated_position.position_uuid:
                msg = (f"Attempted to write open position {updated_position.position_uuid} for miner {updated_position.miner_hotkey} "
                       f"and trade_pair {updated_position.trade_pair.trade_pair_id} but found an existing open"
                       f" position with a different position_uuid {positions[0].position_uuid}.")
                raise ValiRecordsMisalignmentException(msg)

    def save_miner_position_to_disk(self, position: Position, delete_open_position_if_exists=True) -> None:
        miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                     position.trade_pair.trade_pair_id,
                                                                     order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                                                                     running_unit_tests=self.running_unit_tests)
        if position.is_closed_position and delete_open_position_if_exists:
            self.delete_open_position_if_exists(position)
        elif position.is_open_position:
            self.verify_open_position_write(miner_dir, position)

        ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)

    def clear_all_miner_positions_from_disk(self, target_hotkey=None):
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
        return len(self.get_miner_eliminations_from_disk())

    def get_number_of_plagiarism_scores(self):
        return len(self.get_plagiarism_scores_from_disk())

    def get_number_of_miners_with_any_positions(self):
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        ret = 0
        try:
            for file in os.listdir(dir):
                file_path = os.path.join(dir, file)
                ret += os.path.isdir(file_path)
            bt.logging.info(f"Number of miners with any positions: {ret}. Positions dir: {dir}")
        except FileNotFoundError:
            bt.logging.info(f"Directory for miners doesn't exist [{dir}].")
        return ret

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
            positions = self.get_all_miner_positions(hotkey)
            for p in positions:
                for o in p.orders:
                    min_time = min(min_time, o.processed_ms)
                    max_time = max(max_time, o.processed_ms)
        return min_time, max_time



    def get_open_position_for_a_miner_trade_pair(self, hotkey: str, trade_pair_id: str) -> Position | None:
        dir = ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=OrderStatus.OPEN,
                                                               running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(dir)
        # Print all files found for dir
        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {hotkey} and trade_pair."
                                                   f" {trade_pair_id}. Please restore cache. Positions: {positions}")
        return positions[0] if len(positions) == 1 else None

    def get_filepath_for_position(self, hotkey, trade_pair_id, position_uuid, is_open):
        order_status = OrderStatus.CLOSED if not is_open else OrderStatus.OPEN
        return ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=order_status,
                                                               running_unit_tests=self.running_unit_tests) + position_uuid

    def delete_position_from_disk(self, p:Position):
        hotkey = p.miner_hotkey
        trade_pair_id = p.trade_pair.trade_pair_id
        position_uuid = p.position_uuid
        is_open = p.is_open_position
        filepath = self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, is_open)
        os.remove(filepath)
        bt.logging.info(f"Deleted position from disk: {filepath}")


    def check_elimination(self, positions, miner_hotkey, orig_portfolio_return, ALL_MSGS, ELIM_MSGS):
        max_portfolio_return = 1.0
        cur_portfolio_return = 1.0
        dd_to_log = None
        most_recent_elimination_idx = None
        n_positions_flipped_to_loss = 0
        n_positions_flipped_to_gain = 0
        for i, position in enumerate(positions):
            orig_return = position.return_at_close
            position.rebuild_position_with_updated_orders()

            new_return = position.return_at_close
            if new_return < 1.0 and orig_return >= 1.0:
                n_positions_flipped_to_loss += 1
            elif new_return >= 1.0 and orig_return < 1.0:
                n_positions_flipped_to_gain += 1

            if position.is_open_position:
                continue

            cur_portfolio_return *= new_return
            max_portfolio_return = max(max_portfolio_return, cur_portfolio_return)
            drawdown = self.calculate_drawdown(cur_portfolio_return, max_portfolio_return)
            mdd_failure = self.is_drawdown_beyond_mdd(drawdown,
                                                      time_now=TimeUtil.millis_to_datetime(position.close_ms))
            if mdd_failure:
                dd_to_log = drawdown
                most_recent_elimination_idx = i

        if most_recent_elimination_idx is not None:
            msg = (
                f"MDD failure occurred at position {most_recent_elimination_idx} out of {len(positions)} positions for hotkey "
                f"{miner_hotkey}. Drawdown: {dd_to_log}. MDD failure: {mdd_failure}. Portfolio return: {cur_portfolio_return}. ")
            ELIM_MSGS.append(msg)
            bt.logging.warning(msg)
        msg = (f"hotkey: {miner_hotkey}. unrealized return as shown on dash: {orig_portfolio_return} new realized return (excludes open positions): {cur_portfolio_return}"
              f" n_positions_flipped_to_loss: {n_positions_flipped_to_loss} n_positions_flipped_to_gain: {n_positions_flipped_to_gain}")
        ALL_MSGS.append(msg)
        print(msg)


    def apply_corrections_from_price_audit(self):
        f_path = ValiConfig.BASE_DIR + '/price_audits/price_audit_4-18-24.txt'

        trade_pair_str_to_n_times_seen_per_hk = {}
        position_to_corrections_per_hk = {}
        hotkey_to_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)

        order_timestamp_to_position = {}
        for hotkey, positions in hotkey_to_positions.items():
            for i, position in enumerate(positions):
                for o in position.orders:
                    assert o.processed_ms not in order_timestamp_to_position
                    order_timestamp_to_position[o.processed_ms] = position

        with open(f_path, 'r') as f:
            miner_hotkey = None
            position = None
            for line in f:
                line = line.strip()
                if line.startswith('trade_pair'):  # Ignore header
                    continue

                parts = line.split('	')
                if len(line) == 48:  # Assuming the hotkey is always 48 characters long
                    if position is not None and miner_hotkey is not None:
                        trade_pair_str_to_n_times_seen_per_hk[miner_hotkey][position.trade_pair.trade_pair] += 1

                    miner_hotkey = line
                    if miner_hotkey not in position_to_corrections_per_hk:
                        position_to_corrections_per_hk[miner_hotkey] = defaultdict(list)
                        trade_pair_str_to_n_times_seen_per_hk[miner_hotkey] = defaultdict(int)

                elif len(parts) == 10:
                    trade_pair = parts[0]
                    timestamp_ms = int(parts[2])
                    original_price = float(parts[3])
                    corrected_price = float(parts[4]) if parts[4] != '?' else None
                    correction_status = parts[6]
                    if timestamp_ms not in order_timestamp_to_position:
                        bt.logging.warning(f'Ignoring correction from missing order miner {miner_hotkey}')
                        position = None
                        continue
                    assert miner_hotkey in hotkey_to_positions, f"Hotkey {miner_hotkey} not found in positions."
                    position = order_timestamp_to_position[timestamp_ms]
                    idx = trade_pair_str_to_n_times_seen_per_hk[miner_hotkey][trade_pair]
                    position_to_corrections_per_hk[miner_hotkey][position.position_uuid].append(
                        (trade_pair, timestamp_ms, original_price, corrected_price, correction_status, idx))
                else:
                    print('BREAKING AT LINE ', line)
                    break  # Done

                #print(f"Hotkey: {miner_hotkey}, Order Info: {order_info}")

        n_attempts = 0
        n_corrections = 0
        n_positions_grew_since_last_audit = 0
        unique_corrections = set()
        for hotkey, positions in hotkey_to_positions.items():
            position_to_corrections = position_to_corrections_per_hk.get(hotkey)
            if not position_to_corrections:
                continue

            for position in positions:
                assert position.position_uuid not in unique_corrections
                corrections = position_to_corrections.get(position.position_uuid, [])
                if not corrections:
                    continue

                new_order_prices = []
                any_price_corrections = False
                idx = None
                for correction in corrections:
                    trade_pair_str, timestamp_ms, original_price, corrected_price, correction_status, i = correction
                    trade_pair = TradePair.get_latest_tade_pair_from_trade_pair_str(trade_pair_str)
                    if idx is None:
                        idx = i
                    else:
                        assert idx == correction[-1]

                    if correction_status in ('price already accurate'):
                        new_order_prices.append(original_price)
                    elif corrected_price is None:
                        new_order_prices.append(original_price)
                    else:
                        new_order_prices.append(corrected_price)
                        any_price_corrections = True

                if not any_price_corrections:
                    continue

                if len(new_order_prices) == 0:
                    raise Exception(f"len(new_order_prices) == 0")
                elif len(new_order_prices) < len(position.orders):
                    n_positions_grew_since_last_audit += 1
                elif len(new_order_prices) > len(position.orders):
                    raise Exception(f"len(new_order_prices) {len(new_order_prices)} > len(position.orders) {len(position.orders)}")
                elif not new_order_prices:
                    raise Exception(f"len(new_order_prices) {len(new_order_prices)}")

                #bt.logging.error(f"Hotkey: {hotkey}, Position: {position.position_uuid}, Trade Pair: {position.trade_pair.trade_pair_id},")
                prev_n_attempts, prev_n_corrections = n_attempts, n_corrections
                n_attempts, n_corrections = self.correct_for_tp(positions, idx, new_order_prices,
                    trade_pair, n_attempts=n_attempts, n_corrections=n_corrections,
                    unique_corrections=unique_corrections, pos=position)
                if n_corrections == prev_n_corrections:
                    print(f"Failed to correct for hotkey {hotkey} trade pair {trade_pair.trade_pair_id} at idx {idx}")
        print(f"n_attempts: {n_attempts}, n_corrections: {n_corrections}, n_positions_corrected: {len(unique_corrections)},"
              f"n_positions_grew_since_last_audit: {n_positions_grew_since_last_audit}")
        assert n_corrections == n_attempts, f"n_corrections {n_corrections} != n_attempts {n_attempts}"

    def generate_correction_templates(self):
        f = open('/Users/jbonilla/Documents/price_audit_4-18-24.txt', 'w')
        ALL_MSGS = []
        ELIM_MSGS = []

        eliminations = self.get_miner_eliminations_from_disk()
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)

        template = ', '.join(['trade_pair', 'order_date_utc', 'order_timestamp', 'price_on_dash', 'corrected_price', 'percent_change', 'automated', 'price_freshness_ms', 'current position return', 'new position return'])
        f.write(template + '\n')
        hotkey_to_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)
        for miner_hotkey, positions in hotkey_to_positions.items():
            if miner_hotkey in eliminated_hotkeys:
                continue
            msg = f'Processing hotkey {miner_hotkey}'
            f.write(miner_hotkey + '\n')
            print(msg)
            orig_return = 1.0

            for position in positions:
                pending_rows = []
                orig_return *= position.return_at_close
                for order in position.orders:
                    order_already_calibrated = bool(order.price_sources)
                    date_utc = TimeUtil.millis_to_formatted_date_str(order.processed_ms)
                    closest_price, smallest_delta_ms = self.live_price_fetcher.get_close_at_date(position.trade_pair, order.processed_ms)
                    automated = 'needs human edit'
                    corrected_price = '?'
                    original_price = order.price
                    if order_already_calibrated:
                        automated = 'price already accurate'
                        corrected_price = original_price
                        delta_ms = order.price_sources[0].lag_ms
                    elif smallest_delta_ms <= 1000:
                        automated = 'automatically corrected'
                        corrected_price = closest_price
                        delta_ms = smallest_delta_ms
                        order.price = corrected_price
                    else:
                        delta_ms = smallest_delta_ms
                    if corrected_price != '?' and corrected_price != original_price:
                        percent_change = f"{(corrected_price - original_price) / original_price * 100:.2f}%"
                    else:
                        percent_change = 'N/A'
                    pending_rows.append([position.trade_pair.trade_pair, date_utc, order.processed_ms, original_price, corrected_price, percent_change, automated, delta_ms, position.return_at_close])

                temp = deepcopy(position)
                temp.rebuild_position_with_updated_orders()
                if position.is_closed_position:
                    new_return = temp.return_at_close
                else:
                    new_return = 'depends on live price'
                for row in pending_rows:
                    row.append(new_return)
                    f.write(', '.join([str(x) for x in row]))
                    f.write('\n')

            self.check_elimination(positions, miner_hotkey, orig_return, ALL_MSGS, ELIM_MSGS)

        for x in ALL_MSGS:
            print(x)
            f.write(str(x) + '\n')

        for x in ELIM_MSGS:
            print(x)
            f.write(str(x) + '\n')

        f.close()
