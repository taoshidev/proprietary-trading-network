# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import shutil
from copy import deepcopy
from typing import Dict

from time_util.time_util import TimeUtil
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource


class EliminationManager(CacheController):
    """"
    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """

    def __init__(self, metagraph, position_manager, challengeperiod_manager,
                 running_unit_tests=False, shutdown_dict=None, ipc_manager=None):
        super().__init__(metagraph=metagraph)
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict
        self.challengeperiod_manager = challengeperiod_manager
        self.running_unit_tests = running_unit_tests

        if ipc_manager:
            self.eliminations = ipc_manager.list()
        else:
            self.eliminations = []
        self.eliminations.extend(self.get_miner_eliminations_from_disk())
        if len(self.eliminations) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                {CacheController.ELIMINATIONS: []}
            )

    def handle_perf_ledger_eliminations(self, position_locks):
        perf_ledger_eliminations = self.position_manager.perf_ledger_manager.get_perf_ledger_eliminations()
        n_eliminations = 0
        for e in perf_ledger_eliminations:
            if self.hotkey_in_eliminations(e['hotkey']):
                continue

            n_eliminations += 1
            self.eliminations.append(e)

            price_info = e['price_info']
            trade_pair_to_price_source_used_for_elimination_check = {}
            for k, v in price_info.items():
                trade_pair = TradePair.get_latest_tade_pair_from_trade_pair_str(k)
                elimination_initiated_time_ms = e['elimination_initiated_time_ms']
                trade_pair_to_price_source_used_for_elimination_check[trade_pair] = PriceSource(source='elim', open=v, close=v, start_ms=elimination_initiated_time_ms, timespan_ms=1000, websocket=False, trade_pair=trade_pair)
            self.handle_eliminated_miner(e['hotkey'], trade_pair_to_price_source_used_for_elimination_check, position_locks)

        if n_eliminations:
            self.save_eliminations()
            bt.logging.info(f'Wrote {n_eliminations} perf ledger eliminations to disk')

    def handle_eliminated_miner(self, hotkey: str,
                                trade_pair_to_price_source_used_for_elimination_check: Dict[TradePair, PriceSource],
                                position_locks,
                                open_position_trade_pairs=None):

        tps_to_iterate_over = open_position_trade_pairs if open_position_trade_pairs else TradePair
        for trade_pair in tps_to_iterate_over:
            with position_locks.get_lock(hotkey, trade_pair.trade_pair_id):
                open_position = self.position_manager.get_open_position_for_a_miner_trade_pair(hotkey, trade_pair.trade_pair_id)
                source_for_elimination = trade_pair_to_price_source_used_for_elimination_check.get(trade_pair)
                if open_position:
                    bt.logging.info(
                        f"Closing open position for hotkey: {hotkey} and trade_pair: {trade_pair.trade_pair_id}. "
                        f"Source for elimination {source_for_elimination}")
                    open_position.close_out_position(TimeUtil.now_in_millis())
                    if source_for_elimination:
                        open_position.orders[-1].price_sources.append(source_for_elimination)
                    self.position_manager.save_miner_position(open_position)

    def process_eliminations(self, position_locks):
        if not self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS):
            return
        bt.logging.info("running elimination manager")
        # self._handle_plagiarism_eliminations()
        self.handle_perf_ledger_eliminations(position_locks)
        self._delete_eliminated_expired_miners()
        self.set_last_update_time()

    def _handle_plagiarism_eliminations(self, position_locks):
        bt.logging.debug("checking plagiarism.")
        if self.shutdown_dict:
            return
        self.challengeperiod_manager._refresh_plagiarism_scores_in_memory_and_disk()
        # miner_copying_json[miner_hotkey] = current_hotkey_mc
        for miner_hotkey, current_plagiarism_score in self.challengeperiod_manager.miner_plagiarism_scores.items():
            if self.shutdown_dict:
                return
            if self.hotkey_in_eliminations(miner_hotkey):
                continue
            if current_plagiarism_score > ValiConfig.MAX_MINER_PLAGIARISM_SCORE:
                self.handle_eliminated_miner(miner_hotkey, {}, position_locks)
                self.append_elimination_row(miner_hotkey, -1, 'plagiarism')
                bt.logging.info(
                    f"miner eliminated with hotkey [{miner_hotkey}] with plagiarism score of [{current_plagiarism_score}]")

    def is_zombie_hotkey(self, hotkey):
        if not isinstance(self.eliminations, list):
            return False

        if hotkey in self.metagraph.hotkeys:
            return False

        if any(x['hotkey'] == hotkey for x in self.eliminations):
            return False

        return True

    def hotkey_in_eliminations(self, hotkey):
        for x in self.eliminations:
            if x['hotkey'] == hotkey:
                return deepcopy(x)
        return None

    def _delete_eliminated_expired_miners(self):
        deleted_hotkeys = set()
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()
        # self.eliminations were just refreshed in process_eliminations
        any_challenege_period_changes = False
        for x in self.eliminations:
            if self.shutdown_dict:
                return
            hotkey = x['hotkey']
            elimination_initiated_time_ms = x['elimination_initiated_time_ms']
            # Don't delete this miner until it hits the minimum elimination time.
            if TimeUtil.now_in_millis() - elimination_initiated_time_ms < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS:
                continue
            # We will not delete this miner's cache until it has been deregistered by BT
            if hotkey in self.metagraph.hotkeys:
                bt.logging.trace(f"miner [{hotkey}] has not been deregistered by BT yet. Not deleting miner dir.")
                continue

            # If the miner is no longer in the metagraph, we can remove them from the challengeperiod information
            if hotkey in self.challengeperiod_manager.challengeperiod_testing:
                any_challenege_period_changes = True
                self.challengeperiod_manager.challengeperiod_testing.pop(hotkey)

            if hotkey in self.challengeperiod_manager.challengeperiod_success:
                any_challenege_period_changes = True
                self.challengeperiod_manager.challengeperiod_success.pop(hotkey)

            miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests) + hotkey
            all_positions = self.position_manager.get_positions_for_one_hotkey(hotkey)
            for p in all_positions:
                self.position_manager.delete_position(p)
            try:
                shutil.rmtree(miner_dir)
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found. Already deleted. [{miner_dir}]")
            bt.logging.info(
                f"miner eliminated with hotkey [{hotkey}] with max dd of [{x.get('dd', 'N/A')}]. reason: [{x['reason']}]"
                f"Removing miner dir [{miner_dir}]"
            )
            deleted_hotkeys.add(hotkey)

        # Write the challengeperiod information to disk
        if any_challenege_period_changes:
            self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        if deleted_hotkeys:
            self.delete_eliminations(deleted_hotkeys)

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        for hotkey in CacheController.get_directory_names(all_miners_dir):
            if self.shutdown_dict:
                return
            miner_dir = all_miners_dir + hotkey
            if self.is_zombie_hotkey(hotkey):
                try:
                    shutil.rmtree(miner_dir)
                    bt.logging.info(f"Zombie miner dir removed [{miner_dir}]")
                except FileNotFoundError:
                    bt.logging.info(f"Zombie miner dir not found. Already deleted. [{miner_dir}]")

    def save_eliminations(self):
        self.write_eliminations_to_disk(self.eliminations)

    def write_eliminations_to_disk(self, eliminations):
        vali_eliminations = {CacheController.ELIMINATIONS: eliminations}
        bt.logging.trace(f"Writing [{len(eliminations)}] eliminations from memory to disk: {vali_eliminations}")
        output_location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, vali_eliminations)

    def clear_eliminations(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                                {CacheController.ELIMINATIONS: []})
        del self.eliminations[:]

    def get_eliminated_hotkeys(self):
        return set([x['hotkey'] for x in self.eliminations]) if self.eliminations else set()

    def get_eliminations_from_memory(self):
        return deepcopy(self.eliminations)

    def get_eliminations_from_disk(self):
        #with self.eliminations_lock:
            location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
            cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
            bt.logging.trace(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
            return cached_eliminations

    def get_miner_eliminations_from_disk(self) -> list:
        return ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests), CacheController.ELIMINATIONS
        )

    def append_elimination_row(self, hotkey, current_dd, mdd_failure, t_ms=None, price_info=None, return_info=None):
        #with self.eliminations_lock:
            elimination_row = self.generate_elimination_row(hotkey, current_dd, mdd_failure, t_ms=t_ms,
                                                            price_info=price_info, return_info=return_info)
            self.eliminations.append(elimination_row)
            self.save_eliminations()

    def delete_eliminations(self, deleted_hotkeys):
        #with self.eliminations_lock:
        items_to_remove = [x for x in self.eliminations if x['hotkey'] in deleted_hotkeys]
        for item in items_to_remove:
            self.eliminations.remove(item)
        self.save_eliminations()
