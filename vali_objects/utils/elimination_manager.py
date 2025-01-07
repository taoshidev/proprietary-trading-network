# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import shutil
import threading
from copy import deepcopy

from time_util.time_util import TimeUtil
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt


class EliminationManager(CacheController):
    """"
    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """

    def __init__(self, metagraph, position_manager, challengeperiod_manager, running_unit_tests=False, shutdown_dict=None):
        super().__init__(metagraph=metagraph)
        self.position_manager = position_manager
        self.eliminations_lock = threading.Lock()
        self.shutdown_dict = shutdown_dict
        self.challengeperiod_manager = challengeperiod_manager
        self.running_unit_tests = running_unit_tests

        self.eliminations = self.get_miner_eliminations_from_disk()
        if len(self.eliminations) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                {CacheController.ELIMINATIONS: []}
            )

    def process_eliminations(self):
        if not self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS):
            return
        bt.logging.info("running elimination manager")
        self.eliminations = self.get_eliminations_from_disk()
        # self._handle_plagiarism_eliminations()
        self._delete_eliminated_expired_miners()
        self.set_last_update_time()

    def _handle_plagiarism_eliminations(self):
        bt.logging.debug("checking plagiarism.")
        if self.shutdown_dict:
            return
        self.challengeperiod_manager._refresh_plagiarism_scores_in_memory_and_disk()
        # miner_copying_json[miner_hotkey] = current_hotkey_mc
        for miner_hotkey, current_plagiarism_score in self.challengeperiod_manager.miner_plagiarism_scores.items():
            if self.shutdown_dict:
                return
            if self._hotkey_in_eliminations(miner_hotkey):
                continue
            if current_plagiarism_score > ValiConfig.MAX_MINER_PLAGIARISM_SCORE:
                self.position_manager.handle_eliminated_miner(miner_hotkey, {})
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

    def _hotkey_in_eliminations(self, hotkey):
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
            try:
                shutil.rmtree(miner_dir)
                bt.logging.info(
                    f"miner eliminated with hotkey [{hotkey}] with max dd of [{x.get('dd', 'N/A')}]. reason: [{x['reason']}]"
                    f"Removing miner dir [{miner_dir}]"
                )
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found. Already deleted. [{miner_dir}]")
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
        self.eliminations = []

    def _refresh_eliminations_in_memory(self):
        self.eliminations = self.get_eliminations_from_disk()

    def get_eliminated_hotkeys(self):
        return set([x['hotkey'] for x in self.eliminations]) if self.eliminations else set()

    def get_eliminations_from_memory(self):
        return deepcopy(self.eliminations)

    def get_eliminations_from_disk(self):
        with self.eliminations_lock:
            location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
            cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
            bt.logging.trace(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
            return cached_eliminations

    def get_miner_eliminations_from_disk(self) -> list:
        return ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests), CacheController.ELIMINATIONS
        )

    def append_elimination_row(self, hotkey, current_dd, mdd_failure, t_ms=None, price_info=None, return_info=None):
        with self.eliminations_lock:
            elimination_row = self.generate_elimination_row(hotkey, current_dd, mdd_failure, t_ms=t_ms,
                                                            price_info=price_info, return_info=return_info)
            self.eliminations.append(elimination_row)
            self.save_eliminations()

    def delete_elimination(self, hotkey):
        with self.eliminations_lock:
            self.eliminations = [x for x in self.eliminations if x['hotkey'] != hotkey]
            self.save_eliminations()

    def delete_eliminations(self, deleted_hotkeys):
        with self.eliminations_lock:
            self.eliminations = [x for x in self.eliminations if x['hotkey'] not in deleted_hotkeys]
            self.save_eliminations()
