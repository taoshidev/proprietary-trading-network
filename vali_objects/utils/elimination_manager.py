# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import shutil

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter

import bittensor as bt


class EliminationManager(CacheController):
    """"
    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """

    def __init__(self, metagraph, position_manager, eliminations_lock, running_unit_tests=False, shutdown_dict=None):
        super().__init__(metagraph=metagraph)
        self.position_manager = position_manager
        self.eliminations_lock = eliminations_lock
        self.shutdown_dict = shutdown_dict
        assert running_unit_tests == self.position_manager.running_unit_tests

    def process_eliminations(self):
        if not self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS):
            return
        bt.logging.info("running elimination manager")
        with self.eliminations_lock:
            self.eliminations = self.get_eliminations_from_disk()
        # self._handle_plagiarism_eliminations()
        self._delete_eliminated_expired_miners()
        self._eliminate_MDD()
        self.set_last_update_time()

    def _handle_plagiarism_eliminations(self):
        bt.logging.debug("checking plagiarism.")
        if self.shutdown_dict:
            return
        self._refresh_plagiarism_scores_in_memory_and_disk()
        # miner_copying_json[miner_hotkey] = current_hotkey_mc
        for miner_hotkey, current_plagiarism_score in self.miner_plagiarism_scores.items():
            if self.shutdown_dict:
                return
            if self._hotkey_in_eliminations(miner_hotkey):
                continue
            if current_plagiarism_score > ValiConfig.MAX_MINER_PLAGIARISM_SCORE:
                self.position_manager.handle_eliminated_miner(miner_hotkey, {})
                self.append_elimination_row(miner_hotkey, -1, 'plagiarism')
                bt.logging.info(
                    f"miner eliminated with hotkey [{miner_hotkey}] with plagiarism score of [{current_plagiarism_score}]")
        with self.eliminations_lock:
            self._write_eliminations_from_memory_to_disk()

    def _delete_eliminated_expired_miners(self):
        eliminated_hotkeys = set()
        self._refresh_challengeperiod_in_memory()
        # self.eliminations were just refreshed in process_eliminations
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
            if hotkey in self.challengeperiod_testing:
                self.challengeperiod_testing.pop(hotkey)

            if hotkey in self.challengeperiod_success:
                self.challengeperiod_success.pop(hotkey)

            miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests) + hotkey
            try:
                shutil.rmtree(miner_dir)
                bt.logging.info(
                    f"miner eliminated with hotkey [{hotkey}] with max dd of [{x.get('dd', 'N/A')}]. reason: [{x['reason']}]"
                    f"Removing miner dir [{miner_dir}]"
                )
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found. Already deleted. [{miner_dir}]")
            eliminated_hotkeys.add(hotkey)

        # Write the challengeperiod information to disk
        self._write_challengeperiod_from_memory_to_disk()

        if eliminated_hotkeys:
            self.eliminations = [x for x in self.eliminations if x['hotkey'] not in eliminated_hotkeys]
            with self.eliminations_lock:
                self._write_eliminations_from_memory_to_disk()
            self.set_last_update_time()

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

    def _eliminate_MDD(self):
        """
        Checks the mdd of each miner and eliminates any miners that surpass MAX_TOTAL_DRAWDOWN
        """
        bt.logging.debug("checking for maximum drawdown.")
        if self.shutdown_dict:
            return

        subtensor_weight_setter = SubtensorWeightSetter(
            config=None,
            wallet=None,
            metagraph=None,
            running_unit_tests=False
        )

        # Collect information from the disk and populate variables in memory
        subtensor_weight_setter._refresh_eliminations_in_memory()
        subtensor_weight_setter._refresh_challengeperiod_in_memory()

        # Get the hotkeys
        challengeperiod_testing_hotkeys = subtensor_weight_setter.challengeperiod_testing.keys()
        challengeperiod_success_hotkeys = subtensor_weight_setter.challengeperiod_success.keys()

        # full ledger of all miner hotkeys
        all_miner_hotkeys = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys

        filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=all_miner_hotkeys)

        for miner_hotkey, ledger in filtered_ledger.items():
            if self.shutdown_dict:
                return
            if self._hotkey_in_eliminations(miner_hotkey):
                continue

            miner_cps = ledger.cps
            if miner_cps is None or len(miner_cps) == 0:
                continue

            miner_mdd = max([miner_cps.mdd for miner_cps in miner_cps])

            if miner_mdd < ValiConfig.MAX_TOTAL_DRAWDOWN:
                self.position_manager.handle_eliminated_miner(miner_hotkey, {})
                self.append_elimination_row(miner_hotkey, -1, 'MAX_TOTAL_DRAWDOWN')
                bt.logging.info(
                    f"miner eliminated with hotkey [{miner_hotkey}] with drawdown [{miner_mdd}]")
        with self.eliminations_lock:
            self._write_eliminations_from_memory_to_disk()


