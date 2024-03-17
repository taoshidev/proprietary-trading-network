# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import shutil

from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt

class EliminationManager(CacheController):
    """"
    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """
    def __init__(self, metagraph, running_unit_tests=False):
        super().__init__(metagraph=metagraph)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)

    def process_eliminations(self):
        if not self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS):
            return
        bt.logging.info("running elimination manager")
        self._refresh_eliminations_in_memory_and_disk()
        self._handle_plagiarism_eliminations()
        self._delete_eliminated_expired_miners()
        self.set_last_update_time()


    def _handle_plagiarism_eliminations(self):
        existing_plagiarism_eliminations = set(x for x in self.eliminations if x['reason'] == 'plagiarism')
        self._refresh_eliminations_in_memory_and_disk()
        bt.logging.debug("checking plagiarism.")
        self._refresh_plagiarism_scores_in_memory_and_disk()
        # miner_copying_json[miner_hotkey] = current_hotkey_mc
        for miner_hotkey, current_plagiarism_score in self.miner_plagiarism_scores.items():
            if miner_hotkey in existing_plagiarism_eliminations:
                continue
            if current_plagiarism_score > ValiConfig.MAX_MINER_PLAGIARISM_SCORE:
                self.position_manager.close_open_positions_for_miner(miner_hotkey)
                self.append_elimination_row(miner_hotkey, -1, 'plagiarism')
                bt.logging.info(f"miner eliminated with hotkey [{miner_hotkey}] with plagiarism score of [{current_plagiarism_score}]")

        self._write_eliminations_from_memory_to_disk()
        

    def _delete_eliminated_expired_miners(self):
        eliminated_hotkeys = set()
        # self.eliminations were just refreshed in _load_latest_eliminations_from_disk and _handle_plagiarism_eliminations
        for x in self.eliminations:
            hotkey = x['hotkey']
            elimination_initiated_time_ms = x['elimination_initiated_time_ms']
            # Don't delete this miner until it hits the minimum elimination time.
            if self.refresh_allowed(ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS):
                continue
            # We will not delete this miner's cache until it has been deregistered by BT
            if hotkey in self.metagraph.hotkeys:
                bt.logging.info(f"miner [{hotkey}] has not been deregistered by BT yet. Not deleting miner dir.")
                continue
            miner_dir = ValiBkpUtils.get_miner_dir()
            try:
                shutil.rmtree(miner_dir)
                bt.logging.info(
                    f"miner eliminated with hotkey [{hotkey}] with max dd of [{x.get('dd', 'N/A')}]. reason: [{x['reason']}]"
                    f"Removing miner dir [{miner_dir}]"
                )
                eliminated_hotkeys.add(hotkey)
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found [{miner_dir}]")
                
        if eliminated_hotkeys:
            self.eliminations = [x for x in self.eliminations if x['hotkey'] not in eliminated_hotkeys]
            self._write_eliminations_from_memory_to_disk()
            self.set_last_update_time()