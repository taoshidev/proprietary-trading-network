import shutil
import time

from vali_config import ValiConfig
from shared_objects.challenge_utils import ChallengeBase
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt

class EliminationManager(ChallengeBase):
    def __init__(self, metagraph):
        super().__init__(metagraph=metagraph)

    def process_eliminations(self):
        if time.time() - self.get_last_update_time() < ValiConfig.ELIMINATION_CHECK_INTERVAL_S:
            return
        bt.logging.info("running elimination manager")
        self._load_latest_eliminations_from_disk()
        self._handle_plagiarism_eliminations()
        self._delete_eliminated_expired_miners()
        self.set_last_update_time()


    def _handle_plagiarism_eliminations(self):
        existing_plagiarism_eliminations = set(x for x in self.eliminations if x['reason'] == 'plagiarism')
        self._load_latest_eliminations_from_disk()
        bt.logging.debug("checking plagiarism.")
        self._load_latest_miner_plagiarism_from_cache()
        # miner_copying_json[miner_hotkey] = current_hotkey_mc
        for miner_hotkey, current_plagiarism_score in self.miner_plagiarism_scores.items():
            if miner_hotkey in existing_plagiarism_eliminations:
                continue
            if current_plagiarism_score > ValiConfig.MAX_MINER_PLAGIARISM_SCORE:
                self.eliminations.append(self.deregister_and_generate_elimination_row(miner_hotkey, -1, 'plagiarism'))
                bt.logging.info(f"miner eliminated with hotkey [{miner_hotkey}] with plagiarism score of [{current_plagiarism_score}]")

        self._write_eliminations_from_memory_to_disk()
        

    def _delete_eliminated_expired_miners(self):
        updated_eliminations = []
        for x in self.eliminations: # self.eliminations were just refreshed in _handle_plagiarism_eliminations
            hotkey = x['hotkey']
            dereg_time = x['dereg_time']
            # Don't delete this miner until it hits the minimum elimination time.
            if time.time() - dereg_time < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_S:
                updated_eliminations.append(x)
                continue
            dd = x.get('dd', 'N/A')
            miner_dir = ValiBkpUtils.get_miner_dir()
            bt.logging.info(
                f"miner eliminated with hotkey [{hotkey}] with max dd of [{dd}]. reason: [{x['reason']}]"
                f"Removing miner dir [{miner_dir}]"
            )
            try:
                shutil.rmtree(miner_dir)
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found [{miner_dir}]")
                
        if len(updated_eliminations) != len(self.eliminations):
            self.eliminations = updated_eliminations
            self._write_eliminations_from_memory_to_disk()
            self.set_last_update_time()