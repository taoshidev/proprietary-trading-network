import shutil
import time

from vali_config import ValiConfig
from vali_objects.utils.challenge_utils import ChallengeBase
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt

class EliminationManager(ChallengeBase):
    def __init__(self):
        super().__init__()   

    def process_eliminations(self):
        if time.time() - self.last_update_time_s < ValiConfig.ELIMINATION_CHECK_INTERVAL_S:
            return
        self._load_eliminations_from_cache()
        self._handle_plagiarism_eliminations()
        self._delete_eliminated_expired_miners()
        self.last_update_time_s = time.time()


    def _handle_plagiarism_eliminations(self):
        existing_plagiarism_eliminations = set(x for x in self.eliminations if x['reason'] == 'plagiarism')
        self._load_eliminations_from_cache()
        bt.logging.debug("checking plagiarism.")
        miner_copying_json = self._load_miner_copying_from_cache()
        # miner_copying_json[miner_hotkey] = current_hotkey_mc
        for miner_hotkey, current_hotkey_mc in miner_copying_json.items():
            if miner_hotkey in existing_plagiarism_eliminations:
                continue
            if current_hotkey_mc > ValiConfig.MAX_MINER_COPYING:
                self.eliminations.append({'hotkey': miner_hotkey, 'elimination_time': time.now(), 'reason': 'plagiarism', 'dd':-1})
                bt.logging.info(f"miner eliminated with hotkey [{miner_hotkey}] with max mc of [{current_hotkey_mc}]. reason: plagiarism")

        self._write_updated_eliminations()
        

    def _delete_eliminated_expired_miners(self):
        updated_eliminations = []
        for x in self.eliminations: # eliminations was just updated in _handle_plagiarism_eliminations
            hotkey = x['hotkey']
            elimination_time = x['elimination_time']
            # Don't delete this miner until it hits the minimum elimination time.
            if time.time() - elimination_time < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_S:
                updated_eliminations.append(x)
                continue
            dd = x.get('dd', 'N/A')
            miner_dir = ValiBkpUtils.get_miner_dir(hotkey)
            bt.logging.info(
                f"miner eliminated with hotkey [{hotkey}] with max dd of [{dd}]. reason: [{x['reason']}]"
                f"Removing miner dir [{miner_dir}]"
            )
            try:
                shutil.rmtree(miner_dir)
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found [{miner_dir}]")
                
        if len(updated_eliminations) != len(self.eliminations):
            self._write_updated_eliminations(updated_eliminations)
            self.last_update_time_s = time.time()