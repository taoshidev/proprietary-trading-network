import shutil
import time

from vali_config import ValiConfig
from vali_objects.utils.challenge_utils import ChallengeBase
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt

class CacheCleaner(ChallengeBase):
    def __init__(self):
        super().__init__()   

    def _delete_eliminated_miners(self):
        if time.time() - self.last_update_time_s < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_S:
            return
        updated_eliminations = []
        eliminations_snapshot = self._load_eliminations_from_cache()
        for x in eliminations_snapshot:
            hotkey = x['hotkey']
            elimination_time = x['elimination_time']
            # Don't delete this miner until it hits the minimum elimination time.
            if time.time() - elimination_time < ValiConfig.MINER_ELIMINATION_TIME_S:
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
                
        if len(updated_eliminations) != len(eliminations_snapshot):
            self._write_updated_eliminations(updated_eliminations)
            self.last_update_time_s = time.time()