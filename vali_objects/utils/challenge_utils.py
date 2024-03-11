import time
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

class ChallengeBase:
    def __init__(self, config=None, metagraph=None):
        self.config = config
        if config is not None:
            self.subtensor = bt.subtensor(config=config)
        else:
            self.subtensor = None

        self.metagraph = metagraph # Refreshes happen on validator
        self.last_update_time_s = 0
        self.eliminations = None
        self.miner_plagiarism_scores = None

    def deregister_and_generate_elimination_row(self, hotkey, dd, reason):

        return {'hotkey': hotkey, 'dereg_time': time.time(), 'dd': dd, 'reason': reason}

    def _write_eliminations_from_memory_to_disk(self):
        vali_elims = {ValiUtils.ELIMINATIONS: self.eliminations}
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(), vali_elims)

    def _write_updated_plagiarism_scores_from_memory_to_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_miner_copying_dir(), self.miner_plagiarism_scores)

    def _load_latest_eliminations_from_disk(self):
        cached_eliminations = ValiUtils.get_vali_json_file(ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS)
        updated_eliminations = [elimination for elimination in cached_eliminations if elimination in self.metagraph.hotkeys]
        self.eliminations = updated_eliminations
        self._write_eliminations_from_memory_to_disk()

    def _load_latest_miner_plagiarism_from_cache(self):
        cached_miner_plagiarism = ValiUtils.get_vali_json_file(ValiBkpUtils.get_miner_copying_dir())
        self.miner_plagiarism_scores = {mch: mc for mch, mc in cached_miner_plagiarism.items() if mch in self.metagraph.hotkeys}
        self._write_updated_plagiarism_scores_from_memory_to_disk()

