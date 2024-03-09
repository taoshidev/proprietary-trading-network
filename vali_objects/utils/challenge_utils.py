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
        self.miner_copying = None

    def _write_updated_eliminations(self, updated_eliminations):
        vali_elims = {ValiUtils.ELIMINATIONS: updated_eliminations}
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(), vali_elims)

    def _write_updated_copying(self, updated_miner_copying):
        ValiBkpUtils.write_file(ValiBkpUtils.get_miner_copying_dir(), updated_miner_copying)

    def _load_eliminations_from_cache(self):
        cached_eliminations = ValiUtils.get_vali_json_file(ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS)
        updated_eliminations = [elimination for elimination in cached_eliminations if elimination in self.metagraph.hotkeys]
        if len(updated_eliminations) < len(cached_eliminations):
            self._write_updated_eliminations(updated_eliminations)
        self.eliminations = updated_eliminations

    def _load_miner_copying_from_cache(self):
        cached_miner_copying = ValiUtils.get_vali_json_file(ValiBkpUtils.get_miner_copying_dir())
        updated_miner_copying = {mch: mc for mch, mc in cached_miner_copying.items() if mch in self.metagraph.hotkeys}
        if len(updated_miner_copying) < len(cached_miner_copying):
            self._write_updated_copying(updated_miner_copying)
        self.miner_copying = updated_miner_copying
