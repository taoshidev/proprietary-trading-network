# developer: jbonilla
# Copyright © 2023 Taoshi Inc

import time

from time_util.time_util import TimeUtil
from vali_objects.utils.position_utils import PositionUtils
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

        self.metagraph = metagraph  # Refreshes happen on validator
        self._last_update_time_ms = 0
        self.eliminations = []
        self.miner_plagiarism_scores = {}

    def get_last_update_time_ms(self):
        return self._last_update_time_ms

    def refresh_allowed(self, refresh_interval_ms):
        return TimeUtil.now_in_millis() - self.get_last_update_time_ms() > refresh_interval_ms

    def set_last_update_time(self):
        # Log that the class has finished updating and the time it finished updating
        bt.logging.success(f"Finished updating class {self.__class__.__name__}")
        self._last_update_time_ms = TimeUtil.now_in_millis()

    @staticmethod
    def generate_elimination_row(hotkey, dd, reason):
        return {'hotkey': hotkey, 'elimination_initiated_time_ms': TimeUtil.now_in_millis(), 'dd': dd, 'reason': reason}
    def close_positions_and_append_elimination_row(self, hotkey, dd, reason):
        """
         We are closing the positions here. By adding the miner to the elimination file, we ensure that subsequent orders
         are not allowed to be placed. Orders can be placed again after the elimination time has passed and all old
         miner positions have been deleted. The elimination time gives bittensor sufficient time to deregister the
         miner. However, deregistration isn't guaranteed if there are few miners on the network.

         Close positions before writing elimination to disk. This is because the miner will be immediately
         blacklisted and unable to modify their data anymore.
        """
        #
        open_positions = PositionUtils.get_all_miner_positions(hotkey, only_open_positions=True)
        bt.logging.info(f"Closing [{len(open_positions)}] positions for hotkey: {hotkey}")
        for open_position in open_positions:
            open_position.close_out_position(TimeUtil.now_in_millis())
            ValiUtils.save_miner_position_to_disk(open_position)

        r = ChallengeBase.generate_elimination_row(hotkey, dd, reason)
        bt.logging.info(f"Created elimination row: {r}")
        self.eliminations.append(r)



    def _write_eliminations_from_memory_to_disk(self):
        vali_elims = {ValiUtils.ELIMINATIONS: self.eliminations}
        bt.logging.info(f"Writing [{len(self.eliminations)}] eliminations from memory to disk: {vali_elims}")
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(), vali_elims)

    @staticmethod
    def clear_eliminations_from_disk():
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(), {ValiUtils.ELIMINATIONS: []})

    @staticmethod
    def clear_plagiarism_scores_from_disk():
        ValiBkpUtils.write_file(ValiBkpUtils.get_miner_copying_dir(), {})

    def _write_updated_plagiarism_scores_from_memory_to_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_miner_copying_dir(), self.miner_plagiarism_scores)

    def _refresh_eliminations_in_memory_and_disk(self):
        self.eliminations = ChallengeBase.get_filtered_eliminations_from_disk(self.metagraph.hotkeys)
        self._write_eliminations_from_memory_to_disk()

    @staticmethod
    def get_filtered_eliminations_from_disk(hotkeys):
        cached_eliminations = ValiUtils.get_vali_json_file(ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS)
        bt.logging.info(f"Loaded [{len(cached_eliminations)}] eliminations from disk: {cached_eliminations}")
        updated_eliminations = [elimination for elimination in cached_eliminations if
                                elimination['hotkey'] in hotkeys]
        return updated_eliminations

    def _refresh_plagiarism_scores_in_memory_and_disk(self):
        cached_miner_plagiarism = ValiUtils.get_vali_json_file(ValiBkpUtils.get_miner_copying_dir())
        self.miner_plagiarism_scores = {mch: mc for mch, mc in cached_miner_plagiarism.items() if
                                        mch in self.metagraph.hotkeys}
        self._write_updated_plagiarism_scores_from_memory_to_disk()
