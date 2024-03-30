# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import datetime
import threading

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt


class CacheController:
    ELIMINATIONS = "eliminations"
    MAX_DAILY_DRAWDOWN = 'MAX_DAILY_DRAWDOWN'
    MAX_TOTAL_DRAWDOWN = 'MAX_TOTAL_DRAWDOWN'
    def __init__(self, config=None, metagraph=None, running_unit_tests=False):
        self.config = config
        if config is not None:
            self.subtensor = bt.subtensor(config=config)
        else:
            self.subtensor = None

        self.running_unit_tests = running_unit_tests
        self.metagraph = metagraph  # Refreshes happen on validator
        self._last_update_time_ms = 0
        self.eliminations = []
        self.miner_plagiarism_scores = {}

    def get_last_update_time_ms(self):
        return self._last_update_time_ms

    def _hotkey_in_eliminations(self, hotkey):
        return any(hotkey == x['hotkey'] for x in self.eliminations)

    @staticmethod
    def generate_elimination_row(hotkey, dd, reason):
        return {'hotkey': hotkey, 'elimination_initiated_time_ms': TimeUtil.now_in_millis(), 'dd': dd, 'reason': reason}

    def refresh_allowed(self, refresh_interval_ms):
        return TimeUtil.now_in_millis() - self.get_last_update_time_ms() > refresh_interval_ms

    def set_last_update_time(self):
        # Log that the class has finished updating and the time it finished updating
        bt.logging.success(f"Finished updating class {self.__class__.__name__}")
        self._last_update_time_ms = TimeUtil.now_in_millis()

    def _write_eliminations_from_memory_to_disk(self):
        self.write_eliminations_to_disk(self.eliminations)

    def write_eliminations_to_disk(self, eliminations):
        vali_eliminations = {CacheController.ELIMINATIONS: eliminations}
        bt.logging.info(f"Writing [{len(eliminations)}] eliminations from memory to disk: {vali_eliminations}")
        output_location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, vali_eliminations)

    def clear_eliminations_from_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests), {CacheController.ELIMINATIONS: []})

    def clear_plagiarism_scores_from_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_plagiarism_scores_file_location(running_unit_tests=self.running_unit_tests), {})

    def _write_updated_plagiarism_scores_from_memory_to_disk(self):
        self.write_plagiarism_scores_to_disk(self.miner_plagiarism_scores)

    def write_plagiarism_scores_to_disk(self, scores):
        ValiBkpUtils.write_file(ValiBkpUtils.get_plagiarism_scores_file_location(
            running_unit_tests=self.running_unit_tests), scores)

    def _refresh_eliminations_in_memory_and_disk(self):
        self.eliminations = self.get_filtered_eliminations_from_disk()
        self._write_eliminations_from_memory_to_disk()

    def _refresh_eliminations_in_memory(self):
        self.eliminations = self.get_eliminations_from_disk()

    def get_filtered_eliminations_from_disk(self):
        # Filters out miners that have already been deregistered. (Not in the metagraph)
        # This allows the miner to participate again once they re-register
        cached_eliminations = self.get_eliminations_from_disk()
        updated_eliminations = [elimination for elimination in cached_eliminations if
                                elimination['hotkey'] in self.metagraph.hotkeys]
        if len(updated_eliminations) != len(cached_eliminations):
            bt.logging.info(f"Filtered [{len(cached_eliminations) - len(updated_eliminations)}] / "
                            f"{len(cached_eliminations)} eliminations from disk due to not being in the metagraph")
        return updated_eliminations

    def get_eliminations_from_disk(self):
        location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
        bt.logging.info(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
        return cached_eliminations

    def get_plagiarism_scores_from_disk(self):
        location = ValiBkpUtils.get_plagiarism_scores_file_location(running_unit_tests=self.running_unit_tests)
        ans = ValiUtils.get_vali_json_file(location)
        bt.logging.info(f"Loaded [{len(ans)}] plagiarism scores from disk: {ans}. Dir: {location}")
        return ans

    def _refresh_plagiarism_scores_in_memory_and_disk(self):
        # Filters out miners that have already been deregistered. (Not in the metagraph)
        # This allows the miner to participate again once they re-register
        cached_miner_plagiarism = self.get_plagiarism_scores_from_disk()
        
        blocklist_dict = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_plagiarism_blocklist_file_location()
        )

        blocklist_scores = {
            key['miner_id']: 1 for key in blocklist_dict
        }

        self.miner_plagiarism_scores = {mch: mc for mch, mc in cached_miner_plagiarism.items() if
                                        mch in self.metagraph.hotkeys}
        
        self.miner_plagiarism_scores = {
            **self.miner_plagiarism_scores,
            **blocklist_scores
        }

        bt.logging.info(f"Loaded [{len(self.miner_plagiarism_scores)}] miner plagiarism scores from disk.")

        self._write_updated_plagiarism_scores_from_memory_to_disk()

    def _update_plagiarism_scores_in_memory(self):
        cached_miner_plagiarism = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_plagiarism_scores_file_location(running_unit_tests=self.running_unit_tests))
        self.miner_plagiarism_scores = {mch: mc for mch, mc in cached_miner_plagiarism.items()}


    def init_cache_files(self) -> None:
        ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_dir(running_unit_tests=self.running_unit_tests))

        if len(self.get_miner_eliminations_from_disk()) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests), {CacheController.ELIMINATIONS: []}
            )

        if len(ValiUtils.get_vali_json_file(ValiBkpUtils.get_plagiarism_scores_file_location(running_unit_tests=self.running_unit_tests))) == 0:
            hotkeys = self.metagraph.hotkeys if self.metagraph is not None else []
            miner_copying_file = {hotkey: 0 for hotkey in hotkeys}
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_plagiarism_scores_file_location(running_unit_tests=self.running_unit_tests), miner_copying_file
            )

        # Check if the get_miner_dir directory exists. If not, create it
        dir_path = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def get_miner_eliminations_from_disk(self) -> dict:
        return ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests), CacheController.ELIMINATIONS
        )

    def get_last_modified_time_miner_directory(self, directory_path):
        try:
            # Get the last modification time of the directory
            timestamp = os.path.getmtime(directory_path)

            # Convert the timestamp to a datetime object
            last_modified_date = datetime.datetime.fromtimestamp(timestamp)

            return last_modified_date
        except FileNotFoundError:
            # Handle the case where the directory does not exist
            print(f"The directory {directory_path} was not found.")
            return None

    def append_elimination_row(self, hotkey, current_dd, mdd_failure):
        r = self.generate_elimination_row(hotkey, current_dd, mdd_failure)
        bt.logging.info(f"Created and appended elimination row: {r}")
        self.eliminations.append(r)

    def calculate_drawdown(self, final, initial):
        # Ex we went from return of 1 to 0.9. Drawdown is -10% or in this case -0.1. Return 1 - 0.1 = 0.9
        # Ex we went from return of 0.9 to 1. Drawdown is +10% or in this case 0.1. Return 1 + 0.1 = 1.1 (not really a drawdown)
        return 1.0 + ((float(final) - float(initial)) / float(initial))
    def is_drawdown_beyond_mdd(self, dd, time_now=None) -> str | bool:
        if time_now is None:
            time_now = TimeUtil.generate_start_timestamp(0)
        if (dd < ValiConfig.MAX_DAILY_DRAWDOWN and time_now.hour == 0 and time_now.minute < 5):
            return CacheController.MAX_DAILY_DRAWDOWN
        elif (dd < ValiConfig.MAX_TOTAL_DRAWDOWN):
            return CacheController.MAX_TOTAL_DRAWDOWN
        else:
            return False
