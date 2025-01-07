# developer: jbonilla
import os
import datetime

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from pathlib import Path


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
        self.init_cache_files()
        self.metagraph = metagraph  # Refreshes happen on validator
        self._last_update_time_ms = 0
        self.eliminations = []
        self.DD_V2_TIME = TimeUtil.millis_to_datetime(1715359820000 + 1000 * 60 * 60 * 2)  # 5/10/24 TODO: Update before mainnet release

    def get_last_update_time_ms(self):
        return self._last_update_time_ms

    def _hotkey_in_eliminations(self, hotkey):
        for x in self.eliminations:
            if x['hotkey'] == hotkey:
                return x
        return None

    def set_last_update_time(self, skip_message=False):
        # Log that the class has finished updating and the time it finished updating
        if not skip_message:
            bt.logging.success(f"Finished updating class {self.__class__.__name__}")
        self._last_update_time_ms = TimeUtil.now_in_millis()

    @staticmethod
    def get_directory_names(query_dir):
        """
        Returns a list of directory names contained in the specified directory.

        :param query_dir: Path to the directory to query.
        :return: List of directory names.
        """
        directory_names = [item for item in os.listdir(query_dir) if (Path(query_dir) / item).is_dir()]
        return directory_names

    # ----------------- Eliminations -----------------

    @staticmethod
    def generate_elimination_row(hotkey, dd, reason, t_ms=None, price_info=None, return_info=None):
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()
        ans = {'hotkey': hotkey, 'elimination_initiated_time_ms': t_ms, 'dd': dd, 'reason': reason}
        if price_info:
            ans['price_info'] = price_info
        if return_info:
            ans['return_info'] = return_info
        bt.logging.info(f"Created elimination row: {ans}")
        return ans

    def refresh_allowed(self, refresh_interval_ms):
        return TimeUtil.now_in_millis() - self.get_last_update_time_ms() > refresh_interval_ms

    def _write_eliminations_from_memory_to_disk(self):
        self.write_eliminations_to_disk(self.eliminations)

    def write_eliminations_to_disk(self, eliminations):
        vali_eliminations = {CacheController.ELIMINATIONS: eliminations}
        bt.logging.trace(f"Writing [{len(eliminations)}] eliminations from memory to disk: {vali_eliminations}")
        output_location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, vali_eliminations)

    def clear_eliminations_from_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests), {CacheController.ELIMINATIONS: []})

    def _clear_eliminations_in_memory_and_disk(self):
        self.eliminations = []
        self.clear_eliminations_from_disk()

    def _refresh_eliminations_in_memory_and_disk(self):
        self.eliminations = self.get_filtered_eliminations_from_disk()
        self._write_eliminations_from_memory_to_disk()

    def _refresh_eliminations_in_memory(self):
        self.eliminations = self.get_eliminations_from_disk()

    def get_eliminated_hotkeys(self):
        return set([x['hotkey'] for x in self.eliminations]) if self.eliminations else set()

    def get_filtered_eliminations_from_disk(self):
        # Filters out miners that have already been deregistered. (Not in the metagraph)
        # This allows the miner to participate again once they re-register
        cached_eliminations = self.get_eliminations_from_disk()
        updated_eliminations = [elimination for elimination in cached_eliminations if elimination['hotkey'] in self.metagraph.hotkeys]
        if len(updated_eliminations) != len(cached_eliminations):
            bt.logging.info(f"Filtered [{len(cached_eliminations) - len(updated_eliminations)}] / "
                            f"{len(cached_eliminations)} eliminations from disk due to not being in the metagraph")
        return updated_eliminations

    def is_zombie_hotkey(self, hotkey):
        if not isinstance(self.eliminations, list):
            return False

        if hotkey in self.metagraph.hotkeys:
            return False

        if any(x['hotkey'] == hotkey for x in self.eliminations):
            return False

        return True

    def get_eliminations_from_disk(self):
        location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
        bt.logging.trace(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
        return cached_eliminations

    def init_cache_files(self) -> None:
        ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_dir(running_unit_tests=self.running_unit_tests))
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

    def append_elimination_row(self, hotkey, current_dd, mdd_failure, t_ms=None, price_info=None, return_info=None):
        elimination_row = self.generate_elimination_row(hotkey, current_dd, mdd_failure, t_ms=t_ms,
                                                        price_info=price_info, return_info=return_info)
        self.eliminations.append(elimination_row)

    @staticmethod
    def calculate_drawdown(final, initial):
        # Ex we went from return of 1 to 0.9. Drawdown is -10% or in this case -0.1. Return 1 - 0.1 = 0.9
        # Ex we went from return of 0.9 to 1. Drawdown is +10% or in this case 0.1. Return 1 + 0.1 = 1.1 (not really a drawdown)
        return 1.0 + ((float(final) - float(initial)) / float(initial))

    def is_drawdown_beyond_mdd(self, dd, time_now=None) -> str | bool:
        # DD V2 5/10/24 - remove daily DD. Make anytime DD limit 95%.
        if time_now is None:
            time_now = TimeUtil.generate_start_timestamp(0)
        if time_now < self.DD_V2_TIME:
            if dd < ValiConfig.MAX_DAILY_DRAWDOWN and time_now.hour == 0 and time_now.minute < 5:
                return CacheController.MAX_DAILY_DRAWDOWN
            elif dd < ValiConfig.MAX_TOTAL_DRAWDOWN:
                return CacheController.MAX_TOTAL_DRAWDOWN
        else:
            if dd < ValiConfig.MAX_TOTAL_DRAWDOWN_V2:
                return CacheController.MAX_TOTAL_DRAWDOWN

        return False
