# developer: jbonilla
import os
import shutil
import datetime
from collections import defaultdict
from pickle import UnpicklingError
from typing import List, Dict

from shared_objects.retry import retry
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.position import Position
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
        self.metagraph = metagraph  # Refreshes happen on validator
        self._last_update_time_ms = 0
        self.last_time_plagiarism_run_ms = 0
        self.eliminations = []
        self.challengeperiod_testing = {}
        self.challengeperiod_success = {}
        self.plagiarism_data = {}
        self.plagiarism_raster = {}
        self.plagiarism_positions = {}
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

    # ----------------- Perf Ledgers -----------------
    def write_perf_ledger_eliminations_to_disk(self, eliminations):
        bt.logging.trace(f"Writing [{len(eliminations)}] eliminations from memory to disk: {eliminations}")
        output_location = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, eliminations)

    def get_perf_ledger_eliminations_from_disk(self):
        location = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=self.running_unit_tests)
        cached_eliminations = ValiUtils.get_vali_json_file(location)
        bt.logging.trace(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
        return cached_eliminations

    # ----------------- Plagiarism -----------------
    def clear_plagiarism_from_disk(self, target_hotkey=None):
        # Clear all files and directories in the directory specified by dir
        dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            if target_hotkey and file != target_hotkey:
                continue
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def plagiarism_detection_allowed(self, plagiarism_interval_ms):
        return TimeUtil.now_in_millis() - self.last_time_plagiarism_run_ms > plagiarism_interval_ms
    
    def write_plagiarism_scores_to_disk(self):
        for plagiarist in self.plagiarism_data:
            self.write_plagiarism_score_to_disk(plagiarist["plagiarist"], plagiarist)

    def write_plagiarism_score_to_disk(self, hotkey, plagiarism_data):
        ValiBkpUtils.write_file(ValiBkpUtils.get_plagiarism_score_file_location(hotkey=hotkey, running_unit_tests=self.running_unit_tests), plagiarism_data)

    def write_plagiarism_raster_to_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_plagiarism_raster_file_location(running_unit_tests=self.running_unit_tests), self.plagiarism_raster)

    def write_plagiarism_positions_to_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_plagiarism_positions_file_location(running_unit_tests=self.running_unit_tests), self.plagiarism_positions)

    def get_plagiarism_scores_from_disk(self):

        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir()
        all_files = ValiBkpUtils.get_all_files_in_dir(plagiarist_dir)

        # Retrieve hotkeys from plagiarism file names
        all_hotkeys = ValiBkpUtils.get_hotkeys_from_file_name(all_files)
        
        plagiarism_data = {hotkey: self.get_miner_plagiarism_data_from_disk(hotkey) for hotkey in all_hotkeys}
        plagiarism_scores = {}
        for hotkey in plagiarism_data:
            plagiarism_scores[hotkey] = plagiarism_data[hotkey].get("overall_score", 0)

        bt.logging.trace(f"Loaded [{len(plagiarism_scores)}] plagiarism scores from disk. Dir: {plagiarist_dir}")
        return plagiarism_scores
    
    def get_plagiarism_data_from_disk(self, ):
        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir()
        all_files = ValiBkpUtils.get_all_files_in_dir(plagiarist_dir)

        # Retrieve hotkeys from plagiarism file names
        all_hotkeys = ValiBkpUtils.get_hotkeys_from_file_name(all_files)
        
        plagiarism_data = {hotkey: self.get_miner_plagiarism_data_from_disk(hotkey) for hotkey in all_hotkeys}

        bt.logging.trace(f"Loaded [{len(plagiarism_data)}] plagiarism scores from disk. Dir: {plagiarist_dir}")
        return plagiarism_data

    def get_miner_plagiarism_data_from_disk(self, hotkey):
        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir()
        file_path = os.path.join(plagiarist_dir, f"{hotkey}.json")
        
        if os.path.exists(file_path):
            data = ValiUtils.get_vali_json_file(file_path)
            return data
        else:
            return {}

    """
    def _refresh_plagiarism_scores_in_memory_and_disk(self):
        # Filters out miners that have already been deregistered. (Not in the metagraph)
        # This allows the miner to participate again once they re-register
        cached_miner_plagiarism = self.get_plagiarism_scores_from_disk()

        blocklist_dict = ValiUtils.get_vali_json_file(ValiBkpUtils.get_plagiarism_blocklist_file_location())
        blocklist_scores = {key['miner_id']: 1 for key in blocklist_dict}

        self.miner_plagiarism_scores = {mch: mc for mch, mc in cached_miner_plagiarism.items() if mch in self.metagraph.hotkeys}

        self.miner_plagiarism_scores = {
            **self.miner_plagiarism_scores,
            **blocklist_scores
        }

        bt.logging.trace(f"Loaded [{len(self.miner_plagiarism_scores)}] miner plagiarism scores from disk.")

        self._write_updated_plagiarism_scores_from_memory_to_disk()
    """
    def _update_plagiarism_scores_in_memory(self):
        raster_positions_location = ValiBkpUtils.get_plagiarism_raster_file_location(running_unit_tests=self.running_unit_tests)
        self.plagiarism_raster = ValiUtils.get_vali_json_file(raster_positions_location)

        positions_location = ValiBkpUtils.get_plagiarism_positions_file_location(running_unit_tests=self.running_unit_tests)
        self.plagiarism_positions = ValiUtils.get_vali_json_file(positions_location)

        self.plagiarism_data = self.get_plagiarism_data_from_disk()

    # ----------------- Challenge Period -----------------
    def get_challengeperiod_testing(self):
        return ValiUtils.get_vali_json_file_dict(
            ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
        ).get('testing', {})

    def get_challengeperiod_success(self):
        return ValiUtils.get_vali_json_file_dict(
            ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
        ).get('success', {})

    def _refresh_challengeperiod_in_memory(self, eliminations: list[dict] = None):
        if eliminations is None:
            eliminations = self.eliminations

        eliminations_hotkeys = set([x['hotkey'] for x in eliminations])

        location = ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
        existing_challengeperiod = ValiUtils.get_vali_json_file_dict(location)
        existing_challengeperiod_testing = existing_challengeperiod.get('testing', {})
        existing_challengeperiod_success = existing_challengeperiod.get('success', {})

        self.challengeperiod_testing = {k: v for k, v in existing_challengeperiod_testing.items() if k not in eliminations_hotkeys}
        self.challengeperiod_success = {k: v for k, v in existing_challengeperiod_success.items() if k not in eliminations_hotkeys}

    def _refresh_challengeperiod_in_memory_and_disk(self, eliminations=None):
        if eliminations is None:
            eliminations = []

        self._refresh_challengeperiod_in_memory(eliminations=eliminations)
        self._write_challengeperiod_from_memory_to_disk()

    def clear_challengeperiod_from_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_challengeperiod_file_location(
            running_unit_tests=self.running_unit_tests),
            {"testing": {}, "success": {}}
        )

    def _clear_challengeperiod_in_memory_and_disk(self):
        self.challengeperiod_testing = {}
        self.challengeperiod_success = {}
        self.clear_challengeperiod_from_disk()

    def _promote_challengeperiod_in_memory(self, hotkeys: list[str], current_time: int):
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting hotkeys {hotkeys} to challengeperiod success.")

        new_success = {hotkey: current_time for hotkey in hotkeys}
        self.challengeperiod_success = {
            **self.challengeperiod_success,
            **new_success
        }

        for hotkey in hotkeys:
            if hotkey in self.challengeperiod_testing:
                self.challengeperiod_testing.pop(hotkey)
            else:
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but promotion to success was attempted.")

    def _demote_challengeperiod_in_memory(self, hotkeys: list[str]):
        for hotkey in hotkeys:
            bt.logging.info(f"Removing hotkeys {hotkey} from challenge period.")
            if hotkey in self.challengeperiod_testing:
                self.challengeperiod_testing.pop(hotkey)
            else:
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but demotion to failure was attempted.")

        for hotkey in hotkeys:
            bt.logging.info(f"Eliminating hotkey {hotkey}.")

            # This will also add the hotkey to the in memory self.eliminations list
            self.append_elimination_row(hotkey, -1, 'FAILED_CHALLENGE_PERIOD')

    def _write_challengeperiod_from_memory_to_disk(self):
        challengeperiod_data = {
            "testing": self.challengeperiod_testing,
            "success": self.challengeperiod_success
        }
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_challengeperiod_file_location(
                running_unit_tests=self.running_unit_tests
            ),
            challengeperiod_data
        )

    def _add_challengeperiod_testing_in_memory_and_disk(
            self,
            new_hotkeys: list[str],
            eliminations: list[dict] = None,
            current_time: int = None
    ):
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        if eliminations is None:
            eliminations = self.eliminations

        elimination_hotkeys = [x['hotkey'] for x in eliminations]

        # check all hotkeys which have at least one position
        miners_with_positions = self.get_all_miner_hotkeys_with_at_least_one_position()

        for hotkey in new_hotkeys:
            if hotkey in miners_with_positions:
                if hotkey not in elimination_hotkeys:
                    if hotkey not in self.challengeperiod_testing:
                        if hotkey not in self.challengeperiod_success:
                            bt.logging.info(f"Adding hotkey {hotkey} to challengeperiod miners.")
                            self.challengeperiod_testing[hotkey] = current_time

        self._write_challengeperiod_from_memory_to_disk()

    def init_cache_files(self) -> None:
        ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_dir(running_unit_tests=self.running_unit_tests))

        if len(self.get_miner_eliminations_from_disk()) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                {CacheController.ELIMINATIONS: []}
            )

        plagiarism_dir = ValiBkpUtils.get_plagiarism_dir(running_unit_tests=self.running_unit_tests)
        if not os.path.exists(plagiarism_dir):
            ValiBkpUtils.make_dir(ValiBkpUtils.get_plagiarism_dir(running_unit_tests=self.running_unit_tests))
            ValiBkpUtils.make_dir(ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests))
        
        if len(self.get_challengeperiod_testing()) == 0 and len(self.get_challengeperiod_success()) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests),
                {"testing": {}, "success": {}}
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

    def append_elimination_row(self, hotkey, current_dd, mdd_failure, t_ms=None, price_info=None, return_info=None):
        elimination_row = self.generate_elimination_row(hotkey, current_dd, mdd_failure, t_ms=t_ms,
                                                        price_info=price_info, return_info=return_info)
        self.eliminations.append(elimination_row)

    def calculate_drawdown(self, final, initial):
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

    def get_all_disk_positions_for_all_miners(self, **args):
        all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(
            ValiBkpUtils.get_miner_dir()
        )
        return self.get_all_miner_positions_by_hotkey(all_miner_hotkeys, **args)

    def get_miner_position_from_disk(self, file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        file_string = None
        try:
            file_string = ValiBkpUtils.get_file(file)
            ans = Position.model_validate_json(file_string)
            if not ans.orders:
                bt.logging.warning(f"Anomalous position has no orders: {ans.to_dict()}. Ignoring.")
            return ans
        except FileNotFoundError:
            raise ValiFileMissingException(f"Vali position file is missing {file}")
        except UnpicklingError as e:
            raise ValiBkpCorruptDataException(f"file_string is {file_string}, {e}")
        except UnicodeDecodeError as e:
            raise ValiBkpCorruptDataException(
                f" Error {e} for file {file} You may be running an old version of the software. Confirm with the team if you should delete your cache. file string {file_string[:2000] if file_string else None}")
        except Exception as e:
            raise ValiBkpCorruptDataException(f"Error {e} file_path {file} file_string: {file_string}")

    def sort_by_close_ms(self, _position):
        return (
            _position.close_ms if _position.is_closed_position else float("inf")
        )

    def exorcise_positions(self, positions, all_files) -> List[Position]:
        """
        Disk positions can be left in a bad state for a variety of reasons. Let's clean them up here.
        If a dup is encountered, deleted both and let position syncing add the correct one back.
        """
        filtered_positions = []
        position_uuid_to_count = defaultdict(int)
        order_uuid_to_count = defaultdict(int)
        order_uuids_to_purge = set()
        for position in positions:
            position_uuid_to_count[position.position_uuid] += 1
            for order in position.orders:
                order_uuid_to_count[order.order_uuid] += 1
                if order_uuid_to_count[order.order_uuid] > 1:
                    order_uuids_to_purge.add(order.order_uuid)

        for file_name, position in zip(all_files, positions):
            if position_uuid_to_count[position.position_uuid] > 1:
                bt.logging.info(f"Exorcising position from disk due to duplicate position uuid: {file_name} {position}")
                os.remove(file_name)
                continue

            elif not position.orders:
                bt.logging.info(f"Exorcising position from disk due to no orders: {file_name} {position.to_dict()}")
                os.remove(file_name)
                continue

            new_orders = [x for x in position.orders if order_uuid_to_count[x.order_uuid] == 1]
            if len(new_orders) != len(position.orders):
                bt.logging.info(f"Exorcising position from disk due to order mismatch: {file_name} {position}")
                os.remove(file_name)
            else:
                filtered_positions.append(position)
        return filtered_positions

    def get_all_miner_positions(self,
                                miner_hotkey: str,
                                only_open_positions: bool = False,
                                sort_positions: bool = False,
                                acceptable_position_end_ms: int = None,
                                perform_exorcism: bool = False
                                ) -> List[Position]:

        miner_dir = ValiBkpUtils.get_miner_all_positions_dir(miner_hotkey, running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)

        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if perform_exorcism:
            positions = self.exorcise_positions(positions, all_files)

        if acceptable_position_end_ms is not None:
            positions = [
                position
                for position in positions
                if position.open_ms > acceptable_position_end_ms
            ]

        if only_open_positions:
            positions = [
                position for position in positions if position.is_open_position
            ]

        if sort_positions:
            positions = sorted(positions, key=self.sort_by_close_ms)

        return positions

    @retry(tries=5, delay=1, backoff=1)
    def get_all_miner_positions_by_hotkey(self, hotkeys: List[str], eliminations: List = None, **args) -> Dict[
        str, List[Position]]:
        """
        Retry due to a race condition where an open position is deleted and the file is not found.
        """
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()

        return {
            hotkey: self.get_all_miner_positions(hotkey, **args)
            for hotkey in hotkeys
            if hotkey not in eliminated_hotkeys
        }

    def get_all_miner_hotkeys_with_at_least_one_position(self) -> set[str]:
        all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests))
        positions_dict = self.get_all_miner_positions_by_hotkey(all_miner_hotkeys)

        miner_nonzero_positions = set()
        for k, v in positions_dict.items():
            if len(v) > 0:
                miner_nonzero_positions.add(k)

        return miner_nonzero_positions
