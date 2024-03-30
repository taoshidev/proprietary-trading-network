# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import shutil
import time
from collections import defaultdict
from pickle import UnpicklingError
from typing import List, Dict, Union, Set
import bittensor as bt
from pathlib import Path

from copy import deepcopy
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.position import Position
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import OrderStatus
from vali_objects.utils.position_utils import PositionUtils


class PositionManager(CacheController):
    def __init__(self, config=None, metagraph=None, running_unit_tests=False, load_retroactive_eliminations=False):
        self.position_locks = PositionLocks()
        super().__init__(config=config, metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.position_uuids_to_ignore = self.get_retroactive_eliminations_from_disk()


    def close_open_positions_for_miner(self, hotkey):
        for trade_pair in TradePair:
            trade_pair_id = trade_pair.trade_pair_id
            with self.position_locks.get_lock(hotkey, trade_pair_id):
                open_position = self.get_open_position_for_a_miner_trade_pair(hotkey, trade_pair_id)
                if open_position:
                    bt.logging.info(f"Closing open position for hotkey: {hotkey} and trade_pair: {trade_pair_id}")
                    open_position.close_out_position(TimeUtil.now_in_millis())
                    self.save_miner_position_to_disk(open_position)

    def recalculate_return_at_close(self, position: Position):
        # Recalculate return_at_close for the position. Note, we are not modifying the input position
        disposable_clone = deepcopy(position)
        disposable_clone.position_type = None
        disposable_clone._update_position()
        #bt.logging.info(f"Recalculated return_at_close for position {position.position_uuid}. New value: {disposable_clone.return_at_close}. Original value: {position.return_at_close}")
        return disposable_clone.return_at_close


    def get_all_disk_positions_for_all_miners(self, **a):
        all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(
            ValiBkpUtils.get_miner_dir()
        )
        return self.get_all_miner_positions_by_hotkey(all_miner_hotkeys, only_open_positions=False, sort_positions=sort_positions)

    def get_retroactive_eliminations_from_disk(self) -> Set[str]:
        # Scan all closed positions on disk and recalculate return_at_close.
        # If the drawdown ever exceeds the threshold, add the position_uuid to the return set
        ret = set()
        all_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)
        hotkey_to_ignores = defaultdict(set)
        eliminations = self.get_miner_eliminations_from_disk()
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
        bt.logging.info(f"Found {len(eliminations)} eliminations on disk.")

        for hotkey, positions in all_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            max_portfolio_return = 1.0
            cur_portfolio_return = 1.0
            most_recent_elimination_idx = None

            if len(positions) == 0:
                continue

            prev_t = positions[0].close_ms
            for i, position in enumerate(positions):
                if position.is_open_position:
                    assert all(p.is_open_position for p in positions[i:])
                    break
                assert position.close_ms >= prev_t, f"Positions must be sorted by close time for this calculation to work."
                cur_portfolio_return *= self.recalculate_return_at_close(position)
                max_portfolio_return = max(max_portfolio_return, cur_portfolio_return)
                drawdown = self.calculate_drawdown(cur_portfolio_return, max_portfolio_return)
                mdd_failure = self.is_drawdown_beyond_mdd(drawdown,
                                                           time_now=TimeUtil.millis_to_datetime(position.close_ms))
                if mdd_failure:
                    most_recent_elimination_idx = i
                prev_t = position.close_ms
            if most_recent_elimination_idx is not None:
                for position in positions[:most_recent_elimination_idx + 1]:
                    ret.add(position.position_uuid)
                    hotkey_to_ignores[hotkey].add(position.position_uuid)
        if ret:
            bt.logging.info(f"Found {len(ret)} positions to ignore from retroactive eliminations across {len(hotkey_to_ignores)} hotkeys.")
        for hotkey, position_uuids in hotkey_to_ignores.items():
            bt.logging.trace(f"hotkey: {hotkey}. n_positions_to_ignore: {len(position_uuids)}")
        #self.log_remaining_miners_debug(all_positions, eliminated_hotkeys, ret)
        return ret

    def log_remaining_miners_debug(self, all_positions, eliminated_hotkeys, ret):
        # Figure out how many miners have >= 10 positions, excluding eliminations and retroactive eliminations
        hotkey_to_n_positions = defaultdict(int)
        for hotkey, positions in all_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            for position in positions:
                if position.position_uuid in ret:
                    continue
                hotkey_to_n_positions[hotkey] += 1
        j = 1
        for hotkey, n_positions in hotkey_to_n_positions.items():
            if n_positions >= 10:
                bt.logging.info(f"hotkey # {j} will still get weight: {hotkey}. n_positions: {n_positions}")
                j += 1


    def sort_by_close_ms(self, _position):
        return (
            _position.close_ms if _position.is_closed_position else float("inf")
        )

    def get_return_per_closed_position(self, positions: List[Position]) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []
        for position in positions:
            if position.is_open_position:
                continue
            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms
            closed_position_returns.append(position.return_at_close)

        cumulative_return = 1
        per_position_return = []

        # calculate the return over time at each position close
        for value in closed_position_returns:
            cumulative_return *= value
            per_position_return.append(cumulative_return)
        return per_position_return
    
    def get_return_per_closed_position_augmented(
            self, 
            positions: List[Position], 
            evaluation_time_ms: Union[None, int] = None
        ) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []
        for position in positions:
            if position.is_open_position:
                continue
            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms

            # this value will probably be around 0, or between -1 and 1
            logged_return_at_close = PositionUtils.log_transform(position.return_at_close)

            # this should be even closer to 0 with an absolute value less than the logged_return_at_close
            if evaluation_time_ms is None:
                raise ValueError("evaluation_time_ms must be provided if augmented is True")
            
            dampened_return_at_close = PositionUtils.dampen_return(
                logged_return_at_close, 
                position.open_ms, 
                position.close_ms, 
                evaluation_time_ms
            )
            closed_position_returns.append(dampened_return_at_close)

        # cumulative_return_logged = 0
        # per_position_return_logged = []

        # # calculate the return over time at each position close
        # for value in closed_position_returns:
        #     cumulative_return_logged += value
        #     per_position_return_logged.append(value)

        return [ PositionUtils.exp_transform(value) for value in closed_position_returns ]

    def get_all_miner_positions(self,
                                miner_hotkey: str,
                                only_open_positions: bool = False,
                                sort_positions: bool = False,
                                acceptable_position_end_ms: int = None,
                                filter_retroactive_eliminations: bool = False,
                                ) -> List[Position]:

        miner_dir = ValiBkpUtils.get_miner_all_positions_dir(miner_hotkey, running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)

        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions):
            bt.logging.trace(f"miner_dir: {miner_dir}, n_positions: {len(positions)}")

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

        if filter_retroactive_eliminations:
            positions = [position for position in positions if position.position_uuid not in self.position_uuids_to_ignore]

        return positions

    def get_all_miner_positions_by_hotkey(self, hotkeys: List[str], eliminations: List = None, **args) -> Dict[
        str, List[Position]]:
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()
        return {
            hotkey: self.get_all_miner_positions(hotkey, **args)
            for hotkey in hotkeys
            if hotkey not in eliminated_hotkeys
        }

    @staticmethod
    def positions_are_the_same(position1: Position, position2: Position | dict) -> (bool, str):
        # Iterate through all the attributes of position1 and compare them to position2.
        # Get attributes programmatically.
        comparing_to_dict = isinstance(position2, dict)
        for attr in dir(position1):
            attr_is_property = isinstance(getattr(type(position1), attr, None), property)
            if attr.startswith("_") or callable(getattr(position1, attr)) or (comparing_to_dict and attr_is_property):
                continue

            value1 = getattr(position1, attr)
            # Check if position2 is a dict and access the value accordingly.
            if comparing_to_dict:
                # Use .get() to avoid KeyError if the attribute is missing in the dictionary.
                value2 = position2.get(attr)
            else:
                value2 = getattr(position2, attr, None)

            if value1 != value2:
                return False, f"{attr} is different. {value1} != {value2}"
        return True, ""

    def get_miner_position_from_disk_using_position_in_memory(self, memory_position: Position) -> Position:
        # Position could have changed to closed
        fp = self.get_filepath_for_position(hotkey=memory_position.miner_hotkey,
                                            trade_pair_id=memory_position.trade_pair.trade_pair_id,
                                            position_uuid=memory_position.position_uuid,
                                            is_open=memory_position.is_open_position)
        try:
            return self.get_miner_position_from_disk(fp)
        except Exception as e:
            bt.logging.info(f"Error getting position from disk: {e}")
            if memory_position.is_open_position:
                bt.logging.warning(f"Attempting to get closed position from disk for memory position {memory_position}")
                fp = self.get_filepath_for_position(hotkey=memory_position.miner_hotkey,
                                                    trade_pair_id=memory_position.trade_pair.trade_pair_id,
                                                    position_uuid=memory_position.position_uuid,
                                                    is_open=False
                                                )
                return self.get_miner_position_from_disk(fp)
            else:
                raise e

    def get_miner_position_from_disk(self, file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        try:
            file_string = ValiBkpUtils.get_file(file)
            ans = Position.parse_raw(file_string)
            #bt.logging.info(f"vali_utils get_miner_position: {ans}")
            return ans
        except FileNotFoundError:
            raise ValiFileMissingException("Vali position file is missing")
        except UnpicklingError:
            raise ValiBkpCorruptDataException("position data is not pickled")
        except UnicodeDecodeError as e:
            raise ValiBkpCorruptDataException(f" Error {e} You may be running an old version of the software. Confirm with the team if you should delete your cache.")

    def get_recently_updated_miner_hotkeys(self):
        # Define the path to the directory containing the directories to check
        query_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        # Get the current time
        current_time = time.time()
        # List of directories updated in the last 24 hours
        updated_directory_names = []
        # Loop through each item in the specified folder
        for item in os.listdir(query_dir):
            item_path = Path(query_dir) / item  # Construct the full path
            if item_path.is_dir():  # Check if the item is a directory
                # Get the last modification time of the directory
                root_last_modified_time = self._get_file_mod_time(item_path)
                latest_modification_time = self._get_latest_file_modification_time(item_path, root_last_modified_time)
                # Check if the directory was updated in the last 24 hours
                if current_time - latest_modification_time < 259200:  # 3 days in seconds
                    updated_directory_names.append(item)

        return updated_directory_names

    def _get_latest_file_modification_time(self, dir_path, root_last_modified_time):
        """
        Recursively finds the max modification time of all files within a directory.
        """
        latest_mod_time = root_last_modified_time
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                mod_time = self._get_file_mod_time(file_path)
                latest_mod_time = max(latest_mod_time, mod_time)

        return latest_mod_time

    def _get_file_mod_time(self, file_path):
        try:
            return os.path.getmtime(file_path)
        except OSError:  # Handle the case where the file is inaccessible
            return 0

    def delete_open_position_if_exists(self, position: Position) -> None:
        # See if we need to delete the open position file
        open_position = self.get_open_position_for_a_miner_trade_pair(position.miner_hotkey,
                                                                      position.trade_pair.trade_pair_id)
        if open_position:
            self.delete_position_from_disk(open_position.miner_hotkey, open_position.trade_pair.trade_pair_id,
                                           open_position.position_uuid, True)

    def verify_open_position_write(self, miner_dir, updated_position):
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
        # Print all files found for dir
        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions) == 0:
            return  # First time open position is being saved
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {updated_position.miner_hotkey} and trade_pair."
                                                   f" {updated_position.trade_pair.trade_pair_id}. Please restore cache. Positions: {positions}")
        elif len(positions) == 1:
            if positions[0].position_uuid != updated_position.position_uuid:
                raise ValiRecordsMisalignmentException(f"Open position for miner {updated_position.miner_hotkey} and trade_pair."
                                                       f" {updated_position.trade_pair.trade_pair_id} does not match the updated position."
                                                       f" Please restore cache. Position: {positions[0]} Updated position: {updated_position}")

    def save_miner_position_to_disk(self, position: Position) -> None:
        miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                     position.trade_pair.trade_pair_id,
                                                                     order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                                                                     running_unit_tests=self.running_unit_tests)
        if position.is_closed_position:
            self.delete_open_position_if_exists(position)
        else:
            self.verify_open_position_write(miner_dir, position)

        ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)

    def clear_all_miner_positions_from_disk(self):
        # Clear all files and directories in the directory specified by dir
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def get_number_of_eliminations(self):
        return len(self.get_miner_eliminations_from_disk())

    def get_number_of_plagiarism_scores(self):
        return len(self.get_plagiarism_scores_from_disk())

    def get_number_of_miners_with_any_positions(self):
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        ret = 0
        try:
            for file in os.listdir(dir):
                file_path = os.path.join(dir, file)
                ret += os.path.isdir(file_path)
            bt.logging.info(f"Number of miners with any positions: {ret}. Positions dir: {dir}")
        except FileNotFoundError:
            bt.logging.info(f"Directory for miners doesn't exist [{dir}].")
        return ret

    def get_extreme_position_order_processed_on_disk_ms(self):
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        min_time = float("inf")
        max_time = 0
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                continue
            hotkey = file
            # Read all positions in this directory
            positions = self.get_all_miner_positions(hotkey)
            for p in positions:
                for o in p.orders:
                    min_time = min(min_time, o.processed_ms)
                    max_time = max(max_time, o.processed_ms)
        return min_time, max_time



    def get_open_position_for_a_miner_trade_pair(self, hotkey: str, trade_pair_id: str) -> Position | None:
        dir = ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=OrderStatus.OPEN,
                                                               running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(dir)
        # Print all files found for dir
        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {hotkey} and trade_pair."
                                                   f" {trade_pair_id}. Please restore cache. Positions: {positions}")
        return positions[0] if len(positions) == 1 else None

    def get_filepath_for_position(self, hotkey, trade_pair_id, position_uuid, is_open):
        order_status = OrderStatus.CLOSED if not is_open else OrderStatus.OPEN
        return ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=order_status,
                                                               running_unit_tests=self.running_unit_tests) + position_uuid

    def delete_position_from_disk(self, hotkey, trade_pair_id, position_uuid, is_open):
        filepath = self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, is_open)
        os.remove(filepath)
        bt.logging.info(f"Deleted position from disk: {filepath}")

