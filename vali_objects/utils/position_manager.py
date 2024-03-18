# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import shutil
from pickle import UnpicklingError
from typing import List, Dict, Union
import bittensor as bt
import numpy as np

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
    def __init__(self, config=None, metagraph=None, running_unit_tests=False):
        self.position_locks = PositionLocks()
        super().__init__(config=config, metagraph=metagraph, running_unit_tests=running_unit_tests)

    def close_open_positions_for_miner(self, hotkey):
        for trade_pair in TradePair:
            trade_pair_id = trade_pair.trade_pair_id
            with self.position_locks.get_lock(hotkey, trade_pair_id):
                open_position = self.get_open_position_for_a_miner_trade_pair(hotkey, trade_pair_id)
                if open_position:
                    bt.logging.info(f"Closing open position for hotkey: {hotkey} and trade_pair: {trade_pair_id}")
                    open_position.close_out_position(TimeUtil.now_in_millis())
                    self.save_miner_position_to_disk(open_position)


    def get_return_per_closed_position(self, positions: List[Position]) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []
        for position in positions:
            if not position.is_closed_position:
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
            if not position.is_closed_position:
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

        cumulative_return_logged = 0
        per_position_return_logged = []

        # calculate the return over time at each position close
        for value in closed_position_returns:
            cumulative_return_logged += value
            per_position_return_logged.append(cumulative_return_logged)

        return [ PositionUtils.exp_transform(value) for value in per_position_return_logged ]

    def get_all_miner_positions(self,
                                miner_hotkey: str,
                                only_open_positions: bool = False,
                                sort_positions: bool = False,
                                acceptable_position_end_ms: int = None
                                ) -> List[Position]:
        def _sort_by_close_ms(_position):
            # Treat None values as largest possible value
            return (
                _position.close_ms if _position.close_ms is not None else float("inf")
            )

        miner_dir = ValiBkpUtils.get_miner_all_positions_dir(miner_hotkey, running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)

        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions):
            bt.logging.info(f"miner_dir: {miner_dir}, n_positions: {len(positions)}")

        if acceptable_position_end_ms is not None:
            positions = [
                position
                for position in positions
                if position.open_ms > acceptable_position_end_ms
            ]

        if only_open_positions:
            positions = [
                position for position in positions if position.close_ms is None
            ]

        if sort_positions:
            positions = sorted(positions, key=_sort_by_close_ms)

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
        for attr in dir(position1):
            if not attr.startswith("_") and not callable(getattr(position1, attr)):
                value1 = getattr(position1, attr)

                # Check if position2 is a dict and access the value accordingly.
                if isinstance(position2, dict):
                    # Use .get() to avoid KeyError if the attribute is missing in the dictionary.
                    value2 = position2.get(attr)
                else:
                    value2 = getattr(position2, attr, None)

                if value1 != value2:
                    return False, f"{attr} is different. {value1} != {value2}"
        return True, ""

    def get_miner_position_from_disk(self, file) -> str | List[Position]:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        try:
            ans = ValiBkpUtils.get_file(file, True)
            #bt.logging.info(f"vali_utils get_miner_position: {ans}")
            return ans
        except FileNotFoundError:
            raise ValiFileMissingException("Vali position file is missing")
        except UnpicklingError:
            raise ValiBkpCorruptDataException("position data is not pickled")

    def delete_open_position_if_exists(self, position: Position) -> None:
        # See if we need to delete the open position file
        open_position = self.get_open_position_for_a_miner_trade_pair(position.miner_hotkey,
                                                                      position.trade_pair.trade_pair_id)
        if open_position:
            self.delete_position_from_disk(open_position.miner_hotkey, open_position.trade_pair.trade_pair_id,
                                           open_position.position_uuid, True)
    def save_miner_position_to_disk(self, position: Position) -> None:
        if position.is_closed_position:
            order_status = OrderStatus.CLOSED
            self.delete_open_position_if_exists(position)
        else:
            order_status = OrderStatus.OPEN
        miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                     position.trade_pair.trade_pair_id,
                                                                     order_status=order_status,
                                                                     running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(miner_dir + position.position_uuid, position, True)

    def clear_all_miner_positions_from_disk(self):
        # Clear all files and directories in the directory specified by dir
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def get_open_position_for_a_miner_trade_pair(self, hotkey: str, trade_pair_id: str) -> Position | None:
        dir = ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=OrderStatus.OPEN,
                                                               running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(dir)
        # Print all files found for dir
        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {hotkey} and trade_pair {trade_pair_id}")
        return positions[0] if len(positions) == 1 else None

    def get_filepath_for_position(self, hotkey, trade_pair_id, position_uuid, is_open):
        order_status = OrderStatus.CLOSED if not is_open else OrderStatus.OPEN
        return ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=order_status,
                                                               running_unit_tests=self.running_unit_tests) + position_uuid

    def delete_position_from_disk(self, hotkey, trade_pair_id, position_uuid, is_open):
        filepath = self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, is_open)
        os.remove(filepath)
        bt.logging.info(f"Deleted position from disk: {filepath}")

