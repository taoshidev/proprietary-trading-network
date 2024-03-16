# developer: jbonilla
# Copyright © 2023 Taoshi Inc
from typing import List, Dict
import bittensor as bt
import numpy as np

from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig

from vali_objects.scoring.historical_scoring import HistoricalScoring

class PositionUtils:
    @staticmethod
    def get_return_per_closed_position(
        positions: List[Position],
        evaluation_time_ms: int,
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
    
    @staticmethod
    def log_transform(
        return_value: float,
    ) -> float:
        """
        Args:
            return_value: float - the return of the miner
        """
        return_value = np.clip(return_value, 1e-12, None)
        return np.log(return_value)
    
    @staticmethod
    def exp_transform(
        return_value: float,
    ) -> float:
        """
        Args:
            return_value: float - the return of the miner
        """
        return np.exp(return_value)
    
    @staticmethod
    def compute_lookback_fraction(
        position_open_ms: int, 
        position_close_ms: int, 
        evaluation_time_ms: int
    ) -> float:
        lookback_period = ValiConfig.SET_WEIGHT_REFRESH_TIME_MS
        time_since_closed = evaluation_time_ms - position_close_ms
        time_fraction = time_since_closed / lookback_period
        time_fraction = np.clip(time_fraction, 0, 1)
        return time_fraction
    
    @staticmethod
    def dampen_return(
        return_value: float, 
        position_open_ms: int, 
        position_close_ms: int,
        evaluation_time_ms: int
    ) -> float:
        """
        Args:
            return_value: float - the return of the miner
            position_open_ms: int - the open time of the position
            position_close_ms: int - the close time of the position
            dampening_factor: float - the dampening factor
        """
        lookback_fraction = PositionUtils.compute_lookback_fraction(
            position_open_ms,
            position_close_ms,
            evaluation_time_ms
        )

        return HistoricalScoring.historical_decay_return(return_value, lookback_fraction)

    @staticmethod
    def get_all_miner_positions(
        miner_hotkey: str,
        only_open_positions: bool = False,
        sort_positions: bool = False,
        acceptable_position_end_ms: int = None,
    ) -> List[Position]:
        def _sort_by_close_ms(_position):
            # Treat None values as largest possible value
            return (
                _position.close_ms if _position.close_ms is not None else float("inf")
            )

        miner_dir = ValiBkpUtils.get_miner_position_dir(miner_hotkey)
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)

        positions = [ValiUtils.get_miner_positions_from_disk(file) for file in all_files]
        # log miner_dir, files, and positions
        #bt.logging.info(f"miner_dir: {miner_dir}, all_files: {all_files}, n_positions: {len(positions)}")

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

    @staticmethod
    def get_all_miner_positions_by_hotkey(
        hotkeys: List[str], eliminations: List = None, **args
    ) -> Dict[str, List[Position]]:
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()
        bt.logging.info(f"eliminated hotkeys: {eliminated_hotkeys}")
        return {
            hotkey: PositionUtils.get_all_miner_positions(hotkey, **args)
            for hotkey in hotkeys
            if hotkey not in eliminated_hotkeys
        }

   

