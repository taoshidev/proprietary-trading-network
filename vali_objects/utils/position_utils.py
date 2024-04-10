# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import numpy as np

from vali_objects.position import Position
from vali_config import ValiConfig

from vali_objects.scoring.historical_scoring import HistoricalScoring

class PositionUtils:    
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
        lookback_period = ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_MS
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
    def compute_consistency_penalty(
        positions: list[Position],
        evaluation_time_ms: int
    ) -> float:
        """
        Args:
            close_ms_list: list[int] - the list of close times for the positions
            evaluation_time_ms: int - the evaluation time
        """
        if len(positions) == 0:
            return 0
        
        window_size = ValiConfig.HISTORICAL_PENALTY_WINDOW
        stride = ValiConfig.HISTORICAL_PENALTY_STRIDE
        
        lookback_fractions = [
            PositionUtils.compute_lookback_fraction(
                position.open_ms,
                position.close_ms,
                evaluation_time_ms
            ) for position in positions
            if position.is_closed_position and position.max_leverage_seen() >= ValiConfig.MIN_LEVERAGE_CONSITENCY_PENALTY
        ]

        # Sort the lookback fractions in ascending order
        lookback_fractions = sorted(lookback_fractions)
        
        # Initialize variables
        total_windows = int((1 - window_size) / stride) + 1
        represented_windows = 0
        
        # Iterate over the sliding windows
        for i in range(total_windows):
            window_start = i * stride
            window_end = window_start + window_size
            
            # Check if any lookback fraction falls within the current window
            for fraction in lookback_fractions:
                if window_start <= fraction < window_end:
                    represented_windows += 1
                    break
        
        # Calculate the penalty score
        penalty_score = represented_windows / total_windows

        if penalty_score >= 0.6:
            return 1
        elif penalty_score >= 0.5:
            return 0.95
        elif penalty_score >= 0.4:
            return 0.9
        elif penalty_score >= 0.3:
            return 0.85
        elif penalty_score >= 0.2:
            return 0.8
        elif penalty_score >= 0.1:
            return 0.75
                
        return 0.75