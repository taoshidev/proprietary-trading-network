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
