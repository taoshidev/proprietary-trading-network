# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import math
import numpy as np

from vali_objects.position import Position
from vali_config import ValiConfig

import bittensor as bt

from vali_objects.scoring.historical_scoring import HistoricalScoring
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint

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
    def augment_benefit(
        coefficient_of_augmentation: float,
        lookback_fraction: float,
    ) -> float:
        """
        Args:
            coefficient_of_augmentation: float - the coefficient of augmentation
            lookback_fraction: float - the fraction of the lookback period since the position was closed.
        """
        coefficient_of_augmentation = np.clip(coefficient_of_augmentation, 0, 1)
        lookback_fraction = np.clip(lookback_fraction, 0, 1)

        resulting_augmentation = (coefficient_of_augmentation - 1) * lookback_fraction + 1
        return np.clip(resulting_augmentation, 0, 1)
    
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
    def compute_average_leverage(positions: list[Position]) -> float:
        """
        Computes the time-weighted average leverage of a list of positions.

        Args:
            positions: list[Position] - the list of positions

        Returns:
            float - the time-weighted average leverage
        """
        if not positions:
            return 0.0

        total_time = 0
        total_timeleverage = 0

        for position in positions:
            if len(position.orders) < 2:
                continue

            last_time = position.orders[0].processed_ms
            running_leverage = position.orders[0].leverage

            for i in range(1, len(position.orders)):
                current_time = position.orders[i].processed_ms
                time_delta = current_time - last_time
                total_time += time_delta
                total_timeleverage += time_delta * abs(running_leverage)
                last_time = current_time
                running_leverage += position.orders[i].leverage

        if total_time == 0:
            return 0.0

        return total_timeleverage / total_time
    
    @staticmethod
    def compute_total_position_duration(
        positions: list[Position]
    ) -> int:
        """
        Args:
            positions: list[Position] - the list of positions
        """
        time_deltas = []

        for position in positions:
            if position.is_closed_position:
                time_deltas.append( position.close_ms - position.open_ms )

        return sum(time_deltas)
    
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
    def dampen_value(
        return_value: float,
        lookback_fraction: float,
        time_intensity_coefficient: float = None
    ) -> float:
        """
        Args:
            return_value: float - the return of the miner
            lookback_fraction: float - the fraction of the lookback period since the position was closed.
        """
        return HistoricalScoring.historical_decay_return(
            return_value, 
            lookback_fraction,
            time_intensity_coefficient=time_intensity_coefficient
        )
    
    @staticmethod
    def consistency_sigmoid(delta_discrepancy: float) -> float:
        ## Convert a max delta term into a consistency penalty
        consistency_displacement = ValiConfig.SET_WEIGHT_MINER_CHECKPOINT_CONSISTENCY_DISPLACEMENT
        consistency_taper = ValiConfig.SET_WEIGHT_CHECKPOINT_CONSISTENCY_TAPER
        lower_bound = ValiConfig.SET_WEIGHT_CHECKPOINT_CONSISTENCY_LOWER_BOUND

        exp_term = np.clip(consistency_taper * delta_discrepancy - consistency_displacement, -50, 50)
        return ((1-lower_bound)*(1 + (np.exp(exp_term)))**-1) + lower_bound
    
    ## just looking at the consistency penalties
    def compute_consistency_penalty_cps(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
            evaluation_time_ms: int - the evaluation time
        """

        # activity_threshold = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_TOTAL_ACTIVITY

        # checkpoint_consistency_threshold = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_CHECKPOINT_CONSISTENCY_THRESHOLD
        # checkpoint_consistency_ratio = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_CHECKPOINT_CONSISTENCY_RATIO

        if len(checkpoints) <= 0:
            return 0

        length_threshold = ValiConfig.CHECKPOINT_LENGTH_THRESHOLD
        duration_threshold = ValiConfig.CHECKPOINT_DURATION_THRESHOLD

        checkpoint_length_augmentation = 1
        checkpoint_duration_augmentation = 1

        nonzero_checkpoints = [ checkpoint for checkpoint in checkpoints if checkpoint.open_ms > 0 ]
        if len(nonzero_checkpoints) <= 0:
            return 0
    
        checkpoint_margins = [ abs(checkpoint.gain + checkpoint.loss) / checkpoint.open_ms for checkpoint in nonzero_checkpoints ]

        characteristic_behavior = np.median(checkpoint_margins)
        margins_volume = characteristic_behavior * len(checkpoint_margins)

        if margins_volume <= 0:
            return 0

        margins_consistency = max(checkpoint_margins) / margins_volume
        consistency_value = PositionUtils.consistency_sigmoid(margins_consistency)

        # ## Compute the duration of the checkpoints
        checkpoint_duration = sum([checkpoint.open_ms for checkpoint in nonzero_checkpoints])
        if checkpoint_duration < duration_threshold:
            checkpoint_duration_augmentation = (checkpoint_duration / duration_threshold)**2

        ## Check the length penalty of the checkpoints
        if len(nonzero_checkpoints) < length_threshold:
            checkpoint_length_augmentation = (len(nonzero_checkpoints) / length_threshold)**2

        return consistency_value * checkpoint_length_augmentation * checkpoint_duration_augmentation
    
    @staticmethod
    def compute_consistency_penalty(
        positions: list[Position],
        evaluation_time_ms: int
    ) -> float:
        """
        Args:
            positions: list[Position] - the list of positions
            evaluation_time_ms: int - the evaluation time
        """
        if len(positions) == 0:
            return 0
        
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
        consistency_penalties = PositionUtils.compute_consistency(lookback_fractions)
        return consistency_penalties
    
    @staticmethod
    def compute_consistency_penalty_positions(
        positions: list[Position],
        evaluation_time_ms: int
    ) -> float:
        """
        Args:
            positions: list[Position] - the list of positions
            evaluation_time_ms: int - the evaluation time
        """
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
        consistency_penalties = PositionUtils.compute_consistency(lookback_fractions)
        return consistency_penalties
    
    @staticmethod
    def compute_consistency(
        lookback_fractions: list[Position]
    ) -> float:
        """
        Args:
            close_ms_list: list[int] - the list of close times for the positions
        """
        if len(lookback_fractions) == 0:
            return 0
        
        window_size = ValiConfig.HISTORICAL_PENALTY_WINDOW
        stride = ValiConfig.HISTORICAL_PENALTY_STRIDE
        
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
            return 0.9
        elif penalty_score >= 0.4:
            return 0.8
        elif penalty_score >= 0.3:
            return 0.5
        elif penalty_score >= 0.2:
            return 0.25
        elif penalty_score >= 0.1:
            return 0.1
                
        return 0.1