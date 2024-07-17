# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import math
import numpy as np
import copy

from vali_objects.position import Position
from vali_config import ValiConfig

import bittensor as bt

from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order
from vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.scoring.historical_scoring import HistoricalScoring
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint
from time_util.time_util import TimeUtil

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
    def translate_current_leverage(
        positions: list[Position],
        evaluation_time_ms: int = None
    ) -> list[Position]:
        """
        Adjusts the leverage of each position based on order types and adds a new order with the final leverage at the end.
        """
        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        positions_copy = copy.deepcopy(positions)
        for position in positions_copy:
            running_leverage = 0
            new_orders = []
            for order in position.orders:
                running_leverage += order.leverage

                if order.order_type == OrderType.FLAT:
                    running_leverage = 0  # Reset leverage if order type is FLAT

                order.leverage = running_leverage

            # Create and append a new order with the final running leverage
            new_order = copy.deepcopy(position.orders[-1])
            new_order.processed_ms = evaluation_time_ms
            new_order.leverage = running_leverage
            if new_order.order_type != OrderType.FLAT:
                new_orders.append(new_order)

            position.orders.extend(new_orders)  # Append all new orders after the loop

        return positions_copy
    
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
    def compute_recent_drawdown(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
            evaluation_time_ms: int - the evaluation time
        """
        if len(checkpoints) <= 0:
            return 0

        drawdown_nterms = ValiConfig.DRAWDOWN_NTERMS

        ## Compute the drawdown of the checkpoints
        drawdowns = [ checkpoint.mdd for checkpoint in checkpoints ]

        recent_drawdown = min(drawdowns)
        recent_drawdown = np.clip(recent_drawdown, 0, 1.0)

        return recent_drawdown
    
    @staticmethod
    def consistency_sigmoid(delta_discrepancy: float) -> float:
        ## Convert a max delta term into a consistency penalty
        consistency_displacement = ValiConfig.SET_WEIGHT_MINER_CHECKPOINT_CONSISTENCY_DISPLACEMENT
        consistency_taper = ValiConfig.SET_WEIGHT_CHECKPOINT_CONSISTENCY_TAPER
        lower_bound = ValiConfig.SET_WEIGHT_CHECKPOINT_CONSISTENCY_LOWER_BOUND

        exp_term = np.clip(consistency_taper * delta_discrepancy - consistency_displacement, -50, 50)
        return ((1-lower_bound)*(1 + (np.exp(exp_term)))**-1) + lower_bound
    
    @staticmethod
    def mdd_lower_augmentation(recent_drawdown_percentage: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        drawdown_minvalue = ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE

        ## Protect against division by zero
        if drawdown_minvalue <= 0 or recent_drawdown_percentage <= 0:
            return 1

        ## Drawdown value
        if recent_drawdown_percentage <= drawdown_minvalue:
            return 0
        
        return 1
    
    @staticmethod
    def mdd_upper_augmentation(recent_drawdown_percentage: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        drawdown_maxvalue = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE
        drawdown_scaling = ValiConfig.DRAWDOWN_UPPER_SCALING

        if drawdown_maxvalue <= 0 or recent_drawdown_percentage <= 0:
            return 1
        
        upper_penalty = (-recent_drawdown_percentage + drawdown_maxvalue) / drawdown_scaling
        return float(np.clip(upper_penalty, 0, 1))
    
    @staticmethod
    def mdd_base_augmentation(recent_drawdown_percentage: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if recent_drawdown_percentage <= 0:
            return 1

        return float(1 / recent_drawdown_percentage)
    
    @staticmethod
    def mdd_augmentation(recent_drawdown: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if recent_drawdown <= 0 or recent_drawdown > 1:
            return 0
        
        recent_drawdown_percentage = (1 - recent_drawdown) * 100
        if recent_drawdown_percentage <= ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE:
            return 0
        
        if recent_drawdown_percentage >= ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE:
            return 0

        base_augmentation = PositionUtils.mdd_base_augmentation(recent_drawdown_percentage)
        lower_augmentation = PositionUtils.mdd_lower_augmentation(recent_drawdown_percentage)
        upper_augmentation = PositionUtils.mdd_upper_augmentation(recent_drawdown_percentage)

        drawdown_penalty = base_augmentation * lower_augmentation * upper_augmentation
        return float(drawdown_penalty)
    
    def compute_drawdown_penalty_cps(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
        """
        if len(checkpoints) <= 0:
            return 0
        
        recent_drawdown = PositionUtils.compute_recent_drawdown(checkpoints)
        drawdown_penalty = PositionUtils.mdd_augmentation(recent_drawdown)
        return drawdown_penalty
    
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
        epsilon = ValiConfig.EPSILON

        checkpoint_length_augmentation = 1
        checkpoint_duration_augmentation = 1

        if len(checkpoints) < 1:
            return 0
        
        nonzero_checkpoints = [ checkpoint for checkpoint in checkpoints if checkpoint.open_ms > 0 ]
        if len(nonzero_checkpoints) <= 0:
            return 0
    
        checkpoint_margins = [ checkpoint.gain + checkpoint.loss for checkpoint in nonzero_checkpoints ]
        checkpoint_absolute = [ abs(x) for x in checkpoint_margins ]

        marginsum = max(sum(checkpoint_margins), epsilon)
        margins_consistency = max(checkpoint_absolute) / marginsum
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
    
    @staticmethod
    def flatten_positions(
        positions: dict[str, list[Position]]
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
        """
        positions_list = []
        for minerkey, minerpositions in positions.items():
            for position in minerpositions:
                positions_list.append(position)

        return positions_list
    
    @staticmethod
    def running_leverage_computation(
        positions: list[Position]
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
        """
        positions_copy = copy.deepcopy(positions)
        for position in positions_copy:
            for order in position.orders:
                order.leverage = np.clip(order.leverage, 0, 1)
        
        return positions
    
    @staticmethod
    def to_state_list(
        positions: list[Position],
        current_time: int,
        constrain_lookback: bool = True
    ) -> tuple:
        """
        Args:
            positions: list[Position] - the positions
            return: list[dict] - the order list
        """
        order_list = []

        miners = set()
        trade_pairs = set()

        if constrain_lookback:
            start_time = current_time - ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        else:
            start_time = 0

        for position in positions:
            order_start = 0
            order_end = 0
            order_leverage = 0
            order_tradepair = None
            order_minerid = position.miner_hotkey

            if len(position.orders) == 0:
                continue
            
            for order_number, order in enumerate(position.orders):
                if order_number == 0:
                    order_start = order.processed_ms
                    order_leverage = order.leverage
                    order_tradepair = order.trade_pair.trade_pair_id
                    order_orderid = order.order_uuid
                    continue

                order_end = order.processed_ms

                if order_start >= start_time:
                    miners.add(order_minerid)
                    trade_pairs.add(order_tradepair)
                    order_list.append({
                        "miner_id": order_minerid,
                        "trade_pair": order_tradepair,
                        "leverage": order_leverage,
                        "start": order_start,
                        "end": order_end,
                        "order_id": order_orderid
                    })

                order_start = order_end
                order_leverage = order.leverage
                order_orderid = order.order_uuid

        return (
            sorted(list(miners)), 
            sorted(list(trade_pairs)), 
            order_list
        )
