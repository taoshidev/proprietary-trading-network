import copy

import numpy as np
import bittensor as bt

from vali_objects.vali_config import ValiConfig

from vali_objects.position import Position, Order
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair
from vali_objects.utils.functional_utils import FunctionalUtils

class RiskProfiling:
    @staticmethod
    def monotonic_positions(position: Position) -> Position:
        """
        Extract orders with monotonically increasing leverage on losing positions.

        This function identifies orders that increase leverage while the position is losing,
        which may indicate risky trading behavior such as "doubling down" on losing trades.

        Args:
            position: Position object containing order history

        Returns:
            Position: A copy of the position with only the monotonically increasing leverage orders
                     on losing trades. Returns an empty position if no such orders are found.
        """
        # Copy attributes from the original position
        result_position: Position = copy.deepcopy(position)
        position_orders: list[Order] = [order for order in position.orders]

        # Copy orders from the original position for analysis
        position_order_subset = []

        if len(position_orders) < 2:
            # Return the new position with empty orders
            result_position.orders = []
            return result_position

        # Determine the direction - always look at the first order to avoid issues with closed positions
        position_type = position_orders[0].order_type
        is_long: bool = position_type == OrderType.LONG
        position_direction: int = 1 if is_long else -1
        final_active_order: int = len(position_orders) if not position.is_closed_position else len(position_orders) - 1

        # For SHORT positions, leverage is negative, so we need to take the absolute value for calculations
        # but preserve the sign for direction checks
        initial_leverage = position_orders[0].leverage
        max_leverage = initial_leverage
        aggregate_leverage = initial_leverage

        total_weighted_price = position_orders[0].price * initial_leverage  # need this to be "negative" for SHORT positions
        avg_in_price = total_weighted_price / aggregate_leverage  # this should always be positive though

        for i in range(1, final_active_order):
            current_order = position_orders[i]
            current_leverage_delta: float = current_order.leverage

            # Price movement is against the position if:
            # - For LONG: current price < avg_in_price
            # - For SHORT: current price > avg_in_price
            order_price_direction: bool = current_order.price - avg_in_price  # how is the price of the most recent order changing relative to the average entry price

            # If the price direction is going the same way as the position, it's winning. If they are opposite (less than zero), it's losing.
            losing_order: bool = order_price_direction * position_direction < 0

            # Update aggregate leverage with the current order's leverage
            new_aggregate_leverage = aggregate_leverage + current_leverage_delta

            # Leverage is increasing if the new aggregate leverage is greater than the previous aggregate
            leverage_increased_beyond_max = abs(new_aggregate_leverage) > abs(max_leverage)

            # Flag if position is losing and leverage increased
            if losing_order and leverage_increased_beyond_max:
                # Add the order to the subset
                order_copy = copy.deepcopy(current_order)
                order_copy.leverage = new_aggregate_leverage
                position_order_subset.append(order_copy)

                # Update max leverage for the next iteration
                max_leverage = new_aggregate_leverage

            # Update weighted price and average entry price for the next iteration
            total_weighted_price += current_order.price * current_leverage_delta  # this will actually be "negative" for SHORT positions

            if new_aggregate_leverage == 0:
                bt.logging.warning(f"Monotonic positions new aggregate leverage is zero for current order: {current_order}")
                new_aggregate_leverage = ValiConfig.EPSILON

            avg_in_price = total_weighted_price / new_aggregate_leverage  # don't want to use absolute value here, as the avg in price should always be positive
            aggregate_leverage = new_aggregate_leverage

        # Use the new position object
        result_position.orders = position_order_subset
        return result_position

    @staticmethod
    def risk_assessment_steps_utilization(position: Position) -> int:
        """
        Count the number of steps where leverage was increased on a losing position.

        This function identifies how many times a trader increased leverage while
        the position was already losing, which may indicate risky trading behavior.

        Args:
            position: Position object containing order history

        Returns:
            int: The number of orders where leverage was increased on a losing position.
                 Higher values indicate more aggressive risk-taking.
        """
        position_orders = position.orders
        position_flagged_orders = []

        if len(position_orders) < 2:
            return 0  # less than two steps will never flag the position

        # Determine if it's a LONG or SHORT position
        is_long = position_orders[0].order_type == OrderType.LONG
        position_direction = 1 if is_long else -1

        final_active_order = len(position_orders) if not position.is_closed_position else len(position_orders) - 1

        # Initialize with first order
        initial_leverage = position_orders[0].leverage
        aggregate_leverage = max(abs(initial_leverage), ValiConfig.RISK_PROFILING_STEPS_MIN_LEVERAGE)

        total_weighted_price = position_orders[0].price * aggregate_leverage
        avg_in_price = total_weighted_price / aggregate_leverage

        for i in range(1, final_active_order):
            current_order = position_orders[i]
            current_leverage = current_order.leverage

            # Check if position is losing
            price_delta = ((current_order.price - avg_in_price) / avg_in_price) * 100
            is_losing = (price_delta * position_direction < 0)

            # Check if leverage is being added in the same direction as the position
            is_adding_leverage = (current_leverage > 0 and is_long) or (current_leverage < 0 and not is_long)

            # Flag if position is losing and adding leverage in the same direction
            if is_losing and is_adding_leverage:
                position_flagged_orders.append(current_order)

            # Update aggregate leverage (add absolute value for calculations)
            aggregate_leverage += abs(current_leverage)

            # Update weighted average price
            total_weighted_price += current_order.price * abs(current_leverage)
            avg_in_price = total_weighted_price / aggregate_leverage

        return len(position_flagged_orders)

    @staticmethod
    def risk_assessment_steps_criteria(position: Position) -> bool:
        """
        Determine if a position exceeds the steps criteria threshold for risk.

        This function checks if the number of steps where leverage was increased
        on a losing position exceeds the configured threshold.

        Args:
            position: Position object containing order history

        Returns:
            bool: True if the position exceeds the steps criteria threshold, False otherwise.
        """
        utilization = RiskProfiling.risk_assessment_steps_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_STEPS_CRITERIA

    @staticmethod
    def risk_assessment_monotonic_utilization(position: Position) -> int:
        """
        Count the number of orders with monotonically increasing leverage on losing positions.

        This function measures how many orders are part of a monotonically increasing
        leverage pattern on a losing position, which may indicate risky trading behavior.

        Args:
            position: Position object containing order history

        Returns:
            int: The number of orders with monotonically increasing leverage on losing trades.
        """
        monotone_position = RiskProfiling.monotonic_positions(position)
        return len(monotone_position.orders)

    @staticmethod
    def risk_assessment_monotonic_criteria(position: Position) -> bool:
        """
        Determine if a position exceeds the monotonic criteria threshold for risk.

        This function checks if the number of orders with monotonically increasing leverage
        on a losing position exceeds the configured threshold.

        Args:
            position: Position object containing order history

        Returns:
            bool: True if the position exceeds the monotonic criteria threshold, False otherwise.
        """
        utilization = RiskProfiling.risk_assessment_monotonic_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA

    @staticmethod
    def risk_assessment_margin_utilization(position: Position) -> float:
        """
        Calculate the margin utilization ratio of a position.

        This function measures how much of the available margin range the position
        utilizes, which indicates the relative risk in terms of leverage usage.

        Args:
            position: Position object containing order history

        Returns:
            float: A value between 0 and 1 representing the margin utilization,
                  where higher values indicate higher utilization of available margin.
        """
        # First track the margin range
        position_trade_pair: TradePair = position.trade_pair
        position_trade_pair_min_leverage = position_trade_pair.min_leverage
        position_trade_pair_max_leverage = position_trade_pair.max_leverage
        assert position_trade_pair_min_leverage < position_trade_pair_max_leverage, "Min leverage must be less than max leverage for all trade pairs"

        # Now compute the cumulative leverages on the position since inception
        positional_delta_leverages = [x.leverage for x in position.orders]
        positional_aggregate_leverages = np.abs(np.cumsum(positional_delta_leverages))
        max_utilized_leverage = max(positional_aggregate_leverages)

        margin_utilization = (max_utilized_leverage - position_trade_pair_min_leverage) / (position_trade_pair_max_leverage - position_trade_pair_min_leverage)

        return margin_utilization

    @staticmethod
    def risk_assessment_margin_criteria(position: Position) -> bool:
        """
        Determine if a position exceeds the margin criteria threshold for risk.

        This function checks if the margin utilization ratio exceeds the configured threshold.

        Args:
            position: Position object containing order history

        Returns:
            bool: True if the position exceeds the margin criteria threshold, False otherwise.
        """
        utilization = RiskProfiling.risk_assessment_margin_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_MARGIN_CRITERIA

    @staticmethod
    def risk_assessment_leverage_advancement_utilization(position: Position) -> float:
        """
        Calculate the leverage advancement ratio for a position.

        This function measures how much the position's leverage has increased from
        its initial level, which indicates increasing risk-taking behavior.

        Args:
            position: Position object containing order history

        Returns:
            float: The ratio of maximum leverage to initial leverage.
                  A value of 1.0 means no increase, while higher values indicate
                  more aggressive leverage increases.
        """
        # Now compute the cumulative leverages on the position since inception
        positional_delta_leverages = [x.leverage for x in position.orders]
        if not positional_delta_leverages:
            return 0.0  # Default to no advancement for empty positions

        # if the position is closed, excluding the last closing order from analysis
        if position.is_closed_position and len(position.orders) > 1:
            positional_delta_leverages = positional_delta_leverages[:-1]

        positional_aggregate_leverages = np.abs(np.cumsum(positional_delta_leverages))
        max_utilized_leverage = max(positional_aggregate_leverages)
        min_utilized_leverage = max(min(positional_aggregate_leverages), ValiConfig.RISK_PROFILING_STEPS_MIN_LEVERAGE)
        leverage_advancement = max_utilized_leverage / min_utilized_leverage

        return leverage_advancement

    @staticmethod
    def risk_assessment_leverage_advancement_criteria(position: Position) -> bool:
        """
        Determine if a position exceeds the leverage advancement criteria threshold for risk.

        This function checks if the leverage advancement ratio exceeds the configured threshold.

        Args:
            position: Position object containing order history

        Returns:
            bool: True if the position exceeds the leverage advancement criteria threshold, False otherwise.
        """
        utilization = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE

    @staticmethod
    def risk_assessment_time_utilization(position: Position) -> float:
        """
        Calculate the time utilization metric for a position.

        This function measures how evenly spaced the orders are in time, which can
        help identify suspicious trading patterns like TWAP avoidance.

        Args:
            position: Position object containing order history

        Returns:
            float: The average normalized error of order time intervals.
                  Higher values indicate more uneven time spacing between orders.
        """

        position_orders = position.orders
        if len(position_orders) < 3:
            return 0.0    # less than 3 orders (ie 1 or 2) is not meaningful in TWAP detection

        # using only the orders up to (and including) the one that brings the position’s leverage to its maximum level
        # first finding the idx where the max leverage is reached, if there are multiple indices, record the first idx
        positional_leverage_deltas = [order.leverage for order in position_orders]
        positional_aggregate_leverages = np.abs(np.cumsum(positional_leverage_deltas))
        max_leverage_index = int(np.argmax(positional_aggregate_leverages))

        # then collecting orders
        position_order_subset = []
        for i in range(0, max_leverage_index + 1):    # including the order at last_max_idx
            order_copy = copy.deepcopy(position_orders[i])
            position_order_subset.append(order_copy)

        # make sure we have at least 3 meaningful orders for TWAP detection
        if len(position_order_subset) < 3:
            return 0.0

        # now we are ready to calculate order time intervals (time_deltas)
        # ensure all processed_ms values are positive before calculating time deltas
        order_times = [max(1, order.processed_ms) for order in position_order_subset]
        # sort the times to ensure they are monotonically increasing to avoid negative time deltas
        order_times.sort()
        time_deltas = np.diff(order_times)      # no need to convert ms, it will be normalized
        total_order_time = order_times[-1] - order_times[0]
        ideal_interval = total_order_time / (len(position_order_subset) - 1)    # there are at least 3 orders

        # to avoid 0 division error
        if ideal_interval == 0:
            return 0.0

        norm_errors = np.abs(time_deltas - ideal_interval) / ideal_interval
        avg_norm_error = float(np.mean(norm_errors))

        return avg_norm_error


    @staticmethod
    def risk_assessment_time_criteria(position: Position) -> bool:
        """
        Determine if a position exceeds the time criteria threshold for risk.

        This function checks if the time utilization metric exceeds the configured threshold.

        Args:
            position: Position object containing order history

        Returns:
            bool: True if the position exceeds the time criteria threshold, False otherwise.
        """
        utilization = RiskProfiling.risk_assessment_time_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_TIME_CRITERIA


    @staticmethod
    def risk_profile_single(position: Position) -> dict:
        """
        Create a comprehensive risk profile for a single position.

        This function evaluates a position against all risk criteria and produces
        a detailed report of the risk factors and overall risk assessment.

        Args:
            position: Position object containing order history

        Returns:
            dict: A dictionary containing the position's risk profile, including:
                - position_return: The percentage return of the position
                - relative_weighting_strength: The weighting strength based on return
                - overall_flag: Whether the position is flagged as risky (1=risky, 0=not risky)
                - Individual risk utilization and criteria results for all risk factors
        """
        # Apply the appropriate classifier based on the asset class
        steps_utilization = RiskProfiling.risk_assessment_steps_utilization(position)
        steps_criteria = int(RiskProfiling.risk_assessment_steps_criteria(position))

        monotonic_utilization = RiskProfiling.risk_assessment_monotonic_utilization(position)
        monotonic_criteria = int(RiskProfiling.risk_assessment_monotonic_criteria(position))

        margin_utilization = RiskProfiling.risk_assessment_margin_utilization(position)
        margin_criteria = int(RiskProfiling.risk_assessment_margin_criteria(position))

        leverage_advancement_utilization = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        leverage_advancement_criteria = int(RiskProfiling.risk_assessment_leverage_advancement_criteria(position))

        time_utilization = RiskProfiling.risk_assessment_time_utilization(position)
        time_criteria = int(RiskProfiling.risk_assessment_time_criteria(position))

        # Full flag
        overall_flag = int(RiskProfiling.risk_profile_full_criteria(position))

        return {
            "position_return": round((position.return_at_close-1) * 100, 4),
            "relative_weighting_strength": position.return_at_close**ValiConfig.RISK_PROFILING_SCOPING_MECHANIC,
            "overall_flag": overall_flag,
            "steps_utilization": steps_utilization,
            "steps_criteria": steps_criteria,
            "monotonic_utilization": monotonic_utilization,
            "monotonic_criteria": monotonic_criteria,
            "margin_utilization": margin_utilization,
            "margin_criteria": margin_criteria,
            "leverage_advancement_utilization": leverage_advancement_utilization,
            "leverage_advancement_criteria": leverage_advancement_criteria,
            "time_utilization": time_utilization,
            "time_criteria": time_criteria
        }

    @staticmethod
    def risk_profile_reporting(positions: list[Position]) -> dict:
        """
        Generate risk profiles for a list of positions.

        Args:
            positions: List of Position objects to evaluate

        Returns:
            dict: A dictionary mapping position UUIDs to their risk profiles
        """
        return {position.position_uuid: RiskProfiling.risk_profile_single(position) for position in positions}

    @staticmethod
    def risk_profile_full_criteria(position: Position) -> bool:
        """
        Determines the relative risk profile of various assets.

        Returns:
            bool: False meaning no risk, True meaning full risk.

        Risk factors are grouped into three categories:
        1. Step-based factors (steps_criteria or monotonic_criteria)
        2. Leverage-based factors (margin_criteria or leverage_advancement_criteria)
        3. Time-based factors (time_criteria)

        ALL three categories must be triggered for a position to be flagged as risky.
        """
        # Step-based risk factors
        steps_criteria = RiskProfiling.risk_assessment_steps_criteria(position)
        monotonic_criteria = RiskProfiling.risk_assessment_monotonic_criteria(position)
        step_risk_triggered = steps_criteria or monotonic_criteria

        # Leverage-based risk factors
        margin_criteria = RiskProfiling.risk_assessment_margin_criteria(position)
        leverage_advancement_criteria = RiskProfiling.risk_assessment_leverage_advancement_criteria(position)
        leverage_risk_triggered = margin_criteria or leverage_advancement_criteria

        # Time-based risk factors
        time_risk_triggered = RiskProfiling.risk_assessment_time_criteria(position)

        # All three risk categories must be triggered for a full risk flag
        return step_risk_triggered and leverage_risk_triggered and time_risk_triggered

    @staticmethod
    def risk_profile_score_list(miner_positions: list[Position]) -> float:
        """
        Calculate the overall risk score for a list of positions with improved numerical stability.

        Args:
            miner_positions: List of Position objects to evaluate

        Returns:
            float: A value between 0.0 and 1.0 representing the overall risk score
        """
        if len(miner_positions) == 0:
            return 0.0

        # Compute risk flags for all positions
        criteria_weight = np.array([int(RiskProfiling.risk_profile_full_criteria(position)) for position in miner_positions])

        # If no positions are flagged as risky, return 0.0
        if np.sum(criteria_weight) == 0:
            return 0.0

        # Get return values for all positions
        return_values = np.array([p.return_at_close for p in miner_positions])

        # Calculate return weights in a numerically stable way
        scoping_mechanic = ValiConfig.RISK_PROFILING_SCOPING_MECHANIC

        # Compute log(return_values) * scoping_mechanic to avoid numerical issues
        log_returns = np.log(np.maximum(return_values, 1e-10))  # Avoid log(0)
        log_weights = log_returns * scoping_mechanic

        # Normalize log weights to prevent underflow/overflow
        max_log_weight = np.max(log_weights)
        normalized_weights = np.exp(log_weights - max_log_weight)

        # Ensure weights are positive and sum to a non-zero value
        if np.sum(normalized_weights) < 1e-10:
            # If all weights are effectively zero, use uniform weights
            normalized_weights = np.ones_like(normalized_weights) / len(normalized_weights)

        # Compute weighted average of risk flags
        profile_score = np.average(criteria_weight, weights=normalized_weights)
        return profile_score

    @staticmethod
    def risk_profile_score(miner_positions: dict[str, list[Position]]) -> dict[str, float]:
        """
        Calculate risk scores for multiple miners.

        Args:
            miner_positions: Dictionary mapping miner hotkeys to lists of their positions

        Returns:
            dict: Dictionary mapping miner hotkeys to their risk scores (0.0-1.0)
        """
        miner_scores = {}

        for miner, positions in miner_positions.items():
            miner_scores[miner] = RiskProfiling.risk_profile_score_list(positions)

        return miner_scores

    @staticmethod
    def risk_profile_penalty(miner_positions: dict[str, list[Position]]) -> dict[str, float]:
        """
        Calculate risk penalty multipliers for multiple miners.

        This function computes penalty multipliers based on risk scores,
        which are used to scale miners' rewards. Miners with higher risk
        scores receive lower penalty multipliers, reducing their rewards.

        Args:
            miner_positions: Dictionary mapping miner hotkeys to lists of their positions

        Returns:
            dict: Dictionary mapping miner hotkeys to their penalty multipliers (0.0-1.0),
                 where 1.0 means no penalty and values close to 0.0 mean heavy penalties.
        """
        risk_profile_scores = RiskProfiling.risk_profile_score(miner_positions)
        risk_profile_penalty = {}

        for miner, score in risk_profile_scores.items():
            risk_profile_penalty[miner] = FunctionalUtils.sigmoid(
                score,
                ValiConfig.RISK_PROFILING_SIGMOID_SHIFT,
                ValiConfig.RISK_PROFILING_SIGMOID_SPREAD
            )

        return risk_profile_penalty
