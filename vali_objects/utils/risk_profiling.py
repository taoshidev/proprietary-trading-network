import copy

import numpy as np
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig

from vali_objects.position import Position, Order
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair
from vali_objects.utils.functional_utils import FunctionalUtils

class RiskProfiling:
    @staticmethod
    def monatome_positions(position: Position) -> Position:
        """Return the length of monatomically increasing leverage on losing positions"""
        position_copy = copy.deepcopy(position)
        position_orders = position.orders
        position_order_subset = []

        if len(position_orders) < 2:
            return 0  # less than two steps will never flag the position

        position_direction = 1 if position_orders[0].order_type == OrderType.LONG else -1  # dot product with this and price delta will yield the return
        final_active_order = len(position_orders) if not position.is_closed_position else len(position_orders) - 1

        # We now need to track aggregate leverages
        aggregate_leverage = position_orders[0].leverage
        total_weighted_price = position_orders[0].price * position_orders[0].leverage
        avg_in_price = total_weighted_price / aggregate_leverage

        for i in range(1, final_active_order):
            price_delta = ((position_orders[i].price - avg_in_price) / avg_in_price) * 100
            leverage_delta = position_orders[i].leverage
            aggregate_leverage += leverage_delta

            if price_delta * position_direction < 0:  # price is moving against the position
                if leverage_delta * position_direction > 0:  # leverage is increasing in the original direction
                    # now we know that leverage is increasing against a losing position
                    order_copy = copy.deepcopy(position_orders[i])
                    order_copy.leverage = aggregate_leverage
                    position_order_subset.append(order_copy)

            # updating the average cost of the position
            total_weighted_price += position_orders[i].price * position_orders[i].leverage
            avg_in_price = total_weighted_price / aggregate_leverage

        # now filter for leverages that are non-monatome -> i.e. just parse out the monotone elements
        monotone_component = []
        prior_max_leverage = 0
        for position_order in position_order_subset:
            if abs(position_order.leverage) > prior_max_leverage:
                monotone_component.append(position_order)
                prior_max_leverage = abs(position_order.leverage)

        position_copy.orders = monotone_component
        return position_copy

    @staticmethod
    def risk_assessment_steps_utilization(position: Position) -> int:
        """Flags if the number of steps for a losing position is a concern"""
        position_orders = position.orders
        position_flagged_orders = []

        if len(position_orders) < 2:
            return 0  # less than two steps will never flag the position

        position_direction = 1 if position_orders[0].order_type == OrderType.LONG else -1  # dot product with this
        final_active_order = len(position_orders) if not position.is_closed_position else len(position_orders) - 1

        aggregate_leverage = position_orders[0].leverage
        total_weighted_price = position_orders[0].price * position_orders[0].leverage
        avg_in_price = total_weighted_price / aggregate_leverage

        for i in range(1, final_active_order):
            price_delta = ((position_orders[i].price - avg_in_price) / avg_in_price) * 100
            leverage_delta = position_orders[i].leverage
            aggregate_leverage += leverage_delta

            if price_delta * position_direction < 0:  # price is moving against the position
                if leverage_delta * position_direction > 0:  # leverage is increasing in the original direction
                    # now we know that leverage is increasing against a losing position
                    position_flagged_orders.append(position_orders[i])

            # updating average cost of the position
            total_weighted_price += position_orders[i].price * position_orders[i].leverage
            avg_in_price = total_weighted_price / aggregate_leverage

        return len(position_flagged_orders)

    @staticmethod
    def risk_assessment_steps_criteria(position: Position) -> bool:
        utilization = RiskProfiling.risk_assessment_steps_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_STEPS_CRITERIA

    @staticmethod
    def risk_assessment_monatome_utilization(position: Position) -> int:
        """Return the length of monatomically increasing leverage on losing positions"""
        monotone_position = RiskProfiling.monatome_positions(position)
        monotone_orders = monotone_position.orders

        return len(monotone_orders)

    @staticmethod
    def risk_assessment_monatome_criteria(position: Position) -> bool:
        utilization = RiskProfiling.risk_assessment_monatome_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_MONOTONE_CRITERIA

    @staticmethod
    def risk_assessment_margin_utilization(position: Position) -> float:
        """Flags if the position was using high levels of margin for the position"""
        # First track the margin range
        position_trade_pair: TradePair = position.trade_pair
        position_trade_pair_min_leverage = position_trade_pair.min_leverage
        position_trade_pair_max_leverage = position_trade_pair.max_leverage

        # Now compute the cumulative leverages on the position since inception
        positional_delta_leverages = [x.leverage for x in position.orders]
        positional_aggregate_leverages = np.abs(np.cumsum(positional_delta_leverages))

        base_leverage = positional_aggregate_leverages[0]
        max_utilized_leverage = max(positional_aggregate_leverages)

        margin_utilization = (max_utilized_leverage - position_trade_pair_min_leverage) / (position_trade_pair_max_leverage - position_trade_pair_min_leverage)

        return margin_utilization

    @staticmethod
    def risk_assessment_margin_criteria(position: Position) -> bool:
        utilization = RiskProfiling.risk_assessment_margin_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_MARGIN_CRITERIA

    @staticmethod
    def risk_assessment_leverage_advancement_utilization(position: Position) -> float:
        """Flags if the position was using high levels of margin for the position"""
        # First track the leverage range
        position_trade_pair: TradePair = position.trade_pair
        position_trade_pair_min_leverage = position_trade_pair.min_leverage
        position_trade_pair_max_leverage = position_trade_pair.max_leverage

        # Now compute the cumulative leverages on the position since inception
        positional_delta_leverages = [x.leverage for x in position.orders]
        positional_aggregate_leverages = np.abs(np.cumsum(positional_delta_leverages))

        base_leverage = positional_aggregate_leverages[0]
        max_utilized_leverage = max(positional_aggregate_leverages)

        leverage_advancement = max_utilized_leverage / base_leverage

        return leverage_advancement

    @staticmethod
    def risk_assessment_leverage_advancement_criteria(position: Position) -> bool:
        utilization = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE

    @staticmethod
    def risk_assessment_time_utilization(position: Position) -> float:
        """Flags the position if it is determined not to contain TWAP orders"""

        position_orders = position.orders
        if len(position_orders) < 3:
            return 0.0    # less than 3 orders (ie 1 or 2) is not meaningful in TWAP detection

        # using only the orders up to (and including) the one that brings the position’s leverage to its maximum level
        # first finding the idx where the max leverage is reached, if there are multiple indices, record the first idx
        leverage_delta_arr = [order.leverage for order in position_orders]
        aggregate_leverage_arr = np.cumsum(leverage_delta_arr)
        max_leverage_index = int(np.argmax(np.abs(aggregate_leverage_arr)))

        # then collecting orders
        position_order_subset = []
        for i in range(0, max_leverage_index + 1):    # including the order at last_max_idx
            order_copy = copy.deepcopy(position_orders[i])
            position_order_subset.append(order_copy)

        # make sure we have at least 3 meaningful orders for TWAP detection
        if len(position_order_subset) < 3:
            return 0.0

        # now we are ready to calculate order time intervals (time_deltas)
        order_times = [order.processed_ms for order in position_order_subset]
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
        utilization = RiskProfiling.risk_assessment_time_utilization(position)
        return utilization >= ValiConfig.RISK_PROFILING_TIME_CRITERIA


    @staticmethod
    def risk_profile_single(position: Position) -> dict:
        """Determines the relative risk profile of various assets."""
        # Apply the appropriate classifier based on the asset class
        steps_utilization = RiskProfiling.risk_assessment_steps_utilization(position)
        steps_criteria = int(RiskProfiling.risk_assessment_steps_criteria(position))

        monatome_utilization = RiskProfiling.risk_assessment_monatome_utilization(position)
        monatome_criteria = int(RiskProfiling.risk_assessment_monatome_criteria(position))

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
            "monatome_utilization": monatome_utilization,
            "monatome_criteria": monatome_criteria,
            "margin_utilization": margin_utilization,
            "margin_criteria": margin_criteria,
            "leverage_advancement_utilization": leverage_advancement_utilization,
            "leverage_advancement_criteria": leverage_advancement_criteria,
            "time_utilization": time_utilization,
            "time_criteria": time_criteria
        }

    @staticmethod
    def risk_profile_reporting(positions: list[Position]) -> dict:
        """Determines the relative risk profile of various assets."""
        return {position.position_uuid: RiskProfiling.risk_profile_single(position) for position in positions}

    @staticmethod
    def risk_profile_full_criteria(position: Position) -> bool:
        """Determines the relative risk profile of various assets. False meaning no risk, True meaning full risk."""
        steps_criteria = RiskProfiling.risk_assessment_steps_criteria(position)
        monotone_criteria = RiskProfiling.risk_assessment_monatome_criteria(position)
        margin_criteria = RiskProfiling.risk_assessment_margin_criteria(position)
        leverage_advancement_criteria = RiskProfiling.risk_assessment_leverage_advancement_criteria(position)
        time_criteria = RiskProfiling.risk_assessment_time_criteria(position)

        return (steps_criteria or monotone_criteria) and (margin_criteria or leverage_advancement_criteria) and time_criteria

    @staticmethod
    def risk_profile_score_list(miner_positions: list[Position]) -> float:
        """Determines the relative risk profile of various assets. 0 meaning no risk, 1 meaning full risk."""
        if len(miner_positions) == 0:
            return 0.0

        return_values = np.array([p.return_at_close for p in miner_positions])
        return_weight = return_values ** ValiConfig.RISK_PROFILING_SCOPING_MECHANIC

        criteria_weight = np.array([int(RiskProfiling.risk_profile_full_criteria(position)) for position in miner_positions])

        profile_score = np.average(criteria_weight, weights=return_weight)
        return profile_score

    @staticmethod
    def risk_profile_score(miner_positions: dict[str, list[Position]]) -> dict[str, float]:
        """Determines the relative risk profile of various assets. 0 meaning no risk, 1 meaning full risk."""
        risk_profile_scoping_mechanic = ValiConfig.RISK_PROFILING_SCOPING_MECHANIC
        miner_scores = {}

        for miner, positions in miner_positions.items():
            miner_scores[miner] = RiskProfiling.risk_profile_score_list(positions)

        return miner_scores

    @staticmethod
    def risk_profile_penalty(miner_positions: dict[str, list[Position]]) -> dict[str, float]:
        """Determines the relative risk profile of various assets. 0 meaning no risk, 1 meaning full risk."""
        risk_profile_scores = RiskProfiling.risk_profile_score(miner_positions)
        risk_profile_penalty = {}

        for miner, score in risk_profile_scores.items():
            risk_profile_penalty[miner] = FunctionalUtils.sigmoid(
                score,
                ValiConfig.RISK_PROFILING_SIGMOID_SHIFT,
                ValiConfig.RISK_PROFILING_SIGMOID_SPREAD
            )

        return risk_profile_penalty
