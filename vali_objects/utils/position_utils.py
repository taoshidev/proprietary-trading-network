# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import math
import numpy as np
import copy
from collections import defaultdict
from datetime import datetime

from typing import Union

from vali_objects.position import Position
from vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.functional_utils import FunctionalUtils

from time_util.time_util import TimeUtil


class PositionUtils:
    @staticmethod
    def filter_single_miner(
            positions: list[Position],
            evaluation_time_ms: int,
            lookback_time_ms: int = None
    ) -> list[Position]:
        """
        Restricts to positions which were closed in the prior lookback window
        """
        if lookback_time_ms is None:
            lookback_time_ms = ValiConfig.TARGET_LEDGER_WINDOW_MS

        lookback_threshold_ms = evaluation_time_ms - lookback_time_ms

        subset_positions = []
        for position in positions:
            if position.is_closed_position is True and position.open_ms >= lookback_threshold_ms:
                subset_positions.append(position)

            if position.is_closed_position is False and position.return_at_close < 1:
                subset_positions.append(position)

        return subset_positions

    @staticmethod
    def filter(
            positions: dict[str, list[Position]],
            evaluation_time_ms: int,
            lookback_time_ms: int = None
    ) -> dict[str, list[Position]]:
        """
        Restricts to positions which were closed in the prior lookback window
        """
        updated_positions = {}

        for miner_hotkey, miner_positions in positions.items():
            updated_positions[miner_hotkey] = PositionUtils.filter_single_miner(
                miner_positions,
                evaluation_time_ms,
                lookback_time_ms
            )

        return updated_positions

    @staticmethod
    def filter_recent(
            positions: dict[str, list[Position]],
            evaluation_time_ms: int,
            lookback_time_ms: int = None,
            lookback_recent_time_ms: int = None
    ) -> dict[str, list[Position]]:
        """
        Restricts to positions which were closed in the prior lookback window
        """
        updated_positions = {}

        if lookback_time_ms is None:
            lookback_time_ms = ValiConfig.TARGET_LEDGER_WINDOW_MS

        if lookback_recent_time_ms is None:
            lookback_recent_time_ms = ValiConfig.RETURN_SHORT_LOOKBACK_TIME_MS

        lookback_recent_threshold_ms = evaluation_time_ms - lookback_recent_time_ms

        for miner_hotkey, miner_positions in positions.items():
            filtered_miner_positions = PositionUtils.filter_single_miner(
                miner_positions,
                evaluation_time_ms,
                lookback_time_ms
            )

            recent_filtered_miner_positions = [position for position in filtered_miner_positions if
                                               position.close_ms >= lookback_recent_threshold_ms]
            updated_positions[miner_hotkey] = recent_filtered_miner_positions

        return updated_positions

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
    def average_leverage(positions: list[Position]) -> float:
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
    def total_duration(
            positions: list[Position]
    ) -> int:
        """
        Args:
            positions: list[Position] - the list of positions
        """
        time_deltas = []

        for position in positions:
            if position.is_closed_position:
                time_deltas.append(position.close_ms - position.open_ms)

        return sum(time_deltas)

    @staticmethod
    def time_consistency_penalty(
            positions: list[Position]
    ) -> float:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions
        """
        return_time_spread = ValiConfig.POSITIONAL_RETURN_TIME_SIGMOID_SPREAD
        return_time_shift = ValiConfig.POSITIONAL_RETURN_TIME_SIGMOID_SHIFT

        # Need at least two positions for this to even make sense
        if len(positions) <= 1:
            return 0.0

        return FunctionalUtils.sigmoid(
            PositionUtils.time_consistency_ratio(positions),
            return_time_shift,
            return_time_spread
        )

    @staticmethod
    def time_consistency_ratio(
            positions: list[Position],
            time_window: Union[int, None] = None
    ) -> float:
        """
        Returns the ratio associated with the time window of the realized returns
        Args:
            positions: list[Position] - the list of positions
            time_window: int - window used to aggregate the returns on closed positions

        Returns:
            float - the ratio of the realized returns in the time window relative to total returns
        """
        if len(positions) == 0:
            return 1  # no aggregate returns, so it should capture the full penalty

        if time_window is None:
            time_window = ValiConfig.POSITIONAL_RETURN_TIME_WINDOW_MS

        close_times = np.array([position.close_ms for position in positions])
        returns = np.log([position.return_at_close for position in positions])
        total_return = np.sum(returns)

        # If there is no return, the ratio is 1, as our denominator is invalid
        if total_return == 0:
            return 1

        # Initialize an empty list to store the results
        sums_in_window = []

        # Iterate through each time point
        for i in range(len(close_times)):
            # Define the start and end of the sliding window
            start_time: int = close_times[i]
            end_time: int = start_time + time_window

            # Sum values within the window
            sum_values = returns[
                (close_times >= start_time) &
                (close_times < end_time)
                ].sum()

            # Store the result
            sums_in_window.append(sum_values)

        if total_return > 0:
            largest_windowed_contribution = max(sums_in_window)
        else:
            largest_windowed_contribution = min(sums_in_window)

        return np.clip(largest_windowed_contribution / total_return, 0, 1)

    @staticmethod
    def returns_ratio_penalty(
            positions: list[Position]
    ) -> float:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions
        """
        max_return_spread = ValiConfig.MAX_RETURN_SIGMOID_SPREAD
        max_return_shift = ValiConfig.MAX_RETURN_SIGMOID_SHIFT

        # Need at least two positions for this to even make sense
        if len(positions) <= 1:
            return 0.0

        max_return_ratio = PositionUtils.returns_ratio(positions)
        return FunctionalUtils.sigmoid(
            max_return_ratio,
            max_return_shift,
            max_return_spread
        )

    @staticmethod
    def returns_ratio(
            positions: list[Position]
    ) -> float:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions
        """
        daily_sums = defaultdict(float)

        closed_positions = [position for position in positions if position.is_closed_position]
        closed_return = sum([math.log(position.return_at_close) for position in closed_positions])
        if closed_return == 0:
            return 1  # no aggregate returns, so it should capture the full penalty

        for position in closed_positions:
            date = datetime.utcfromtimestamp(position.close_ms / 1000).date()
            daily_sums[date] += math.log(position.return_at_close)

        daily_log_returns = daily_sums.values()

        if closed_return > 0:
            positive_returns = [x for x in daily_log_returns if x > 0]
            numerator = max(positive_returns)
        else:
            negative_returns = [x for x in daily_log_returns if x < 0]
            numerator = min(negative_returns)

        denominator = closed_return
        max_return_ratio = np.clip(numerator / denominator, 0, 1)

        return max_return_ratio

    @staticmethod
    def flatten(
            positions: dict[str, list[Position]]
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
        """
        positions_list = []
        for miner_key, miner_positions in positions.items():
            for position in miner_positions:
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
            current_time: int - the current time
            constrain_lookback: bool - whether to constrain the lookback

        Return:
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
