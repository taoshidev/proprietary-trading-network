# developer: trdougherty
from typing import Union
import numpy as np
import math
import pandas as pd

from vali_objects.vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.utils.functional_utils import FunctionalUtils
from vali_objects.utils.position_utils import PositionUtils


class PositionPenalties:
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
            PositionPenalties.time_consistency_ratio(positions),
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
        returns = np.log([
            max(position.return_at_close, .00001)  # Prevent math domain error
            for position in positions])
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

        max_return_ratio = PositionPenalties.returns_ratio(positions)
        return FunctionalUtils.sigmoid(
            max_return_ratio,
            max_return_shift,
            max_return_spread
        )

    @staticmethod
    def martingale_penalty(
            positions: list[Position]
    ) -> float:
        """
        Returns the martingale penalty for each miner

        Args:
            positions: dict[str, list[Position]] - the list of positions with translated leverage
        """
        martingale_score = PositionPenalties.martingale_score(positions)
        return FunctionalUtils.sigmoid(
            martingale_score,
            ValiConfig.MARTINGALE_SHIFT,
            ValiConfig.MARTINGALE_SPREAD
        )

    @staticmethod
    def martingale_score(
            positions: list[Position]
    ) -> float:
        """
        Returns the martingale penalty for each miner

        Args:
            positions: dict[str, list[Position]] - the list of positions with translated leverage
        """
        cumulative_leverage_positions = PositionUtils.cumulative_leverage_position(positions)
        return PositionPenalties.martingale_leverage_average(cumulative_leverage_positions)
        # return FunctionalUtils.martingale_score(martingale_metrics, cumulative_leverage_positions)

    # what we want to do is determine for each position is to determine the relative drawdown percentage
    @staticmethod
    def martingale_leverage_average(
            positions: list[Position]
    ) -> float:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions
        """
        if len(positions) < 1:
            return 0.0

        martingale_values = []
        positional_returns = []

        for position in positions:
            losing_percentages = []
            losing_multipliers = []

            return_at_close = (position.return_at_close-1)*100
            if return_at_close <= 0:
                continue

            entry_order = position.orders[0]
            entry_price = entry_order.price
            entry_leverage = abs(entry_order.leverage)

            for order in position.orders[1:]:
                price = order.price
                leverage = abs(order.leverage)

                losing = price < entry_price
                if losing and leverage > 0 and leverage > entry_leverage:
                    losing_percent = (1-(price / entry_price)) * 100
                    losing_leverage_multiplier = leverage / entry_leverage

                    losing_percentages.append(losing_percent)
                    losing_multipliers.append(losing_leverage_multiplier)

            if len(losing_percentages) < 1 or len(losing_multipliers) < 1:
                continue

            losing_leverage_losses = [losing_percent * losing_leverage_multiplier for losing_percent, losing_leverage_multiplier in zip(losing_percentages, losing_multipliers)]
            martingale_values.append(np.mean(losing_leverage_losses) / return_at_close)
            positional_returns.append(return_at_close)

        if len(martingale_values) == 0:
            return 0

        positional_returns = np.array(positional_returns)
        if sum(positional_returns) == 0:
            positional_returns += ValiConfig.EPSILON

        return np.average(martingale_values, weights=positional_returns)

    @staticmethod
    def martingale_metrics(
            positions: list[Position]
    ) -> dict[str, list[float]]:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions, with each position containing the cumulative leverage
        """

        # Need at least one positions for this to even make sense
        if len(positions) < 1:
            return {
                "losing_value_percents": [],
                "entry_holding_timing": [],
                "losing_leverages_decimal_multiplier": [],
                "positional_returns": []
            }

        losing_value_percents = []
        losing_leverages_decimal_multiplier = []
        positional_returns = []
        order_holding_timings = []
        times_readable = []
        position_times = []
        steps = []

        for position in positions:
            return_at_close = position.return_at_close
            entry_order = position.orders[0]
            entry_price = entry_order.price

            entry_leverage = abs(entry_order.leverage) + ValiConfig.EPSILON
            entry_time = position.orders[0].processed_ms
            exit_time = position.orders[-1].processed_ms
            entry_leverage = abs(entry_order.leverage)
            direction_is_long = entry_order.leverage > 0

            for step in range(1, len(position.orders)):
                order = position.orders[step]
                price = order.price
                leverage = abs(order.leverage)
                time_of_execution = order.processed_ms

                losing = price < entry_price and direction_is_long or price > entry_price and not direction_is_long
                if losing and leverage > 0:
                    losing_percent = (1-(price / entry_price)) * 100
                    losing_leverage_multiplier = leverage / entry_leverage
                    losing_entry_timing = (time_of_execution - entry_time) / (exit_time - entry_time)
                    times_readable.append(pd.to_datetime(time_of_execution, unit='ms', utc=True))
                    position_times.append(pd.to_datetime(entry_time, unit='ms', utc=True))
                    order_holding_timings.append(losing_entry_timing)
                    losing_value_percents.append(losing_percent)
                    losing_leverages_decimal_multiplier.append(losing_leverage_multiplier)
                    positional_returns.append(return_at_close)
                    steps.append(step)

        return {
            "losing_value_percents": losing_value_percents,
            "entry_holding_timing": order_holding_timings,
            "losing_leverages_decimal_multiplier": losing_leverages_decimal_multiplier,
            "positional_returns": positional_returns,
            "times_readable": times_readable,
            "position_times": position_times,
            "steps": steps
        }

    @staticmethod
    def returns_ratio(
            positions: list[Position]
    ) -> float:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions
        """
        closed_positions = [position for position in positions if position.is_closed_position]
        closed_position_returns = [math.log(
                                        max(position.return_at_close, .00001))  # Prevent math domain error
                                    for position in closed_positions]
        closed_return = sum(closed_position_returns)

        # Return early if there will be an issue with the ratio denominator
        if closed_return == 0:
            return 1

        numerator = max(closed_position_returns) if closed_return > 0 else min(closed_position_returns)
        denominator = closed_return

        max_return_ratio = np.clip(numerator / denominator, 0, 1)

        return max_return_ratio
