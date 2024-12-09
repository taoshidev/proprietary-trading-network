# developer: trdougherty
from typing import Union
import numpy as np
import math

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
    def miner_martingale_penalties(
            hotkey_positions: dict[str, list[Position]]
    ) -> dict[str, float]:
        """
        Returns the martingale penalty for each miner

        Args:
            hotkey_positions: dict[str, list[Position]] - the list of positions with translated leverage
        """
        miner_martingales = PositionPenalties.miner_martingales(hotkey_positions)
        return {
            miner_id: martingale_score < ValiConfig.MARTINGALE_THRESHOLD
            for miner_id, martingale_score in miner_martingales.items()
        }

    @staticmethod
    def miner_martingales(
            hotkey_positions: dict[str, list[Position]]
    ) -> dict[str, float]:
        """
        Returns the martingale penalty for each miner

        Args:
            hotkey_positions: dict[str, list[Position]] - the list of positions with translated leverage
        """
        cumulative_leverage_dict = PositionUtils.cumulative_leverage_dict(hotkey_positions)
        return {
            miner_id: PositionPenalties.martingale(positions)
            for miner_id, positions in cumulative_leverage_dict.items()
        }

    @staticmethod
    def martingale(
            positions: list[Position]
    ) -> float:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions, with each position containing the cumulative leverage
        """

        # Need at least one positions for this to even make sense
        if len(positions) < 1:
            return 0.0

        martingale_score = 0
        for position in positions:
            entry_order = position.orders[0]
            entry_price = entry_order.price
            entry_leverage = abs(entry_order.leverage)
            positional_martingale = 0

            for order in position.orders[1:]:
                price = order.price
                leverage = abs(order.leverage)

                losing = price < entry_price
                if losing:
                    positional_martingale = leverage / entry_leverage

            martingale_score = max(martingale_score, abs(positional_martingale))

        return martingale_score

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
