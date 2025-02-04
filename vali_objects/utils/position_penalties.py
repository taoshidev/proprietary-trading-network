# developer: trdougherty
import numpy as np
import pandas as pd

from vali_objects.vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.utils.functional_utils import FunctionalUtils
from vali_objects.utils.position_utils import PositionUtils


class PositionPenalties:

    @staticmethod
    def martingale_penalty(
            positions: list[Position],
            evaluation_time_ms: int = None
    ) -> float:
        """
        Returns the martingale penalty for each miner

        Args:
            positions: dict[str, list[Position]] - the list of positions with translated leverage
        """
        martingale_score = PositionPenalties.martingale_score(positions, evaluation_time_ms)
        return FunctionalUtils.sigmoid(
            martingale_score,
            ValiConfig.MARTINGALE_SHIFT,
            ValiConfig.MARTINGALE_SPREAD
        )

    @staticmethod
    def martingale_score(
            positions: list[Position],
            evaluation_time_ms: int = None,
    ) -> float:
        """
        Returns the martingale penalty for each miner

        Args:
            positions: dict[str, list[Position]] - the list of positions with translated leverage
        """
        cumulative_leverage_positions = PositionUtils.cumulative_leverage_position(positions, evaluation_time_ms)
        return PositionPenalties.martingale_percentile(cumulative_leverage_positions)
        # return FunctionalUtils.martingale_score(martingale_metrics, cumulative_leverage_positions)

    # what we want to do is determine for each position is to determine the relative drawdown percentage
    @staticmethod
    def martingale_percentile(
            positions: list[Position]
    ) -> float:
        """
        Returns the penalty associated with uneven distributions for realized returns

        Args:
            positions: list[Position] - the list of positions
        """
        if len(positions) < 1:
            return 0.0

        position_is_martingale = []
        positional_returns = []

        for position in positions:
            step_count = 0
            return_at_close = position.return_at_close

            entry_order = position.orders[0]
            entry_price = entry_order.price
            max_leverage = abs(entry_order.leverage)

            for order in position.orders[1:]:
                price = order.price
                leverage = abs(order.leverage)

                losing = price < entry_price and entry_order.leverage > 0 or price > entry_price and entry_order.leverage < 0
                if losing and leverage > max_leverage:
                    step_count += 1
                    max_leverage = max(max_leverage, leverage)

            positional_returns.append(return_at_close ** ValiConfig.MARTINGALE_CONCENTRATION)
            if step_count > ValiConfig.MARTINGALE_STEP_THRESHOLD:
                position_is_martingale.append(True)
            else:
                position_is_martingale.append(False)

        martingale_binaries = np.array(position_is_martingale, dtype=int)
        martingale_weights = np.array(positional_returns)
        return np.average(martingale_binaries, weights=martingale_weights)

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
