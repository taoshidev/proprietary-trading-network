import numpy as np
from sklearn.linear_model import LinearRegression

from vali_objects.vali_config import ValiConfig
from vali_objects.position import Position


class FunctionalUtils:
    @staticmethod
    def sigmoid(x: float, shift: float = 0, spread: float = 1) -> float:
        """
        Args:
            x: Value to be transformed
            shift: displacement of the sigmoid function
            spread: spread of the sigmoid function

        Returns:
            float - the sigmoid value of the input
        """
        if spread == 0:
            raise ValueError("The spread parameter must be different from 0")

        exp_term = np.clip(spread * (x - shift), -100, 100)
        return np.clip(1 / (1 + np.exp(exp_term)), 0, 1)

    @staticmethod
    def concentration(x: list[float]) -> float:
        """
        Args:
            x: list[float] - the list of float values

        Returns:
            float - the concentration penalty
        """
        if len(x) == 0:
            return 1

        positional_returns = [abs(xi) for xi in x]
        total_return = max(sum(positional_returns), ValiConfig.MIN_HHI_RETURN)

        hhi = [(x / total_return) ** 2 for x in positional_returns]
        return sum(hhi)

    @staticmethod
    def concentration_readable(x: list[float]) -> float:
        """
        Args:
            x: list[Position] - the list of positions

        Returns:
            float - the concentration penalty
        """
        return FunctionalUtils.concentration(x) * 10_000

    @staticmethod
    def concentration_penalty(x: list[float]) -> float:
        """
        Args:
            x: list[float] - the list of positions

        Returns:
            float - the concentration penalty
        """
        concentration_spread = ValiConfig.POSITIONAL_CONCENTRATION_SIGMOID_SPREAD
        concentration_shift = ValiConfig.POSITIONAL_CONCENTRATION_SIGMOID_SHIFT

        return FunctionalUtils.sigmoid(
            FunctionalUtils.concentration(x),
            concentration_shift,
            concentration_spread
        )

    @staticmethod
    def martingale_score(
            metrics: dict[str, list[float]],
            positions: list[Position]
    ) -> float:
        """
        Returns the martingale score for each miner, which is a regression based on the leverage step in and losses

        Args:
            metrics: list[tuple[float, float]] - the list of metrics for each miner
            positions: list[Position] - the list of positions with translated leverage
        """

        holding_intervals = np.array(metrics['entry_holding_timing'])
        y = losses = np.array(metrics['losing_value_percents'])
        x = log_leverages = np.log(metrics['losing_leverages_decimal_multiplier'])

        weighted_loss = np.average(losses, weights=log_leverages)

        aggregate_return = (np.prod(np.array([position.return_at_close for position in positions]))-1)*100 + \
            ValiConfig.EPSILON

        if len(x) == 0 or len(y) == 0:
            return 0

        if len(x) != len(y):
            raise ValueError("The lengths of the input arrays must be the same")

        return np.average(x, weights=y) / aggregate_return
