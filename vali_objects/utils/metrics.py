
import math
import numpy as np
from scipy.stats import ttest_1samp
from typing import Union

from vali_objects.vali_config import ValiConfig
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint

class Metrics:

    @staticmethod
    def weighted_log_returns(log_returns: list[float]) -> list[float]:
        if len(log_returns) < 1:
            return []

        weighting_distribution = Metrics.weighting_distribution(log_returns)
        weighted_returns = np.multiply(np.array(log_returns), weighting_distribution) / np.sum(weighting_distribution)
        return list(weighted_returns)

    @staticmethod
    def weighting_distribution(log_returns: Union[list[float], np.ndarray]) -> np.ndarray:
        """
        Returns the weighting distribution that decays from max_weight to min_weight
        using the configured decay rate
        """
        max_weight = ValiConfig.WEIGHTED_AVERAGE_DECAY_MAX
        min_weight = ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN
        decay_rate = ValiConfig.WEIGHTED_AVERAGE_DECAY_RATE

        if len(log_returns) < 1:
            return np.ones(0)

        weighting_distribution_days = np.arange(0, len(log_returns))

        # Calculate decay from max to min
        weight_range = max_weight - min_weight
        decay_values = min_weight + (weight_range * np.exp(-decay_rate * weighting_distribution_days))

        return decay_values[::-1][-len(log_returns):]

    @staticmethod
    def average(log_returns: Union[list[float], np.ndarray], weighting=False, indices: Union[list[int], None] = None) -> float:
        """
        Returns the mean of the log returns
        """
        if len(log_returns) == 0:
            return 0.0

        weighting_distribution = Metrics.weighting_distribution(log_returns)

        if indices is not None and len(indices) != 0:
            indices = [i for i in indices if i in range(len(log_returns))]
            log_returns = [log_returns[i] for i in indices]
            weighting_distribution = [weighting_distribution[i] for i in indices]

        if weighting:
            avg_value = np.average(log_returns, weights=weighting_distribution)
        else:
            avg_value = np.mean(log_returns)

        return float(avg_value)

    @staticmethod
    def variance(log_returns: list[float], ddof: int = 1, weighting=False, indices: Union[list[int], None] = None) -> float:
        """
        Returns the standard deviation of the log returns
        """
        if len(log_returns) == 0:
            return 0.0

        window = len(indices) if indices is not None else len(log_returns)
        if window < ddof + 1:
            return np.inf

        return Metrics.average((np.array(log_returns) - Metrics.average(log_returns, weighting=weighting, indices=indices)) ** 2, weighting=weighting, indices=indices)

    @staticmethod
    def ann_excess_return(log_returns: list[float], weighting=False) -> float:
        """
        Calculates annualized excess return using mean daily log returns and mean daily 1yr risk free rate.
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        """
        annual_risk_free_rate = ValiConfig.ANNUAL_RISK_FREE_DECIMAL
        days_in_year = ValiConfig.DAYS_IN_YEAR

        if len(log_returns) == 0:
            return 0.0

        # Annualize the mean daily excess returns
        annualized_excess_return = (Metrics.average(log_returns, weighting=weighting) * days_in_year) - annual_risk_free_rate
        return annualized_excess_return

    @staticmethod
    def ann_volatility(log_returns: list[float], ddof: int = 1, weighting=False, indices: list[int]=None) -> float:
        """
        Calculates annualized volatility ASSUMING DAILY OBSERVATIONS
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        ddof int: Delta Degrees of Freedom. The divisor used in the calculation is N - ddof, where N represents the number of elements.
        weighting bool: Whether to use weighted average.
        indices list[int]: The indices of the log returns to consider.
        """
        if indices is None:
            indices = list(range(len(log_returns)))
            
        # Annualize volatility of the daily log returns assuming sample variance
        days_in_year = ValiConfig.DAYS_IN_YEAR

        window = len(indices)
        if window < ddof + 1:
            return np.inf

        annualized_volatility = np.sqrt(Metrics.variance(log_returns, ddof=ddof, weighting=weighting, indices=indices) * days_in_year)

        return float(annualized_volatility)

    @staticmethod
    def ann_downside_volatility(log_returns: list[float], target: int = ValiConfig.DAILY_LOG_RISK_FREE_RATE, weighting=False) -> float:
        """
        Args:
            log_returns: list[float] - Daily Series of log returns.
            target: int: The target return (default: 0).
            weighting: bool: Whether to use weighted average.

        Returns:
            The downside annualized volatility as a float assuming sample variance
        """
        indices = [i for i, log_return in enumerate(log_returns) if log_return < target]
        return Metrics.ann_volatility(log_returns, weighting=weighting, indices=indices)

    @staticmethod
    def base_return_log(log_returns: list[float], weighting=False) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            weighting: whether to use weighted average

        Returns:
             The aggregate total return of the miner as a log total
        """
        if len(log_returns) == 0:
            return 0.0

        return float(Metrics.average(log_returns, weighting=weighting)) * ValiConfig.DAYS_IN_YEAR

    @staticmethod
    def base_return_log_percentage(log_returns: list[float], weighting=False, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            weighting: whether to use weighted average

        Returns:
             The aggregate total return of the miner as a percentage log total
                """
        if len(log_returns) == 0:
            return 0.0

        return Metrics.average(log_returns, weighting=weighting) * ValiConfig.DAYS_IN_YEAR * 100

    @staticmethod
    def base_return(log_returns: list[float], weighting: bool = False) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            weighting: whether to use weighted average

        Returns:
             The aggregate total return of the miner as a percentage
        """
        return float((math.exp(Metrics.base_return_log(log_returns, weighting=weighting)) - 1) * 100)

    @staticmethod
    def calmar(log_returns: list[float], checkpoints: list[PerfCheckpoint], weighting: bool = False, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            checkpoints: the ledger of the miner
            weighting: whether to use weighted average
        """
        # Positional Component
        if len(log_returns) == 0:
            return 0.0

        base_return_percentage = Metrics.base_return_log_percentage(log_returns, weighting=weighting)
        drawdown_normalization_factor = LedgerUtils.risk_normalization(checkpoints)

        return float(base_return_percentage * drawdown_normalization_factor)

    @staticmethod
    def sharpe(log_returns: list[float], bypass_confidence: bool = False, weighting: bool = False, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
        """
        # Need a large enough sample size
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            if not bypass_confidence:
                return ValiConfig.SHARPE_NOCONFIDENCE_VALUE

        # Hyperparameter
        min_std_dev = ValiConfig.SHARPE_STDDEV_MINIMUM

        excess_return = Metrics.ann_excess_return(log_returns, weighting=weighting)
        volatility = Metrics.ann_volatility(log_returns, weighting=weighting)
        
        return float(excess_return / max(volatility, min_std_dev))

    @staticmethod
    def omega(log_returns: list[float], bypass_confidence: bool = False, weighting: bool = False, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
        """
        # Need a large enough sample size
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            if not bypass_confidence:
                return ValiConfig.OMEGA_NOCONFIDENCE_VALUE

        if weighting:
            weighing_array = Metrics.weighting_distribution(log_returns)

            positive_indices = []
            negative_indices = []

            for c,log_return in enumerate(log_returns):
                if log_return > 0:
                    positive_indices.append(c)
                else:
                    negative_indices.append(c)

            positive_indices_arr = np.array(positive_indices)
            negative_indices_arr = np.array(negative_indices)

            log_return_arr = np.array(log_returns)
            sum_of_weights_positive = max(np.sum(weighing_array[positive_indices_arr]), ValiConfig.OMEGA_LOSS_MINIMUM)
            sum_of_weights_negative = max(np.sum(weighing_array[negative_indices_arr]), ValiConfig.OMEGA_LOSS_MINIMUM)

            positive_sum = np.sum(np.multiply(log_return_arr[positive_indices_arr], weighing_array[positive_indices_arr])) * sum_of_weights_negative
            negative_sum = np.sum(np.multiply(log_return_arr[negative_indices_arr], weighing_array[negative_indices_arr])) * sum_of_weights_positive
        else:
            positive_sum = 0
            negative_sum = 0

            for log_return in log_returns:
                if log_return > 0:
                    positive_sum += log_return
                else:
                    negative_sum += log_return

        numerator = positive_sum
        denominator = max(abs(negative_sum), ValiConfig.OMEGA_LOSS_MINIMUM)

        return float(numerator / denominator)

    @staticmethod
    def statistical_confidence(log_returns: list[float], bypass_confidence: bool = False, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
        """
        # Impose a minimum sample size on the miner
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            if not bypass_confidence or len(log_returns) < 2:
                return ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE

        # Also now check for zero variance condition
        zero_variance_condition = bool(np.isclose(np.var(log_returns), 0))
        if zero_variance_condition:
            return ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE

        res = ttest_1samp(log_returns, 0, alternative='greater')
        return float(res.statistic)

    @staticmethod
    def sortino(log_returns: list[float], bypass_confidence: bool = False, weighting: bool = False, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
        """
        # Need a large enough sample size
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            if not bypass_confidence:
                return ValiConfig.SORTINO_NOCONFIDENCE_VALUE

        # Hyperparameter
        min_downside = ValiConfig.SORTINO_DOWNSIDE_MINIMUM

        # Sortino ratio is calculated as the mean of the returns divided by the standard deviation of the negative returns
        excess_return = Metrics.ann_excess_return(log_returns, weighting=weighting)
        downside_volatility = Metrics.ann_downside_volatility(log_returns, weighting=weighting)

        return float(excess_return / max(downside_volatility, min_downside))
