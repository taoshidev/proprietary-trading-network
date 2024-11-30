
import math
import numpy as np
from scipy.stats import ttest_1samp

from vali_objects.vali_config import ValiConfig
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerData, PerfCheckpoint
from vali_objects.position import Position
from vali_objects.utils.functional_utils import FunctionalUtils

class Metrics:
    @staticmethod
    def ann_excess_return(log_returns: list[float]) -> float:
        """
        Calculates annualized excess return using mean daily log returns and mean daily 1yr risk free rate.
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        """
        annual_risk_free_rate = ValiConfig.ANNUAL_RISK_FREE_DECIMAL
        days_in_year = ValiConfig.DAYS_IN_YEAR

        if len(log_returns) == 0:
            return 0.0

        mean_daily_log_returns = np.mean(log_returns)

        # Annualize the mean daily excess returns
        annualized_excess_return = (mean_daily_log_returns * days_in_year) - annual_risk_free_rate
        return annualized_excess_return

    @staticmethod
    def ann_volatility(log_returns: list[float], ddof: int = 1) -> float:
        """
        Calculates annualized volatility ASSUMING DAILY OBSERVATIONS
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        """
        # Annualize volatility of the daily log returns assuming sample variance
        days_in_year = ValiConfig.DAYS_IN_YEAR

        window = len(log_returns)
        if window < ddof + 1:
            return np.inf

        ann_factor = days_in_year / window
        
        annualized_volatility = np.sqrt(np.var(log_returns, ddof=ddof) * ann_factor)
        return annualized_volatility

    @staticmethod
    def ann_downside_volatility(log_returns: list[float], target: int = 0):
        """
        Args:
            log_returns: list[float] - Daily Series of log returns.
            target: int: The target return (default: 0).

        Returns:
            The downside annualized volatility as a float assuming sample variance
        """
        downside_returns = [log_return for log_return in log_returns if log_return < target]
        return Metrics.ann_volatility(downside_returns)

    @staticmethod
    def base_return_log(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner

        Returns:
             The aggregate total return of the miner as a log total
        """
        if len(log_returns) == 0:
            return 0.0

        return float(np.mean(log_returns)) * ValiConfig.DAYS_IN_YEAR

    @staticmethod
    def base_return(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner

        Returns:
             The aggregate total return of the miner as a percentage
        """
        return (math.exp(Metrics.base_return_log(log_returns)) - 1) * 100

    @staticmethod
    def drawdown_adjusted_return(log_returns: list[float], checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            checkpoints: the ledger of the miner
        """
        # Positional Component
        if len(log_returns) == 0:
            return 0.0

        base_return = Metrics.base_return_log(log_returns)
        drawdown_normalization_factor = LedgerUtils.risk_normalization(checkpoints)

        return base_return * drawdown_normalization_factor

    @staticmethod
    def risk_adjusted_return(returns: list[float], checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            returns: list of returns
            checkpoints: the ledger of the miner
        """
        # Positional Component
        if len(returns) == 0:
            return 0.0

        base_return = Metrics.base_return_log(returns)
        risk_normalization_factor = LedgerUtils.risk_normalization(checkpoints)

        return base_return * risk_normalization_factor

    @staticmethod
    def sharpe(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
        """
        # Need a large enough sample size
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            return ValiConfig.SHARPE_NOCONFIDENCE_VALUE

        # Hyperparameter
        min_std_dev = ValiConfig.SHARPE_STDDEV_MINIMUM

        excess_return = Metrics.ann_excess_return(log_returns)
        volatility = Metrics.ann_volatility(log_returns)
        
        return excess_return / max(volatility, min_std_dev)

    @staticmethod
    def omega(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
        """
        # Need a large enough sample size
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            return ValiConfig.OMEGA_NOCONFIDENCE_VALUE

        positive_sum = 0
        negative_sum = 0

        for log_return in log_returns:
            if log_return > 0:
                positive_sum += log_return
            else:
                negative_sum += log_return

        numerator = positive_sum
        denominator = max(abs(negative_sum), ValiConfig.OMEGA_LOSS_MINIMUM)

        return numerator / denominator

    @staticmethod
    def statistical_confidence(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
        """
        # Impose a minimum sample size on the miner
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            return ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE

        res = ttest_1samp(log_returns, 0, alternative='greater')
        return res.statistic

    @staticmethod
    def sortino(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
        """
        # Need a large enough sample size
        if len(log_returns) < ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N:
            return ValiConfig.SORTINO_NOCONFIDENCE_VALUE

        # Hyperparameter
        min_downside = ValiConfig.SORTINO_DOWNSIDE_MINIMUM

        # Sortino ratio is calculated as the mean of the returns divided by the standard deviation of the negative returns
        excess_return = Metrics.ann_excess_return(log_returns)
        downside_volatility = Metrics.ann_downside_volatility(log_returns)

        return excess_return / max(downside_volatility, min_downside)

    @staticmethod
    def concentration(log_returns: list[float], positions: list[Position]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            positions: list of positions
        """

        if len(log_returns) == 0:
            return -1

        if len(positions) == 0:
            return -1

        positional_returns = [(position.return_at_close-1)*100 for position in positions]

        pnl_concentration = FunctionalUtils.concentration(log_returns)
        position_concentration = FunctionalUtils.concentration(positional_returns)

        return -max(pnl_concentration, position_concentration)
