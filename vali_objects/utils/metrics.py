
import math
import numpy as np

from vali_objects.vali_config import ValiConfig
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerData



class Metrics:
    @staticmethod
    def ann_excess_return(log_returns: list[float]) -> float:
        """
        Calculates annualized excess return using mean daily log returns and mean daily 1yr risk free rate.
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        """
        annual_risk_free_rate = ValiConfig.ANNUAL_RISK_FREE_PERCENTAGE
        trading_days = ValiConfig.MARKET_OPEN_DAYS

        mean_daily_log_returns = np.mean(log_returns)

        # Annualize the mean daily excess returns
        annualized_excess_return = math.exp(mean_daily_log_returns * trading_days) - annual_risk_free_rate
        return annualized_excess_return

    @staticmethod
    def ann_volatility(log_returns: list[float]) -> float:
        """
        Calculates annualized volatility ASSUMING DAILY OBSERVATIONS
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        """
        # Annualize volatility of the daily log returns assuming sample variance
        trading_days = ValiConfig.MARKET_OPEN_DAYS

        window = len(log_returns)
        if window == 0:
            return np.inf

        ann_factor = trading_days / window
        annualized_volatility = np.sqrt(np.var(log_returns, ddof=1) * ann_factor)
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
        trading_days = ValiConfig.MARKET_OPEN_DAYS

        downside_returns = [log_return for log_return in log_returns if log_return < target]
        window = len(downside_returns)
        if window == 0:
            return np.inf

        ann_factor = trading_days / window
        annualized_downside_volatility = np.sqrt(np.var(downside_returns, ddof=1) * ann_factor)
        return annualized_downside_volatility

    @staticmethod
    def base_return_log(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner

        Returns:
             The aggregate total return of the miner as a log total
        """
        return sum(log_returns)

    @staticmethod
    def base_return(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner

        Returns:
             The aggregate total return of the miner as a percentage
        """
        return math.exp(Metrics.base_return_log(log_returns)) - 1

    @staticmethod
    def risk_adjusted_return(returns: list[float], ledger: PerfLedgerData) -> float:
        """
        Args:
            returns: list of returns
            ledger: the ledger of the miner
        """
        # Positional Component
        if len(returns) == 0:
            return 0.0

        base_return = Metrics.base_return_log(returns)
        risk_normalization_factor = LedgerUtils.risk_normalization(ledger.cps)

        return base_return * risk_normalization_factor

    @staticmethod
    def sharpe(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
        """
        if len(log_returns) == 0:
            return 0.0

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
        if len(log_returns) == 0:
            return 0.0

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
    def sortino(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
        """
        if len(log_returns) == 0:
            return 0.0

        # Hyperparameter
        min_downside = ValiConfig.SORTINO_DOWNSIDE_MINIMUM

        # Sortino ratio is calculated as the mean of the returns divided by the standard deviation of the negative returns
        excess_return = Metrics.ann_excess_return(log_returns)
        downside_volatility = Metrics.ann_downside_volatility(log_returns)

        return excess_return / max(downside_volatility, min_downside)
