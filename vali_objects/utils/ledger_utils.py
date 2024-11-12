# developer: trdougherty
import math
import numpy as np
import copy

from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedgerData
from vali_objects.utils.functional_utils import FunctionalUtils


class LedgerUtils:
    @staticmethod
    def risk_free_adjustment(checkpoints: list[PerfCheckpoint]) -> list[PerfCheckpoint]:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints

        Returns:
            float - the risk-free adjustment
        """
        if len(checkpoints) == 0:
            return checkpoints

        checkpoints_copy = copy.deepcopy(checkpoints)

        risk_free_rate = ValiConfig.MS_RISK_FREE_RATE

        for checkpoint in checkpoints_copy:
            checkpoint.loss += risk_free_rate * checkpoint.accum_ms

        return checkpoints_copy

    @staticmethod
    def recent_drawdown(checkpoints: list[PerfCheckpoint], restricted: bool = True) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
            restricted: bool - whether to restrict the lookback window

        Returns:
            float - the most recent drawdown
        """
        drawdown_lookback_window = ValiConfig.RETURN_SHORT_LOOKBACK_LEDGER_WINDOWS
        if drawdown_lookback_window <= 0:
            raise ValueError("Drawdown lookback window must be greater than 0")

        if len(checkpoints) == 0:
            return 1.0

        # Compute the drawdown of the checkpoints
        if restricted:
            checkpoints = checkpoints[-drawdown_lookback_window:]

        drawdowns = [checkpoint.mdd for checkpoint in checkpoints]

        recent_drawdown = min(drawdowns)
        recent_drawdown = np.clip(recent_drawdown, 0.0, 1.0)

        return recent_drawdown

    @staticmethod
    def drawdown_percentage(drawdown_decimal: float) -> float:
        """
        Args:
            drawdown_decimal: float - the drawdown value

        Returns:
            float - the drawdown percentage
        """
        if drawdown_decimal >= 1:
            return 0

        if drawdown_decimal <= 0:
            return 100

        return np.clip((1 - drawdown_decimal) * 100, 0, 100)

    @staticmethod
    def mdd_lower_augmentation(drawdown_percent: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if drawdown_percent <= 0:
            return 0

        if drawdown_percent > 100:
            return 0

        drawdown_minvalue = ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE

        # Drawdown value
        if drawdown_percent <= drawdown_minvalue:
            return 0

        return 1

    @staticmethod
    def mdd_upper_augmentation(drawdown_percent: float) -> float:
        """
        Should only look at the upper region of the drawdown
        """
        if drawdown_percent <= 0:
            return 0

        if drawdown_percent > 100:
            return 0

        drawdown_maxvalue = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE
        drawdown_scaling = ValiConfig.DRAWDOWN_UPPER_SCALING

        upper_penalty = (-drawdown_percent + drawdown_maxvalue) / drawdown_scaling
        return float(np.clip(upper_penalty, 0, 1))

    @staticmethod
    def mdd_base_augmentation(drawdown_percent: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if drawdown_percent <= 0:
            return 0

        if drawdown_percent > 100:
            return 0

        return float(1 / drawdown_percent)

    @staticmethod
    def mdd_augmentation(drawdown: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if drawdown <= 0 or drawdown > 1:
            return 0

        recent_drawdown_percentage = LedgerUtils.drawdown_percentage(drawdown)
        if recent_drawdown_percentage <= ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE:
            return 0

        if recent_drawdown_percentage >= ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE:
            return 0

        base_augmentation = LedgerUtils.mdd_base_augmentation(recent_drawdown_percentage)
        lower_augmentation = LedgerUtils.mdd_lower_augmentation(recent_drawdown_percentage)
        upper_augmentation = LedgerUtils.mdd_upper_augmentation(recent_drawdown_percentage)

        drawdown_penalty = base_augmentation * lower_augmentation * upper_augmentation
        return float(drawdown_penalty)

    @staticmethod
    def max_drawdown_threshold_penalty(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
        """
        drawdown_limit = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE

        if len(checkpoints) == 0:
            return 0

        effective_drawdown = LedgerUtils.recent_drawdown(checkpoints)
        effective_drawdown_percentage = LedgerUtils.drawdown_percentage(effective_drawdown)

        if effective_drawdown_percentage >= drawdown_limit:
            return 0

        return 1

    @staticmethod
    def approximate_drawdown(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
        """
        upper_percentile = 100 * (1 - ValiConfig.APPROXIMATE_DRAWDOWN_PERCENTILE)
        if len(checkpoints) == 0:
            return 0

        # Compute the drawdown of the checkpoints
        drawdowns = [checkpoint.mdd for checkpoint in checkpoints]
        effective_drawdown = np.percentile(drawdowns, upper_percentile)
        final_drawdown = np.clip(effective_drawdown, 0, 1.0)

        return final_drawdown

    @staticmethod
    def effective_drawdown(recent_drawdown: float, approximate_drawdown: float) -> float:
        """
        Args:
            recent_drawdown: float - the most recent drawdown, as a value between 0 and 1
            approximate_drawdown: float - the approximate drawdown, as a value between 0 and 1

        Returns:
            float - the effective drawdown
        """
        if recent_drawdown <= 0:
            return 0

        if approximate_drawdown <= 0:
            return 0

        return min(recent_drawdown, approximate_drawdown)

    @staticmethod
    def risk_normalization(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
        """
        if len(checkpoints) == 0:
            return 0

        recent_drawdown = LedgerUtils.recent_drawdown(checkpoints)
        approximate_drawdown = LedgerUtils.approximate_drawdown(checkpoints)

        effective_drawdown = LedgerUtils.effective_drawdown(recent_drawdown, approximate_drawdown)
        drawdown_penalty = LedgerUtils.mdd_augmentation(effective_drawdown)
        return drawdown_penalty

    @staticmethod
    def daily_consistency_ratio(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints

        Returns:
            float - the daily consistency ratio
        """
        return LedgerUtils.time_consistency_ratio(
            checkpoints,
            ValiConfig.DAILY_CHECKPOINTS
        )

    @staticmethod
    def daily_consistency_penalty(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints

        Returns:
            float - the daily consistency penalty
        """
        return FunctionalUtils.sigmoid(
            LedgerUtils.daily_consistency_ratio(checkpoints),
            ValiConfig.DAILY_SIGMOID_SHIFT,
            ValiConfig.DAILY_SIGMOID_SPREAD
        )

    @staticmethod
    def biweekly_consistency_ratio(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints

        Returns:
            float - the biweekly consistency ratio
        """
        return LedgerUtils.time_consistency_ratio(
            checkpoints,
            ValiConfig.BIWEEKLY_CHECKPOINTS
        )

    @staticmethod
    def biweekly_consistency_penalty(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints

        Returns:
            float - the biweekly consistency penalty
        """
        return FunctionalUtils.sigmoid(
            LedgerUtils.biweekly_consistency_ratio(checkpoints),
            ValiConfig.BIWEEKLY_SIGMOID_SHIFT,
            ValiConfig.BIWEEKLY_SIGMOID_SPREAD
        )

    @staticmethod
    def time_consistency_ratio(
            checkpoints: list[PerfCheckpoint],
            window_length: int
    ) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
            window_length: int - the length of the window

        Returns:
            float - the ledger consistency ratio
        """
        if len(checkpoints) <= 0:
            return 1

        checkpoint_margins = np.array([checkpoint.gain + checkpoint.loss for checkpoint in checkpoints])
        unrealized_return = sum(checkpoint_margins)

        if unrealized_return == 0:
            # all the returns are zero, so ratio between larger element and total is 1
            return 1

        convolution_window = np.ones(window_length)
        convolution_margins = np.convolve(checkpoint_margins, convolution_window, mode='valid')

        if unrealized_return < 0:
            numerator = min(convolution_margins)
        else:
            numerator = max(convolution_margins)

        return np.clip(abs(numerator / unrealized_return), 0, 1)

    @staticmethod
    def cumulative(ledger: dict[str, PerfLedgerData]) -> dict[str, dict]:
        """
        Adds the cumulative return of the ledger to each checkpoint.
        Args:
            ledger: dict[str, PerfLedger] - the ledger of the miners

        Returns:
            dict[str, dict] - the cumulative return of the ledger
        """
        ledger_dict = {k: v.to_dict() for k, v in ledger.items()}
        ledger_copy = copy.deepcopy(ledger_dict)

        for miner, miner_ledger in ledger_copy.items():
            return_overall = 1.0
            if len(miner_ledger['cps']) == 0:
                continue

            for cp in miner_ledger['cps']:
                return_value = math.exp(cp['gain'] + cp['loss'])
                return_overall *= return_value
                cp['overall_returns'] = return_overall

        return ledger_copy
