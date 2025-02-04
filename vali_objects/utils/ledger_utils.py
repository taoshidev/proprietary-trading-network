# developer: trdougherty
import math
import numpy as np
import copy
from datetime import datetime, timezone

from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedger


class LedgerUtils:
    @staticmethod
    def daily_return_log(checkpoints: list[PerfCheckpoint]) -> list[float]:
        """
        Calculate daily returns from performance checkpoints, only including full days
        with complete data and correct total accumulated time.

        Args:
            checkpoints: List[PerfCheckpoint] - list of checkpoints ordered by timestamp
        Returns:
            List[float] - list of daily returns for complete days
        """
        if not checkpoints:
            return []

        daily_groups = {}
        n_checkpoints_per_day = int(ValiConfig.DAILY_CHECKPOINTS)

        # Group checkpoints by date
        for cp in checkpoints:
            running_date = datetime.fromtimestamp(cp.last_update_ms / 1000, tz=timezone.utc).date()
            if cp.accum_ms == ValiConfig.TARGET_CHECKPOINT_DURATION_MS:
                if running_date not in daily_groups:
                    daily_groups[running_date] = []
                daily_groups[running_date].append(cp)

        # Calculate returns for complete days
        returns = []
        for running_date, day_checkpoints in sorted(daily_groups.items()):
            if len(day_checkpoints) == n_checkpoints_per_day:
                daily_return = sum(cp.gain + cp.loss for cp in day_checkpoints)
                returns.append(daily_return)

        return returns

    @staticmethod
    def daily_return_percentage(checkpoints: list[PerfCheckpoint]) -> list[float]:
        # First risk-free adjustment
        return [(math.exp(x)-1) * 100 if x != 0 else 0 for x in LedgerUtils.daily_return_log(checkpoints)]

    @staticmethod
    def ledger_returns(ledger: dict[str, PerfLedger]) -> dict[str, list[float]]:
        """
        Args:
            ledger: dict[str, PerfLedger] - the ledger of the miners
        """
        miner_returns = {}

        for miner, miner_ledger in ledger.items():
            miner_returns[miner] = LedgerUtils.daily_return_percentage(miner_ledger.cps if miner_ledger else [])

        return miner_returns

    @staticmethod
    def ledger_returns_log(ledger: dict[str, PerfLedger]) -> dict[str, list[float]]:
        """
        Args:
            ledger: dict[str, PerfLedger] - the ledger of the miners
        """
        miner_returns = {}

        for miner, miner_ledger in ledger.items():
            miner_returns[miner] = LedgerUtils.daily_return_log(miner_ledger.cps if miner_ledger else [])

        return miner_returns

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
        # Use the number of checkpoints if there aren't enough for the entire window
        #drawdown_lookback_window = min(drawdown_lookback_window, len(checkpoints))

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
        # if recent_drawdown_percentage <= ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE:
        #     return 0

        if recent_drawdown_percentage >= ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE:
            return 0

        base_augmentation = LedgerUtils.mdd_base_augmentation(recent_drawdown_percentage)
        # lower_augmentation = LedgerUtils.mdd_lower_augmentation(recent_drawdown_percentage)
        # upper_augmentation = LedgerUtils.mdd_upper_augmentation(recent_drawdown_percentage)

        drawdown_penalty = base_augmentation # * lower_augmentation * upper_augmentation
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
    def mean_drawdown(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
        """
        if len(checkpoints) == 0:
            return 0

        # Compute the drawdown of the checkpoints
        drawdowns = [checkpoint.mdd for checkpoint in checkpoints]
        effective_drawdown = np.mean(drawdowns)
        final_drawdown = np.clip(effective_drawdown, 0, 1.0)

        return final_drawdown

    @staticmethod
    def max_drawdown(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
        """
        if len(checkpoints) == 0:
            return 0

        # Compute the drawdown of the checkpoints
        drawdowns = [checkpoint.mdd for checkpoint in checkpoints]
        effective_drawdown = np.min(drawdowns)
        final_drawdown = np.clip(effective_drawdown, 0, 1.0)

        return final_drawdown

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
    def is_beyond_max_drawdown(ledger_element: PerfLedger, restricted: bool=False):
        """Checks if the maximum drawdown percentage is surpassed"""
        if ledger_element is None:
            return False, 0

        if len(ledger_element.cps) == 0:
            return False, 0

        maximum_drawdown_percent = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE

        max_drawdown = LedgerUtils.recent_drawdown(ledger_element.cps, restricted=restricted)
        recorded_drawdown_percentage = LedgerUtils.drawdown_percentage(max_drawdown)

        # Drawdown is less than our maximum permitted drawdown
        max_drawdown_criteria = recorded_drawdown_percentage >= maximum_drawdown_percent
        recorded_drawdown_percentage = float(round(recorded_drawdown_percentage, 2))

        return max_drawdown_criteria, recorded_drawdown_percentage

    @staticmethod
    def risk_normalization(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Args:
            checkpoints: list[PerfCheckpoint] - the list of checkpoints
        """
        if len(checkpoints) == 0:
            return 0

        # recent_drawdown = LedgerUtils.recent_drawdown(checkpoints)
        approximate_drawdown = LedgerUtils.max_drawdown(checkpoints)

        # effective_drawdown = LedgerUtils.effective_drawdown(approximate_drawdown)

        drawdown_penalty = LedgerUtils.mdd_augmentation(approximate_drawdown)
        return drawdown_penalty

    @staticmethod
    def cumulative(ledger: dict[str, PerfLedger]) -> dict[str, dict]:
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
