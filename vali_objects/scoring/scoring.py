# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import math
from typing import List, Tuple, Dict

import numpy as np
from scipy.stats import percentileofscore

from vali_config import ValiConfig
from vali_objects.exceptions.incorrect_prediction_size_error import IncorrectPredictionSizeError
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedger
from vali_objects.utils.position_manager import PositionManager
from time_util.time_util import TimeUtil
from vali_objects.utils.position_utils import PositionUtils

import bittensor as bt

from pydantic import BaseModel, validator, field_validator


class ScoringUnit(BaseModel):
    gains: list[float]
    losses: list[float]
    n_updates: list[int]
    open_ms: list[int]
    mdd: list[float]

    @field_validator('gains', 'n_updates', 'open_ms', mode='before')
    def check_non_negative(cls, v):
        if any(x < 0 for x in (v or [])):  # Simplified check
            raise ValueError("All values must be non-negative")
        return v

    @field_validator('losses', mode='before')
    def check_non_positive(cls, v):
        if any(x > 0 for x in (v or [])):  # Simplified check
            raise ValueError("All values must be non-positive")
        return v

    @classmethod
    def from_perf_ledger(cls, perf_ledger: PerfLedger):
        # Extract the required fields from the list of PerfCheckpoint in the PerfLedger
        gains = []
        losses = []
        n_updates = []
        open_ms = []
        mdd = []

        for cp in perf_ledger.cps:
            gains.append(cp.gain)
            losses.append(cp.loss)
            n_updates.append(cp.n_updates)
            open_ms.append(cp.open_ms)
            mdd.append(cp.mdd)

        return cls(gains=gains, losses=losses, n_updates=n_updates, open_ms=open_ms, mdd=mdd)
    
class Scoring:
    @staticmethod
    def compute_results_checkpoint(
        ledger_dict: Dict[str, PerfLedger],
        evaluation_time_ms: int = None,
        verbose=True
    ) -> List[Tuple[str, float]]:
        if len(ledger_dict) == 0:
            bt.logging.debug("No results to compute, returning empty list")
            return []

        if len(ledger_dict) == 1:
            miner = list(ledger_dict.keys())[0]
            if verbose:
                bt.logging.info(f"Only one miner: {miner}, returning 1.0 for the solo miner weight")
            return [(miner, 1.0)]
        
        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()
        
        return_decay_coefficient_short = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS_SHORT
        return_decay_coefficient_long = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS_LONG
        risk_adjusted_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RISKMETRIC

        # Compute miner penalties
        miner_penalties = Scoring.miner_penalties(ledger_dict)

        # Augmented returns ledgers
        returns_ledger_short = PositionManager.augment_perf_ledger(
            ledger_dict,
            evaluation_time_ms=evaluation_time_ms,
            time_decay_coefficient=return_decay_coefficient_short,
        )

        returns_ledger_long = PositionManager.augment_perf_ledger(
            ledger_dict,
            evaluation_time_ms=evaluation_time_ms,
            time_decay_coefficient=return_decay_coefficient_long,
        )

        risk_adjusted_ledger = PositionManager.augment_perf_ledger(
            ledger_dict,
            evaluation_time_ms=evaluation_time_ms,
            time_decay_coefficient=risk_adjusted_decay_coefficient,
        )

        scoring_config = {
            'return_cps_short': {
                'function': Scoring.return_cps,
                'weight': ValiConfig.SCORING_RETURN_CPS_SHORT_WEIGHT,
                'ledger': returns_ledger_short,
            },
            'return_cps_long': {
                'function': Scoring.return_cps,
                'weight': ValiConfig.SCORING_RETURN_CPS_LONG_WEIGHT,
                'ledger': returns_ledger_long,
            },
            'omega_cps': {
                'function': Scoring.omega_cps,
                'weight': ValiConfig.SCORING_OMEGA_CPS_WEIGHT,
                'ledger': risk_adjusted_ledger,
            },
        }

        combined_scores = {}

        for config in scoring_config.values():
            miner_scores = []
            for miner, minerledger in config['ledger'].items():
                scoringunit = ScoringUnit.from_perf_ledger(minerledger)
                score = config['function'](scoringunit=scoringunit)
                miner_scores.append((miner, score))
            
            weighted_scores = Scoring.miner_scores_percentiles(miner_scores)
            
            for miner, score in weighted_scores:
                if miner not in combined_scores:
                    combined_scores[miner] = 1
                combined_scores[miner] *= config['weight'] * score + (1 - config['weight'])

        # ## Force good performance of all error metrics
        combined_weighed = Scoring.weigh_miner_scores(list(combined_scores.items()))
        combined_scores = dict(combined_weighed)

        ## Apply the penalties to each miner
        combined_penalized_scores = { miner: score * miner_penalties.get(miner,0) for miner, score in combined_scores.items() }

        ## Normalize the scores
        normalized_scores = Scoring.normalize_scores(combined_penalized_scores)
        return sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def miner_penalties(ledger_dict: dict[str, PerfLedger]) -> dict[str, float]:
        # Compute miner penalties
        miner_penalties = {}
        for miner, perfledger in ledger_dict.items():
            ledgercps = perfledger.cps

            consistency_penalty = PositionUtils.compute_consistency_penalty_cps(ledgercps)
            drawdown_penalty = PositionUtils.compute_drawdown_penalty_cps(ledgercps)
            miner_penalties[miner] = drawdown_penalty * consistency_penalty

        return miner_penalties
    
    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """
        Args: scores: dict[str, float] - the scores of the miner returns
        """
        # bt.logging.info(f"Normalizing scores: {scores}")
        if len(scores) == 0:
            bt.logging.info(f"No scores to normalize, returning empty list")
            return {}
        
        sum_scores = sum(scores.values())
        if sum_scores == 0:
            bt.logging.info(f"sum_scores is 0, returning empty list")
            return {}
        
        normalized_scores = {
            miner: (score / sum_scores) for miner, score in scores.items()
        }
        # normalized_scores = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
        return normalized_scores
    
    @staticmethod
    def return_cps(scoringunit: ScoringUnit) -> float:
        """
        Args:
            scoringunit: ScoringUnit - the scoring unit for the miner
        """
        gains = scoringunit.gains
        losses = scoringunit.losses

        if len(gains) == 0 or len(losses) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to a bad return (bottom of the list)
            return -1

        total_gain = sum(gains)
        total_loss = sum(losses)
        total_return = total_gain + total_loss

        return total_return
    
    @staticmethod
    def omega_cps(scoringunit: ScoringUnit) -> float:
        """
        Args:
            scoringunit: ScoringUnit - the scoring unit for the miner
        """
        gains = scoringunit.gains
        losses = scoringunit.losses

        if len(gains) == 0 or len(losses) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to 0 weight (bottom of the list)
            return 0

        omega_minimum_denominator = ValiConfig.OMEGA_MINIMUM_DENOMINATOR

        sum_above = sum(gains)
        sum_below = sum(losses)

        sum_below = max(abs(sum_below), omega_minimum_denominator)
        return sum_above / sum_below
    
    @staticmethod
    def inverted_sortino_cps(scoringunit: ScoringUnit) -> float:
        """
        Args:
            scoringunit: ScoringUnit - the scoring unit for the miner
        """
        gains = scoringunit.gains
        losses = scoringunit.losses
        open_ms = scoringunit.open_ms

        if len(gains) == 0 or len(losses) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to 0 weight (bottom of the list)
            return 0
        
        total_loss = sum(losses)
        total_loss = np.clip(total_loss, a_min=None, a_max=0)

        total_position_active_time = sum(open_ms)

        if total_position_active_time == 0:
            return -1 / ValiConfig.SORTINO_MIN_DENOMINATOR # this will be quite large
                
        return total_loss / total_position_active_time
    
    @staticmethod
    def checkpoint_volume_threshold_count(scoringunit: ScoringUnit) -> int:
        """
        Args:
            scoringunit: ScoringUnit - the scoring unit for the miner
        """
        gains = scoringunit.gains
        losses = scoringunit.losses
        if len(gains) == 0 or len(losses) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to 0 weight (bottom of the list)
            return 0
        
        checkpoint_volume_threshold = ValiConfig.CHECKPOINT_VOLUME_THRESHOLD
        volume_arr = np.array(gains) + np.abs(np.array(losses))

        return int(np.sum(volume_arr >= checkpoint_volume_threshold))

    @staticmethod
    def omega(returns: list[float]) -> float:
        """
        Args: returns: list[float] - the logged returns for each miner
        """
        if len(returns) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to 0 weight (bottom of the list)
            return 0

        threshold = ValiConfig.OMEGA_LOG_RATIO_THRESHOLD
        omega_minimum_denominator = ValiConfig.OMEGA_MINIMUM_DENOMINATOR

        # need to convert this to percentage based returns
        sum_above = 0
        sum_below = 0

        threshold_return = [ return_ - threshold for return_ in returns ]
        for return_ in threshold_return:
            if return_ > 0:
                sum_above += return_
            else:
                sum_below += return_

        sum_below = max(abs(sum_below), omega_minimum_denominator)
        return sum_above / sum_below
    
    @staticmethod
    def mad_variation(returns: list[float]) -> float:
        """
        Args: returns: list[float] - the variance of the miner returns
        """
        if len(returns) == 0:
            return 0

        median = np.median(returns)

        if median == 0:
            median = ValiConfig.MIN_MEDIAN

        mad = np.mean(np.abs(returns - median))
        mrad = mad / median
        
        return abs(mrad)
    
    @staticmethod
    def total_return(returns: list[float]) -> float:
        """
        Args: returns: list[float] - the total return of the miner returns
        """
        ## again, won't happen but in case there are no returns
        if len(returns) == 0:
            return 0
        
        return np.exp(np.sum(returns))
    
    @staticmethod
    def probabilistic_sharpe_ratio(returns: list[float]) -> float:
        """
        Calculates the Probabilistic Sharpe Ratio (PSR) for a list of returns using the Adjusted Sharpe Ratio (ASR)
        and the normal CDF approximation.

        Args:
            returns (list[float]): List of returns.
            threshold (float): Threshold return (default is 0).

        Returns:
            float: Probabilistic Sharpe Ratio (PSR).
        """
        if len(returns) == 0:
            return 0
        
        sharpe_ratio = Scoring.sharpe_ratio(returns)

        # skewness, kurtosis = Scoring.calculate_moments(returns)

        # Calculate the Adjusted Sharpe Ratio (ASR) based on '“Risk and Risk Aversion”, in C. Alexander and E. Sheedy, eds.: The Professional Risk Managers’Handbook, PRMIA Publications'

        # adjusted_sharpe_ratio = sharpe_ratio * (1 + (skewness * sharpe_ratio / 6) - (sharpe_ratio**2 * (kurtosis - 3) / 24))

        psr = Scoring.norm_cdf(sharpe_ratio)
        return psr
    
    @staticmethod
    def sharpe_ratio(returns: list[float]) -> float:
        """
        Calculates the Sharpe Ratio for a list of returns.

        Args:
            returns (list[float]): List of returns.

        Returns:
            float: Sharpe Ratio.
        """
        if len(returns) <= 1:
            return 0
        
        returns = np.array(returns)
        mean_return = np.mean(returns)
        std_dev = np.std(returns)

        if std_dev == 0:
            std_dev = ValiConfig.PROBABILISTIC_SHARPE_RATIO_MIN_STD_DEV

        threshold = ValiConfig.PROBABILISTIC_LOG_SHARPE_RATIO_THRESHOLD
        log_sharpe = (mean_return - threshold) / std_dev

        return np.exp(log_sharpe)
    
    @staticmethod # Calculate the Probabilistic Sharpe Ratio (PSR) using the normal CDF approximation
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))
    
    @staticmethod
    def calculate_moments(returns):
        """
        Calculates the skewness and kurtosis of the returns distribution.

        Args:
            returns (list[float]): List of returns.

        Returns:
            tuple: Skewness and kurtosis of the returns distribution.
        """
        mean = np.mean(returns)
        std_dev = np.std(returns)

        third_moment = np.mean((returns - mean)**3)
        fourth_moment = np.mean((returns - mean)**4)

        skewness = third_moment / (std_dev**3)
        kurtosis = fourth_moment / (std_dev**4)

        return skewness, kurtosis
    
    @staticmethod
    def exponential_decay_returns(scale: int) -> np.ndarray:
        """
        Args: scale: int - the number of miners
        """
        top_miner_benefit = ValiConfig.TOP_MINER_BENEFIT
        top_miner_percent = ValiConfig.TOP_MINER_PERCENT

        top_miner_benefit = np.clip(top_miner_benefit, a_min=0, a_max=0.99999999)
        top_miner_percent = np.clip(top_miner_percent, a_min=0.00000001, a_max=1)
        scale = np.clip(scale, a_min=1, a_max=None)
        if scale == 1:
            # base case, if there is only one miner
            return np.array([1])

        k = -np.log(1 - top_miner_benefit) / (top_miner_percent * scale)
        xdecay = np.linspace(0, scale-1, scale)
        decayed_returns = np.exp((-k) * xdecay)

        # Normalize the decayed_returns so that they sum up to 1
        return decayed_returns / np.sum(decayed_returns)
    
    @staticmethod
    def miner_scores_percentiles(miner_scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Args: miner_scores: list[tuple[str, float]] - the scores of the miners
        """
        if len(miner_scores) == 0:
            bt.logging.debug("No miner scores to compute percentiles, returning empty list")
            return []
        
        if len(miner_scores) == 1:
            miner, score = miner_scores[0]
            bt.logging.info(f"Only one miner: {miner}, returning 1.0 for the solo miner weight")
            return [(miner, 1.0)]
        
        minernames = []
        scores = []

        for miner, score in miner_scores:
            minernames.append(miner)
            scores.append(score)

        percentiles = percentileofscore( scores, scores ) / 100
        miner_percentiles = list(zip(minernames, percentiles))
        
        return miner_percentiles


    @staticmethod
    def weigh_miner_scores(returns: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Assign weights to the returns based on their relative position.
        """
        if not returns:
            bt.logging.debug("No returns to score, returning empty list")
            return []

        if len(returns) == 1:
            bt.logging.info("Only one miner, returning 1.0 for the solo miner weight")
            return [(returns[0][0], 1.0)]

        sorted_returns = sorted(returns, key=lambda x: x[1], reverse=True)
        n_miners = len(sorted_returns)
        decayed_returns = Scoring.exponential_decay_returns(n_miners)

        weighted_returns = [(miner, decayed_returns[i]) for i, (miner, _) in enumerate(sorted_returns)]
        return weighted_returns

    @staticmethod
    def update_weights_remove_deregistrations(miner_uids: List[str]):
        vweights = ValiUtils.get_vali_weights_json()
        for miner_uid in miner_uids:
            if miner_uid in vweights:
                del vweights[miner_uid]
        ValiUtils.set_vali_weights_bkp(vweights)
