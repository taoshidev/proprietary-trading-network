# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import math
from typing import List, Tuple, Dict

import numpy as np

from vali_config import ValiConfig
from vali_objects.exceptions.incorrect_prediction_size_error import IncorrectPredictionSizeError
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedger

import bittensor as bt

class Scoring:
    @staticmethod
    def compute_results_checkpoint(
        ledger_dict: dict[str, PerfLedger]
    ) -> list[tuple[str, list[float]]]:
        """
        Args: ledger: PerfLedger - the ledger of the miner returns
        """
        if len(ledger_dict) == 0:
            bt.logging.debug(f"No results to compute, returning empty list")
            return []
        
        if len(ledger_dict) == 1:
            miner = list(ledger_dict.keys())[0]
            bt.logging.debug(f"Only one miner: {miner}, returning 1.0 for the solo miner weight")
            return [(miner, 1.0)]
        
        scoring_functions = [
            Scoring.return_cps,
            Scoring.omega_cps,
            Scoring.inverted_sortino_cps,
        ]

        scoring_function_weights = [
            0.90,
            0.75,
            0.60
        ]
        
        miner_scores_list: list[list[tuple[str, float]]] = [[] for _ in scoring_functions]

        for scoring_index, scoring_function in enumerate(scoring_functions):
            for miner, minerledger in ledger_dict.items():
                gains = [cp.gain for cp in minerledger.cps]
                losses = [cp.loss for cp in minerledger.cps]
                n_updates = [cp.n_updates for cp in minerledger.cps]
                open_ms = [cp.open_ms for cp in minerledger.cps]

                score = scoring_function(gains=gains, losses=losses, n_updates=n_updates, open_ms=open_ms)
                miner_scores_list[scoring_index].append((miner, score))

        # Combine the scores from the different scoring functions
        weighted_scores: list[list[str, float]] = []

        for miner_score_list in miner_scores_list:
            weighted_scores.append(Scoring.weigh_miner_scores(miner_score_list))

        # Combine the weighted scores from the different scoring functions
        combined_scores: dict[str, float] = {}
        for c,weighted_score in enumerate(weighted_scores):
            for miner, score in weighted_score:
                if miner not in combined_scores:
                    combined_scores[miner] = 1

                combined_scores[miner] *= scoring_function_weights[c] * score + (1 - scoring_function_weights[c])

        ## Force good performance of all error metrics
        combined_weighed = Scoring.weigh_miner_scores(list(combined_scores.items()))
        combined_scoresdict = dict(combined_weighed)

        # this finishes the non-grace period miners
        normalized_scores = Scoring.normalize_scores(combined_scoresdict)

        total_scores = list(normalized_scores.items())
        total_scores = sorted(total_scores, key=lambda x: x[1], reverse=True)

        return total_scores
    
    
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
    def return_cps(
        gains: list[float],
        losses: list[float],
        n_updates: list[int],
        open_ms: list[int]
    ) -> float:
        """
        Args:
            gains: list[float] - the gains for each miner
            losses: list[float] - the losses for each miner
            n_updates: list[int] - the number of updates for each miner
            open_ms: list[int] - the open time for each miner
        """
        if len(gains) == 0 or len(losses) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to a bad return (bottom of the list)
            return -1

        total_gain = sum(gains)
        total_loss = sum(losses)
        total_return = total_gain + total_loss

        return total_return
    
    @staticmethod
    def omega_cps(
        gains: list[float], 
        losses: list[float],
        n_updates: list[int],
        open_ms: list[int]
    ) -> float:
        """
        Args:
            gains: list[float] - the gains for each miner
            losses: list[float] - the losses for each miner
            n_updates: list[int] - the number of updates for each miner
            open_ms: list[int] - the open time for each miner
        """
        if len(gains) == 0 or len(losses) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to 0 weight (bottom of the list)
            return 0

        omega_minimum_denominator = ValiConfig.OMEGA_MINIMUM_DENOMINATOR

        sum_above = sum(gains)
        sum_below = sum(losses)

        sum_below = max(abs(sum_below), omega_minimum_denominator)
        return sum_above / sum_below
    
    @staticmethod
    def inverted_sortino_cps(
        gains: list[float], 
        losses: list[float],
        n_updates: list[int],
        open_ms: list[int]
    ) -> float:
        """
        Args:
            gains: list[float] - the gains for each miner
            losses: list[float] - the losses for each miner
            n_updates: list[int] - the number of updates for each miner
            open_ms: list[int] - the open time for each miner
        """
        if len(gains) == 0 or len(losses) == 0:
            # won't happen because we need a minimum number of trades, but would kick them to 0 weight (bottom of the list)
            return 0
        
        total_loss = sum(losses)
        total_position_active_time = sum(open_ms)

        if total_position_active_time == 0:
            return 0
        
        if total_loss == 0:
            return 1 / ValiConfig.SORTINO_MIN_DENOMINATOR
        
        typical_absolute_loss = abs(total_loss / total_position_active_time)
        return -typical_absolute_loss

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
