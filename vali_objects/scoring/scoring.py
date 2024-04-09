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

import bittensor as bt

class Scoring:
    @staticmethod
    def transform_and_scale_results(
        filtered_results: list[tuple[str, list[float]]]
    ) -> list[tuple[str, float]]:
        """
        Args: filtered_results: list[tuple[str, list[float]]] - takes a list of dampened returns for each miner and computes their risk adjusted returns, weighing them against the competition. It then averages the weights assigned for each of these tasks and returns the final weights for each miner which are normalized to sum to 1.
        """

        if len(filtered_results) == 0:
            bt.logging.debug(f"No results to transform and scale, returning empty list")
            return []
        
        if len(filtered_results) == 1:
            bt.logging.debug(f"Only one miner, returning 1.0 for the solo miner weight")
            return [(filtered_results[0][0], 1.0)]

        scoring_functions = [
            Scoring.omega,
            Scoring.total_return,
            Scoring.sharpe_ratio
        ]

        scoring_function_weights = [
            0.4,
            0.4,
            0.2
        ]

        ## split into grace period miners and non-grace period miners
        grace_period_value = float(ValiConfig.SET_WEIGHT_MINER_GRACE_PERIOD_VALUE)
        grace_period_miners = []
        non_grace_period_miners = []
        debug_miners_in_grace_period = []
        debug_miners_no_returns = []
        for miner, returns in filtered_results:
            if len(returns) < ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
                debug_miners_in_grace_period.append(miner)
                if len(returns) == 0:
                    debug_miners_no_returns.append(miner)
                    continue
                grace_period_miners.append((miner, returns))
            else:
                non_grace_period_miners.append((miner, returns))

        miner_scores_list: list[list[tuple[str,float]]] = []
        debug_miners_not_reach_minimum_positions = []
        for scoring_function in scoring_functions:
            miner_scoring_function_scores = []
            for miner, returns in non_grace_period_miners:
                if len(returns) < int(ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS):
                    debug_miners_not_reach_minimum_positions.append(miner)
                    continue

                score = scoring_function(returns)
                miner_scoring_function_scores.append((miner, score))
            
            miner_scores_list.append(miner_scoring_function_scores)

        # bt.logging.info(f"Grace period miners skipped due to no returns: {debug_miners_no_returns}. "
        #                 f"Miners in grace period: {debug_miners_in_grace_period}"
        #                 f"Miners not reaching minimum positions for scoring: {debug_miners_not_reach_minimum_positions}")

        # Combine the scores from the different scoring functions
        weighted_scores: list[list[str, float]] = []
        for miner_score_list in miner_scores_list:
            weighted_scores.append(Scoring.weigh_miner_scores(miner_score_list))
        
        # Combine the weighted scores from the different scoring functions
        combined_scores: dict[str, float] = {}
        for c,weighted_score in enumerate(weighted_scores):
            for miner, score in weighted_score:
                if miner not in combined_scores:
                    combined_scores[miner] = 0

                combined_scores[miner] += score * scoring_function_weights[c]
        # this finishes the non-grace period miners
        normalized_scores = Scoring.normalize_scores(combined_scores)
        grace_period_miner_ids = [ x[0] for x in grace_period_miners ]
        grace_period_miner_scores = [ float(grace_period_value) for _ in grace_period_miners ]

        grace_period_scores = dict(
            zip(grace_period_miner_ids, grace_period_miner_scores)
        )

        total_score_dict = {
            **normalized_scores, 
            **grace_period_scores
        }

        total_scores = list(total_score_dict.items())
        total_scores = sorted(total_scores, key=lambda x: x[1], reverse=True)

        # bt.logging.info(f"Max miner weight for round: {max([x[1] for x in total_scores])}. "
        #                 f"Transformed results sum: {sum([x[1] for x in total_scores])}")

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
        if len(returns) == 0:
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
        ## Assign weights to the returns based on their relative position
        if len(returns) == 0:
            bt.logging.debug(f"No returns to score, returning empty list")
            return []
        if len(returns) == 1:
            bt.logging.info(f"Only one miner, returning 1.0 for the solo miner weight")
            return [(returns[0][0], 1.0)]

        # Sort the returns in descending order
        sorted_returns = sorted(returns, key=lambda x: x[1], reverse=True)

        n_miners = len(sorted_returns)
        miner_names = [x[0] for x in sorted_returns]
        decayed_returns = Scoring.exponential_decay_returns(n_miners)

        # Create a dictionary to map miner names to their decayed returns
        miner_decay_returns_dict = dict(zip(miner_names, decayed_returns))

        # Assign the decayed returns to the sorted miner names
        weighted_returns = [(miner, miner_decay_returns_dict[miner]) for miner, _ in sorted_returns]

        return weighted_returns

    @staticmethod
    def update_weights_remove_deregistrations(miner_uids: List[str]):
        vweights = ValiUtils.get_vali_weights_json()
        for miner_uid in miner_uids:
            if miner_uid in vweights:
                del vweights[miner_uid]
        ValiUtils.set_vali_weights_bkp(vweights)
