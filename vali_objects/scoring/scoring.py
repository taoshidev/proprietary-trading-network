# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import math
from typing import List, Tuple, Dict

import numpy as np
from scipy.stats import norm

from vali_config import ValiConfig
from vali_objects.exceptions.incorrect_prediction_size_error import IncorrectPredictionSizeError
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

class Scoring:
    @staticmethod
    def filter_results(return_per_netuid:  dict[str, float]) -> list[tuple[str, float]]:
        if len(return_per_netuid) == 0:
            bt.logging.info(f"no returns to filter, returning empty lists")
            return []
        
        # if len(return_per_netuid) == 1:
        #     bt.logging.info(f"Only one miner, returning 1.0 for the solo miner weight")
        #     return [ (list(return_per_netuid.keys())[0], [1.0]) ]

        # mean = np.mean(list(return_per_netuid.values()))
        # std_dev = np.std(list(return_per_netuid.values()))

        # lower_bound = mean - 3 * std_dev
        # bt.logging.info(f"returns lower bound: [{lower_bound}]")

        # if lower_bound < 0:
        #     lower_bound = 0

        return [ 
            (k, v) for k, v in return_per_netuid.items() 
            # if lower_bound < v 
        ]

    @staticmethod
    def transform_and_scale_results(
        filtered_results: list[tuple[str, list[float]]]
    ) -> list[tuple[str, float]]:
        """
        Args: filtered_results: list[tuple[str, list[float]]] - takes a list of dampened returns for each miner and computes their risk adjusted returns, weighing them against the competition. It then averages the weights assigned for each of these tasks and returns the final weights for each miner which are normalized to sum to 1.
        """

        scoring_functions = [
            Scoring.omega,
            Scoring.total_return,
        ]

        scoring_function_weights = [
            0.8,
            0.2
        ]

        miner_scores_list: list[list[tuple[str,float]]] = []
        for scoring_function in scoring_functions:
            miner_scoring_function_scores = []
            for miner, returns in filtered_results:
                if len(returns) == 0:
                    bt.logging.info(f"no returns for miner [{miner}]")
                    continue

                score = scoring_function(returns)
                miner_scoring_function_scores.append((miner, score))
            
            miner_scores_list.append(miner_scoring_function_scores)

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

        # Normalize the combined scores so that they sum up to 1
        sum_scores = sum(combined_scores.values())
        if sum_scores == 0:
            bt.logging.info(f"sum_scores is 0, returning empty list")
            return []
        
        normalized_scores = {miner: score / sum_scores for miner, score in combined_scores.items()}
        bt.logging.info(f"Max miner weight for round: {max([x[1] for x in normalized_scores])}")
        bt.logging.info(f"Transformed results sum: {sum([x[1] for x in normalized_scores])}")
        return normalized_scores    

    @staticmethod
    def omega(returns: list[float]) -> float:
        """
        Args: returns: list[float] - the omega ratio of the miner returns
        """
        if len(returns) == 0:
            return 0

        threshold = 1 + ValiConfig.LOOKBACK_RANGE_DAYS_RISK_FREE_RATE
        omega_minimum_denominator = ValiConfig.OMEGA_MINIMUM_DENOMINATOR

        # need to convert this to percentage based returns
        returns_above_threshold = []
        returns_below_threshold = []

        for return_ in returns:
            if return_ >= threshold:
                returns_above_threshold.append(return_)
            else:
                returns_below_threshold.append(return_)

        sum_above_threshold = sum(returns_above_threshold)
        sum_below_threshold = sum(returns_below_threshold)

        if sum_below_threshold == 0:
            # Modified Omega Ratio: Add a small constant to the denominator
            sum_below_threshold = omega_minimum_denominator

        return sum_above_threshold / abs(sum_below_threshold)
    
    @staticmethod
    def total_return(returns: list[float]) -> float:
        """
        Args: returns: list[float] - the total return of the miner returns
        """
        return np.prod(returns)
    
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
        returns = np.array(returns)
        mean_return = np.mean(returns)
        std_dev = np.std(returns)

        if std_dev == 0:
            std_dev = ValiConfig.PROBABILISTIC_SHARPE_RATIO_MIN_STD_DEV

        threshold = 1 + ValiConfig.LOOKBACK_RANGE_DAYS_RISK_FREE_RATE
        sharpe_ratio = (mean_return - threshold) / std_dev

        return sharpe_ratio
    
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
    def calculate_directional_accuracy(predictions: np, actual: np) -> float:
        pred_len = len(predictions)

        pred_dir = np.sign([predictions[i] - predictions[i - 1] for i in range(1, pred_len)])
        actual_dir = np.sign([actual[i] - actual[i - 1] for i in range(1, pred_len)])

        correct_directions = 0
        for i in range(0, pred_len-1):
            correct_directions += actual_dir[i] == pred_dir[i]

        return correct_directions / (pred_len-1)

    @staticmethod
    def simple_scale_scores(scores: Dict[str, float]) -> Dict[str, float]:
        if len(scores) <= 1:
            raise MinResponsesException("not enough responses")
        score_values = [score for miner_uid, score in scores.items()]
        min_score = min(score_values)
        max_score = max(score_values)

        return {miner_uid:  1 - ((score - min_score) / (max_score - min_score)) for miner_uid, score in scores.items()}

    @staticmethod
    def update_weights_remove_deregistrations(miner_uids: List[str]):
        vweights = ValiUtils.get_vali_weights_json()
        for miner_uid in miner_uids:
            if miner_uid in vweights:
                del vweights[miner_uid]
        ValiUtils.set_vali_weights_bkp(vweights)

    @staticmethod
    def basic_ema(current_value, previous_ema, length=48):
        alpha = 2 / (length + 1)
        ema = alpha * current_value + (1 - alpha) * previous_ema
        return ema
