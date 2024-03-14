# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

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
    def filter_results(return_per_netuid:  dict[str, float]) -> list[tuple[str, float]]:
        if len(return_per_netuid) == 0:
            bt.logging.info(f"no returns to filter, returning empty lists")
            return []
        
        if len(return_per_netuid) == 1:
            bt.logging.info(f"Only one miner, returning 1.0 for the solo miner weight")
            return [ (list(return_per_netuid.keys())[0], 1.0) ]

        mean = np.mean(list(return_per_netuid.values()))
        std_dev = np.std(list(return_per_netuid.values()))

        lower_bound = mean - 3 * std_dev
        bt.logging.info(f"returns lower bound: [{lower_bound}]")

        if lower_bound < 0:
            lower_bound = 0

        return [ (k, v) for k, v in return_per_netuid.items() if lower_bound < v ]

    @staticmethod
    def transform_and_scale_results(filtered_results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        # Exponential decay of the scores
        transformed = Scoring.weigh_miner_scores(filtered_results)
        bt.logging.info(f"Max miner weight for round: {max([x[1] for x in transformed])}")
        bt.logging.info(f"Transformed results sum: {sum([x[1] for x in transformed])}")
        return transformed
        
    @staticmethod
    def exponential_decay_returns(scale: int) -> np.ndarray:
        """
        Args:
            scale: int - the number of miners
            a_percent: float - % benefit to the top % miners
            b_percent: float - top % of miners
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
            return [returns[0][0], 1.0]

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
