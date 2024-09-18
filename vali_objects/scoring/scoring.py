# developer: trdougherty

import math
from typing import List, Tuple
from vali_objects.position import Position

import numpy as np
from scipy.stats import percentileofscore

from vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerData
from time_util.time_util import TimeUtil
from vali_objects.utils.position_filtering import PositionFiltering
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.ledger_utils import LedgerUtils

import bittensor as bt

class Scoring:
    @staticmethod
    def compute_results_checkpoint(
            ledger_dict: dict[str, PerfLedgerData],
            full_positions: dict[str, list[Position]],
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

        filtered_positions = PositionFiltering.filter(
            full_positions,
            evaluation_time_ms=evaluation_time_ms
        )

        filtered_recent_positions = PositionFiltering.filter_recent(
            full_positions,
            evaluation_time_ms=evaluation_time_ms
        )

        # Compute miner penalties
        miner_penalties = Scoring.miner_penalties(filtered_positions, ledger_dict)

        # Miners with full penalty
        full_penalty_miner_scores: list[tuple[str, float]] = [
            (miner, 0) for miner, penalty in miner_penalties.items() if penalty == 0
        ]
        full_penalty_miners = set([x[0] for x in full_penalty_miner_scores])

        scoring_config = {
            'return_long': {
                'function': Scoring.risk_adjusted_return,
                'weight': ValiConfig.SCORING_RETURN_LONG_LOOKBACK_WEIGHT,
                'positions': filtered_positions
            },
            'return_short': {
                'function': Scoring.risk_adjusted_return,
                'weight': ValiConfig.SCORING_RETURN_SHORT_LOOKBACK_WEIGHT,
                'positions': filtered_recent_positions
            },
            'sharpe_ratio': {
                'function': Scoring.sharpe,
                'weight': ValiConfig.SCORING_SHARPE_WEIGHT,
                'positions': filtered_positions
            },
            'omega': {
                'function': Scoring.omega,
                'weight': ValiConfig.SCORING_OMEGA_WEIGHT,
                'positions': filtered_positions
            },
        }

        combined_scores = {}

        for config_name, config in scoring_config.items():
            miner_scores = []
            for miner, positions in config['positions'].items():
                # Get the miner ledger
                ledger = ledger_dict.get(miner, PerfLedgerData())

                # Check if the miner has full penalty - if not include them in the scoring competition
                if miner in full_penalty_miners:
                    continue

                score = config['function'](
                    positions=positions,
                    ledger=ledger
                )

                penalized_score = score * miner_penalties.get(miner, 0)
                miner_scores.append((miner, penalized_score))

            weighted_scores = Scoring.miner_scores_percentiles(miner_scores)
            for miner, score in weighted_scores:
                if miner not in combined_scores:
                    combined_scores[miner] = 1
                combined_scores[miner] *= config['weight'] * score + (1 - config['weight'])

        # Force good performance of all error metrics
        combined_weighed = Scoring.softmax_scores(list(combined_scores.items())) + full_penalty_miner_scores
        combined_scores = dict(combined_weighed)

        # Normalize the scores
        normalized_scores = Scoring.normalize_scores(combined_scores)
        return sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def miner_penalties(
            hotkey_positions: dict[str, list[Position]],
            ledger_dict: dict[str, PerfLedgerData]
    ) -> dict[str, float]:
        # Compute miner penalties
        miner_penalties = {}

        for miner, ledger in ledger_dict.items():
            ledger_checkpoints = ledger.cps
            positions = hotkey_positions.get(miner, [])

            # Positional Consistency
            positional_return_time_consistency = PositionPenalties.time_consistency_penalty(positions)
            positional_consistency = PositionPenalties.returns_ratio_penalty(positions)

            # Ledger Consistency
            daily_consistency = LedgerUtils.daily_consistency_penalty(ledger_checkpoints)
            biweekly_consistency = LedgerUtils.biweekly_consistency_penalty(ledger_checkpoints)
            drawdown_threshold_penalty = LedgerUtils.max_drawdown_threshold_penalty(ledger_checkpoints)

            # Combine penalties
            miner_penalties[miner] = (
                    drawdown_threshold_penalty *
                    positional_return_time_consistency *
                    positional_consistency *
                    daily_consistency *
                    biweekly_consistency
            )

        return miner_penalties

    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """
        Args: scores: dict[str, float] - the scores of the miner returns
        """
        # bt.logging.info(f"Normalizing scores: {scores}")
        if len(scores) == 0:
            bt.logging.info("No scores to normalize, returning empty list")
            return {}

        sum_scores = sum(scores.values())
        if sum_scores == 0:
            bt.logging.info("sum_scores is 0, returning empty list")
            return {}

        normalized_scores = {
            miner: (score / sum_scores) for miner, score in scores.items()
        }
        # normalized_scores = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
        return normalized_scores

    @staticmethod
    def base_return(positions: list[Position]) -> float:
        """
        Args:
            positions: list of positions from the miner
        """
        if len(positions) == 0:
            return 0.0

        positional_returns = [math.log(position.return_at_close) for position in positions]

        aggregate_return = 0.0
        for positional_return in positional_returns:
            aggregate_return += positional_return

        return aggregate_return

    @staticmethod
    def risk_adjusted_return(positions: list[Position], ledger: PerfLedgerData) -> float:
        """
        Args:
            positions: list of positions from the miner
            ledger: the ledger of the miner
        """
        # Positional Component
        if len(positions) == 0:
            return 0.0

        base_return = Scoring.base_return(positions)
        risk_normalization_factor = LedgerUtils.risk_normalization(ledger.cps)

        return base_return * risk_normalization_factor

    @staticmethod
    def sharpe(positions: list[Position], ledger: PerfLedgerData) -> float:
        """
        Args:
            positions: list of positions from the miner
            ledger: the ledger of the miner
        """
        if len(positions) == 0:
            return 0.0

        # Hyperparameter
        min_std_dev = ValiConfig.SHARPE_STDDEV_MINIMUM

        # Return at close should already accommodate the risk-free rate as a cost of carry
        positional_log_returns = [math.log(position.return_at_close) for position in positions]

        # Sharpe ratio is calculated as the mean of the returns divided by the standard deviation of the returns
        mean_return = np.mean(positional_log_returns)
        std_dev = max(np.std(positional_log_returns), min_std_dev)

        if std_dev == 0:
            return 0.0

        return mean_return / std_dev

    @staticmethod
    def omega(positions: list[Position], ledger: PerfLedgerData) -> float:
        """
        Args:
            positions: list of positions from the miner
            ledger: the ledger of the miner
        """
        if len(positions) == 0:
            return 0.0

        # Return at close should already accommodate the risk-free rate as a cost of carry
        positional_log_returns = [math.log(position.return_at_close) for position in positions]

        positive_sum = 0
        negative_sum = 0

        for log_return in positional_log_returns:
            if log_return > 0:
                positive_sum += log_return
            else:
                negative_sum += log_return

        numerator = positive_sum
        denominator = max(abs(negative_sum), ValiConfig.OMEGA_LOSS_MINIMUM)

        return numerator / denominator

    @staticmethod
    def softmax_scores(returns: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Assign weights to the returns based on their relative position and apply softmax with a temperature parameter.
    
        The softmax function is used to convert the scores into probabilities that sum to 1.
        Subtracting the max value from the scores before exponentiation improves numerical stability.
    
        Parameters:
        returns (list[tuple[str, float]]): List of tuples with miner names and their scores.
        temperature (float): Temperature parameter to control the sharpness of the softmax distribution. Default is 1.0.
    
        Returns:
        list[tuple[str, float]]: List of tuples with miner names and their softmax weights.
        """
        epsilon = ValiConfig.EPSILON
        temperature = ValiConfig.SOFTMAX_TEMPERATURE
    
        if not returns:
            bt.debug("No returns to score, returning empty list")
            return []
    
        if len(returns) == 1:
            bt.logging.info("Only one miner, returning 1.0 for the solo miner weight")
            return [(returns[0][0], 1.0)]
    
        # Extract scores and apply softmax with temperature
        scores = [score for _, score in returns]
        max_score = np.max(scores)
        exp_scores = np.exp((scores - max_score) / temperature)
        softmax_scores = exp_scores / max(np.sum(exp_scores), epsilon)
    
        # Combine miners with their respective softmax scores
        weighted_returns = [(returns[i][0], softmax_scores[i]) for i in range(len(returns))]
    
        return weighted_returns

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
        xdecay = np.linspace(0, scale - 1, scale)
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

        percentiles = percentileofscore(scores, scores) / 100
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
