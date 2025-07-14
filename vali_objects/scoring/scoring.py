# developer: trdougherty

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import List, Tuple, Callable
from vali_objects.position import Position
import copy
from collections import defaultdict

import numpy as np
from scipy.stats import percentileofscore

from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
from time_util.time_util import TimeUtil
from vali_objects.utils.position_filtering import PositionFiltering
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.metrics import Metrics
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.vali_config import TradePair, TradePairCategory, TradePairSubcategory
import bittensor as bt


class PenaltyInputType(Enum):
    LEDGER = auto()
    POSITIONS = auto()
    PSEUDO_POSITIONS = auto()


@dataclass
class PenaltyConfig:
    function: Callable
    input_type: PenaltyInputType


class Scoring:
    # Set the scoring configuration
    scoring_config = {
        'calmar': {
            'function': Metrics.calmar,
            'weight': ValiConfig.SCORING_CALMAR_WEIGHT
        },
        'sharpe': {
            'function': Metrics.sharpe,
            'weight': ValiConfig.SCORING_SHARPE_WEIGHT
        },
        'omega': {
            'function': Metrics.omega,
            'weight': ValiConfig.SCORING_OMEGA_WEIGHT
        },
        'sortino': {
            'function': Metrics.sortino,
            'weight': ValiConfig.SCORING_SORTINO_WEIGHT
        },
        'statistical_confidence': {
            'function': Metrics.statistical_confidence,
            'weight': ValiConfig.SCORING_STATISTICAL_CONFIDENCE_WEIGHT
        }
    }

    # Define the configuration with input types
    penalties_config = {
        'drawdown_threshold': PenaltyConfig(
            function=LedgerUtils.max_drawdown_threshold_penalty,
            input_type=PenaltyInputType.LEDGER
        ),
        'risk_profile': PenaltyConfig(
            function=PositionPenalties.risk_profile_penalty,
            input_type=PenaltyInputType.POSITIONS
        )
    }

    @staticmethod
    def compute_results_checkpoint(
            ledger_dict: dict[str, dict[str, PerfLedger]],
            full_positions: dict[str, list[Position]],
            evaluation_time_ms: int = None,
            verbose=True,
            weighting=False
    ) -> tuple[List[Tuple[str, float]], dict[str, float]]:
        if len(ledger_dict) == 0:
            bt.logging.debug("No results to compute, returning empty list")
            return [], {}

        if len(ledger_dict) == 1:
            miner = list(ledger_dict.keys())[0]
            if verbose:
                bt.logging.info(f"compute_results_checkpoint - Only one miner: {miner}, returning 1.0 for the solo miner weight")
            return [(miner, 1.0)], {}

        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        filtered_positions = PositionFiltering.filter(
            full_positions,
            evaluation_time_ms=evaluation_time_ms
        )

        # Compute miner penalties
        miner_penalties = Scoring.miner_penalties(filtered_positions, ledger_dict)

        # Miners with full penalty
        full_penalty_miner_scores: list[tuple[str, float]] = [
            (miner, 0) for miner, penalty in miner_penalties.items() if penalty == 0
        ]
        # Run all scoring functions
        asset_penalized_scores_dict = Scoring.score_miners(
            ledger_dict=ledger_dict,
            positions=full_positions,
            evaluation_time_ms=evaluation_time_ms,
            weighting=weighting
        )

        # Combine and penalize scores
        asset_combined_scores = Scoring.combine_scores(asset_penalized_scores_dict)
        miner_competitiveness = AssetSegmentation.asset_competitiveness_dictionary(asset_combined_scores)
        bt.logging.debug(f"Asset competitiveness: {miner_competitiveness}")

        # Now we probably want to apply the softmax to the asset combined scores
        asset_softmaxed_scores = Scoring.softmax_by_asset(asset_combined_scores)

        # Now combine the percentile scores prior to running a full softmax
        asset_aggregated_scores = Scoring.subclass_score_aggregation(asset_softmaxed_scores)

        # Force good performance of all error metrics
        combined_weighed = asset_aggregated_scores + full_penalty_miner_scores
        combined_scores = dict(combined_weighed)

        # Normalize the scores
        normalized_scores = Scoring.normalize_scores(combined_scores)
        return sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True), miner_competitiveness

    @staticmethod
    def score_miners(
            ledger_dict: dict[str, dict[str, PerfLedger]],
            positions: dict[str, list[Position]],
            evaluation_time_ms: int = None,
            weighting: bool = False
    ) -> dict[str, dict]:
        """
        Scores the miners based on their ledger and positions.
        Args:
            ledger_dict:
            positions:
            evaluation_time_ms:
            weighting:

        Returns:
            dict[str, dict]: A dictionary where keys are asset classes and values are dictionaries containing scores and penalties.

        """

        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        filtered_positions = PositionFiltering.filter(
            positions,
            evaluation_time_ms=evaluation_time_ms
        )

        # psuedo_positions = PositionUtils.build_pseudo_positions(filtered_positions)

        # Compute miner penalties
        miner_penalties = Scoring.miner_penalties(filtered_positions, ledger_dict)
        # miner_psuedo_penalties = Scoring.miner_penalties(psuedo_positions, ledger_dict)

        # full_miner_penalties = {
        #     miner: min(miner_penalties[miner], miner_psuedo_penalties[miner]) for miner in filtered_positions.keys()
        # }

        full_miner_penalties = miner_penalties

        # Miners with full penalty
        full_penalty_miners = set([
            miner for miner, penalty in full_miner_penalties.items() if penalty == 0
        ])

        # We now want to track miner incentive between each asset class
        asset_class_breakdown = ValiConfig.ASSET_CLASS_BREAKDOWN
        asset_subcategories = AssetSegmentation.distill_asset_subcategories(asset_class_breakdown)

        segmentation_machine = AssetSegmentation(ledger_dict)

        # This is going to track miner scores on each asset class
        miner_asset_benefit = {}

        for asset_subcategory in asset_subcategories:
            asset_ledger = segmentation_machine.segmentation(asset_subcategory)
            filtered_ledger_returns = LedgerUtils.ledger_returns_log(asset_ledger)
            days_in_year = segmentation_machine.days_in_year_from_asset_category(asset_subcategory.asset_class)

            scores_dict = {"metrics": {}}
            for config_name, config in Scoring.scoring_config.items():
                scores = []
                for miner, returns in filtered_ledger_returns.items():
                    # Get the miner ledger
                    ledger = asset_ledger.get(miner, PerfLedger())

                    # Check if the miner has full penalty - if not include them in the scoring competition
                    if miner in full_penalty_miners:
                        continue

                    score = config['function'](
                        log_returns=returns,
                        ledger=ledger,
                        weighting=weighting,
                        days_in_year=days_in_year,
                    )

                    scores.append((miner, float(score)))

                scores_dict["metrics"][config_name] = {
                    "scores": scores[:],
                    "weight": config["weight"]
                }

            scores_dict["penalties"] = copy.deepcopy(full_miner_penalties)
            miner_asset_benefit[asset_subcategory] = scores_dict

        return miner_asset_benefit

    @staticmethod
    def combine_scores(scoring_dict: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
        """
        Combines scores and penalties for each of the asset classes into a single score for each asset class.
        Args:
            scoring_dict:

        Returns:

        """

        asset_combined_scores = {}
        for asset_class, asset_scores in scoring_dict.items():
            combined_scores = {}
            for config_name, config in asset_scores["metrics"].items():

                percentile_scores = Scoring.miner_scores_percentiles(config["scores"])
                for miner, percentile_rank in percentile_scores:
                    if miner not in combined_scores:
                        combined_scores[miner] = 0
                    combined_scores[miner] += config['weight'] * percentile_rank  # + (1 - config['weight'])

            # Now applying the penalties post scoring
            for miner, penalty in asset_scores["penalties"].items():
                if miner in combined_scores:
                    combined_scores[miner] *= penalty

            asset_combined_scores[asset_class] = combined_scores

        return asset_combined_scores

    @staticmethod
    def miner_penalties(
            hotkey_positions: dict[str, list[Position]],
            ledger_dict: dict[str, dict[str, PerfLedger]]
    ) -> dict[str, float]:
        # Compute miner penalties
        miner_penalties = {}

        empty_ledger_miners = []
        for miner, ledger in ledger_dict.items():
            positions = hotkey_positions.get(miner, [])

            if not ledger:
                empty_ledger_miners.append((miner, len(positions)))

            # TODO: check if this will cause issues
            portfolio_ledger = ledger.get('portfolio') if ledger else {'portfolio': PerfLedger()}

            cumulative_penalty = 1
            for penalty_name, penalty_config in Scoring.penalties_config.items():
                # Apply penalty based on its input type
                penalty = 1
                if penalty_config.input_type == PenaltyInputType.LEDGER:
                    penalty = penalty_config.function(portfolio_ledger)
                elif penalty_config.input_type == PenaltyInputType.POSITIONS:
                    penalty = penalty_config.function(positions)

                cumulative_penalty *= penalty

            miner_penalties[miner] = cumulative_penalty

        if empty_ledger_miners:
            bt.logging.warning(f"Unexpectedly skipping miners with empty ledgers [(hk, n_positions)]: {empty_ledger_miners}")

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

        positional_returns = [math.log(
            max(position.return_at_close, .00001))  # Prevent math domain error
            for position in positions]

        aggregate_return = 0.0
        for positional_return in positional_returns:
            aggregate_return += positional_return

        return aggregate_return

    @staticmethod
    def class_aggregation(asset_miner_scores: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Aggregates the scores of miners across different asset classes.

        Args:
            asset_miner_scores (dict[str, dict[str, float]]): A dictionary where keys are asset classes and values are dictionaries of miner scores.

        Returns:
            dict[str, float]: A dictionary with aggregated scores for each miner across all asset classes.
        """
        asset_class_breakdown: dict[str, float] = ValiConfig.ASSET_CLASS_BREAKDOWN
        aggregated_scores = defaultdict(float)

        for asset_class, miner_scores in asset_miner_scores.items():
            for miner, score in miner_scores.items():
                aggregated_scores[miner] += score * asset_class_breakdown.get(asset_class, 0)

        return dict(aggregated_scores)

    @staticmethod
    def softmax_by_asset(
            asset_miner_scores: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """
        Applies softmax to the scores of miners within each asset class.

        Args:
            asset_miner_scores (dict[str, dict[str, float]]): A dictionary where keys are asset classes and values are dictionaries of miner scores.

        Returns:
            dict[str, dict[str, float]]: A dictionary with softmax scores for each miner within each asset class.
        """
        softmaxed_scores = {}
        for asset_class, miner_scores in asset_miner_scores.items():
            sorted_returns = sorted(miner_scores.items(), key=lambda x: x[1], reverse=True)
            softmaxed_scores[asset_class] = dict(Scoring.softmax_scores(sorted_returns))

        return softmaxed_scores

    @staticmethod
    def subclass_score_aggregation(
            miner_asset_scores: dict[str, dict[str, float]]
    ) -> list[tuple[str, float]]:
        """
        Aggregates the softmax scores of miners across different asset classes.

        Args:
            miner_asset_scores (dict[str, list[tuple[str, float]]]): A dictionary where keys are asset classes and values are lists of tuples with miner names and their softmax scores.

        Returns:
            list[tuple[str, float]]: A list of tuples with miner names and their aggregated softmax scores.
        """
        aggregated_scores = defaultdict(float)
        asset_class_breakdown = ValiConfig.ASSET_CLASS_BREAKDOWN
        category_lookup = ValiConfig.CATEGORY_LOOKUP

        # Compose the full penalties dictionary based on subcategories and weights
        full_penalties_dictionary = {}
        for asset_subclass, _ in miner_asset_scores.items():
            asset_class = category_lookup.get(asset_subclass, None)
            if asset_class is None:
                bt.logging.warning(f"Asset subclass {asset_subclass} not found in category lookup, assigning forex.")
                asset_class = TradePairCategory.FOREX

            asset_class_information = asset_class_breakdown.get(asset_class, {})

            asset_class_emission = asset_class_information.get('emission', 0)
            asset_subcategory_weight = asset_class_information.get('subcategory_weights', {})

            bt.logging.info(f"Asset class {asset_class} has emission {asset_class_emission} and subcategory weights {asset_subcategory_weight}")

            if asset_class_emission == 0:
                bt.logging.warning(f"Asset class {asset_class} has no emission. Please report this issue!")

            if asset_subcategory_weight is None or len(asset_subcategory_weight) == 0:
                raise ValueError(f"Asset class {asset_class} has no subcategory weights.")

            for subcategory, subcategory_weight in asset_subcategory_weight.items():
                full_penalties_dictionary[subcategory] = asset_class_emission * subcategory_weight

        bt.logging.info(f"Full penalties dictionary: {full_penalties_dictionary}")

        # Now check how the miners are achieving the asset class breakdown
        for subcategory, scores in miner_asset_scores.items():
            for miner, score in scores.items():
                asset_class_emission = full_penalties_dictionary.get(subcategory, 0)
                if miner not in aggregated_scores:
                    aggregated_scores[miner] = 0.0

                aggregated_scores[miner] += score * asset_class_emission

        return sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

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
            bt.logging.debug("No returns to score, returning empty list")
            return []

        if len(returns) == 1:
            bt.logging.info("softmax_scores - Only one miner, returning 1.0 for the solo miner weight")
            return [(returns[0][0], 1.0)]

        # Extract scores and apply softmax with temperature
        scores = np.array([score for _, score in returns])
        max_score = np.max(scores)
        exp_scores = np.exp((scores - max_score) / temperature)
        softmax_scores = exp_scores / max(np.sum(exp_scores), epsilon)

        # Combine miners with their respective softmax scores
        weighted_returns = [(miner, float(softmax_scores[i])) for i, (miner, _) in enumerate(returns)]

        return weighted_returns

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
            bt.logging.info(f"miner_scores_percentiles - Only one miner: {miner}, returning 1.0 for the solo miner weight")
            return [(miner, 1.0)]

        miner_hotkeys = []
        scores = []

        for miner, score in miner_scores:
            miner_hotkeys.append(miner)
            scores.append(score)

        percentiles = percentileofscore(scores, scores, kind='rank') / 100

        miner_percentiles = list(zip(miner_hotkeys, percentiles))

        return miner_percentiles

    @staticmethod
    def score_testing_miners(ledgers, miner_scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Applies time weighting and distributes challenge weights for prioritization
        Args:
            ledgers: list[tuple[str, float]] - the scores of the miners
            miner_scores: list[tuple[str, float]] - the scores of the miners
        Returns:
            list[tuple[str, float]] - the final weights of the miners
        """

        MIN_WEIGHT = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
        MAX_WEIGHT = ValiConfig.CHALLENGE_PERIOD_MAX_WEIGHT

        if not ledgers or not miner_scores:
            bt.logging.info(f"Ledgers: {ledgers} and miner scores: {miner_scores}, returning empty list")
            return []

        time_weighted = sorted(
            Metrics.time_weighted_scores(ledgers, miner_scores),
            key=lambda x: x[1],
            reverse=True
        )

        num_miners = len(time_weighted)

        distributed = np.linspace(MAX_WEIGHT, MIN_WEIGHT, num=num_miners)

        final_scores = [(miner, float(score)) for (miner, _), score in zip(time_weighted, distributed)]

        return sorted(final_scores, key=lambda x: x[1], reverse=True)
