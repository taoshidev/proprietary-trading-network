# developer: trdougherty
import os
from typing import List
import copy
import numpy as np

from scipy.stats import percentileofscore

from time_util.time_util import TimeUtil
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.position_filtering import PositionFiltering
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager


class MinerStatisticsManager:

    def __init__(self, position_manager, subtensor_weight_setter, plagiarism_detector):
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.elimination_manager = position_manager.elimination_manager
        self.challengeperiod_manager = position_manager.challengeperiod_manager
        self.subtensor_weight_setter = subtensor_weight_setter
        self.plagiarism_detector = plagiarism_detector

    def rank_dictionary(self, d, ascending=False):
        """
        Rank the values in a dictionary. Higher values get lower ranks by default.

        Args:
        d (dict): The dictionary to rank.
        ascending (bool): If True, ranks in ascending order. Default is False.

        Returns:
        dict: A dictionary with the same keys and ranked values.
        """
        # Sort the dictionary by value
        sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=not ascending)

        # Assign ranks
        ranks = {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}

        return ranks


    def apply_penalties(self, scores: dict[str, float], penalties: dict[str, float]) -> dict[str, float]:
        """
        Apply penalties to scores.

        Args:
        scores (dict): The scores to penalize.
        penalties (dict): The penalties to apply.

        Returns:
        dict: A dictionary with the same keys and penalized values.
        """
        penalized_scores = {k: scores[k] * penalties.get(k, 1.0) for k in scores.keys()}

        return penalized_scores


    def percentile_rank_dictionary(self, d, ascending=False) -> dict:
        """
        Rank the values in a dictionary as a percentile. Higher values get lower ranks by default.

        Args:
        d (dict): The dictionary to rank.
        ascending (bool): If True, ranks in ascending order. Default is False.

        Returns:
        dict: A dictionary with the same keys and ranked values.
        """
        # Sort the dictionary by value
        miner_names = list(d.keys())
        scores = list(d.values())

        percentiles = percentileofscore(scores, scores, kind='rank') / 100
        miner_percentiles = dict(zip(miner_names, percentiles))

        return miner_percentiles

    def percentiles_config(self, scores_dict: dict[str, dict], inspection_dict: dict[str, dict]):
        """
        Args:
            scores_dict: a dictionary with function names as keys to values that have:
            "scores" which is a list of tuples with (miner, score) for that metric,
            "weight" which is the weight of the metric

        Returns:
            percentile_dict: which just adds a field for each function that has a percentiles
            dictionary for that metric under "percentiles"
        """
        percentile_dict = copy.deepcopy(scores_dict)
        target_percentile = ValiConfig.CHALLENGE_PERIOD_PERCENTILE_THRESHOLD * 100

        # Calculate target percentile with only successful miners for combined scores
        combined_scores = Scoring.combine_scores(percentile_dict)
        combined_percentiles = [score for miner, score in combined_scores.items()]

        if len(combined_percentiles) > 0:
            overall_target = np.percentile(combined_percentiles, target_percentile, method="higher")
        else:
            # If no one in main competition, target score is 0
            overall_target = 0
        percentile_dict["overall_target_score"] = overall_target

        # Calculate target scores for each metric of only successful miners
        for config_name, config in percentile_dict["metrics"].items():

            scores = [score for miner, score in config['scores']]

            # If no one in main competition, target score is 0
            if len(scores) > 0:
                target_score = np.percentile(scores, target_percentile, method="higher")
            else:
                target_score = 0
            config["target_score"] = target_score

        # Append testing miner scores to successful miners
        for config_name, config in percentile_dict["metrics"].items():

            miner_scores = config["scores"]
            miner_scores += inspection_dict["metrics"][config_name]["scores"]

        for config_name, config in percentile_dict["metrics"].items():

            weighted_scores = Scoring.miner_scores_percentiles(config["scores"])
            config["percentiles"] = dict(weighted_scores)

        return percentile_dict


    def generate_miner_statistics_data(self, time_now: int = None, checkpoints: bool = True, selected_miner_hotkeys: List[str] = None):
        if time_now is None:
            time_now = TimeUtil.now_in_millis()

        # Get the dictionaries
        challengeperiod_testing_dictionary = self.challengeperiod_manager.get_challengeperiod_testing()
        challengeperiod_success_dictionary = self.challengeperiod_manager.get_challengeperiod_success()

        # Sort dictionaries by value
        sorted_challengeperiod_testing = dict(sorted(challengeperiod_testing_dictionary.items(), key=lambda item: item[1]))
        sorted_challengeperiod_success = dict(sorted(challengeperiod_success_dictionary.items(), key=lambda item: item[1]))

        challengeperiod_testing_hotkeys = list(challengeperiod_testing_dictionary.keys())
        challengeperiod_success_hotkeys = list(challengeperiod_success_dictionary.keys())

        try:
            if not os.path.exists(ValiBkpUtils.get_miner_dir()):
                raise FileNotFoundError
        except FileNotFoundError:
            raise Exception(
                f"directory for miners doesn't exist "
                f"[{ValiBkpUtils.get_miner_dir()}]. Skip run for now."
            )

        # full ledger of all miner hotkeys
        all_miner_hotkeys = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys
        if selected_miner_hotkeys is None:
            selected_miner_hotkeys = all_miner_hotkeys

        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=all_miner_hotkeys)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(hotkeys=all_miner_hotkeys)
        filtered_returns = LedgerUtils.ledger_returns_log(filtered_ledger)

        plagiarism = self.plagiarism_detector.get_plagiarism_scores_from_disk()
        # # Sync the ledger and positions
        # filtered_ledger, filtered_positions = subtensor_weight_setter.sync_ledger_positions(
        #     filtered_ledger,
        #     filtered_positions
        # )

        # Lookback window positions
        lookback_positions = PositionFiltering.filter(
            filtered_positions,
            evaluation_time_ms=time_now
        )

        # lookback_positions_recent = PositionFiltering.filter_recent(
        #     filtered_positions,
        #     evaluation_time_ms=time_now
        # )

        # Penalties
        miner_penalties = Scoring.miner_penalties(
            lookback_positions,
            filtered_ledger
        )

        # Scoring metrics
        omega_dict = {}
        sharpe_dict = {}
        sortino_dict = {}
        short_return_dict = {}
        return_dict = {}
        short_risk_adjusted_return_dict = {}
        risk_adjusted_return_dict = {}
        statistical_confidence_dict = {}
        concentration_dict = {}

        # Scoring criteria metrics
        minimum_days_threshold_dict = {}

        # Positional ratios
        positional_return_time_consistency_ratios = {}
        positional_realized_returns_ratios = {}

        # Positional penalties
        positional_return_time_consistency_penalties = {}
        positional_realized_returns_penalties = {}
        miner_martingale_scores = {}
        miner_martingale_penalties = {}

        # Ledger Ratios
        ledger_daily_consistency_ratios = {}
        ledger_biweekly_consistency_ratios = {}

        # Ledger Penalties
        daily_consistency_penalty = {}
        biweekly_consistency_penalty = {}
        drawdown_penalties = {}
        max_drawdown_threshold_penalties = {}
        drawdown_abnormality_penalties = {}

        # Ledger Drawdowns
        recent_drawdowns = {}
        approximate_drawdowns = {}
        effective_drawdowns = {}

        # Perf Ledger Calculations
        n_checkpoints = {}
        checkpoint_durations = {}

        # Positional Statistics
        n_positions = {}
        positional_return = {}
        positional_duration = {}

        # Volatility Metrics
        annual_volatility = {}
        annual_downside_volatility = {}

        for hotkey, hotkey_ledger in filtered_ledger.items():
            # Collect miner positions
            miner_positions = filtered_positions.get(hotkey, [])
            miner_returns = filtered_returns.get(hotkey, [])
            short_term_miner_returns = miner_returns[-ValiConfig.SHORT_LOOKBACK_WINDOW:]

            miner_checkpoints = hotkey_ledger.cps

            short_term_miner_checkpoints = hotkey_ledger.cps[-ValiConfig.SHORT_LOOKBACK_WINDOW:]
            # Lookback window positions
            miner_lookback_positions = lookback_positions.get(hotkey, [])

            scoring_input = {
                "log_returns": miner_returns,
                # "ledger": hotkey_ledger
            }

            # Track the positional thresholds
            minimum_days_threshold_dict[hotkey] = len(miner_returns) >= ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N

            # Positional Scoring
            omega_dict[hotkey] = Metrics.omega(**scoring_input, bypass_confidence=True)
            sharpe_dict[hotkey] = Metrics.sharpe(**scoring_input, bypass_confidence=True)
            sortino_dict[hotkey] = Metrics.sortino(**scoring_input, bypass_confidence=True)
            annual_volatility[hotkey] = min(Metrics.ann_volatility(miner_returns), 100)
            annual_downside_volatility[hotkey] = min(Metrics.ann_downside_volatility(miner_returns), 100)
            statistical_confidence_dict[hotkey] = Metrics.statistical_confidence(miner_returns, bypass_confidence=True)
            concentration_dict[hotkey] = Metrics.concentration(miner_returns, positions=miner_lookback_positions)

            # Positional penalties
            miner_martingale_scores[hotkey] = PositionPenalties.martingale_score(miner_lookback_positions)
            miner_martingale_penalties[hotkey] = PositionPenalties.martingale_penalty(miner_lookback_positions)

            short_return_dict[hotkey] = Metrics.base_return(short_term_miner_returns)
            return_dict[hotkey] = Metrics.base_return(miner_returns)

            short_risk_adjusted_return_dict[hotkey] = Metrics.drawdown_adjusted_return(
                short_term_miner_returns,
                short_term_miner_checkpoints
            )

            risk_adjusted_return_dict[hotkey] = Metrics.drawdown_adjusted_return(
                miner_returns,
                miner_checkpoints
            )

            # Ledger consistency penalties
            recent_drawdown = LedgerUtils.recent_drawdown(miner_checkpoints)
            recent_drawdowns[hotkey] = recent_drawdown

            approximate_drawdown = LedgerUtils.approximate_drawdown(miner_checkpoints)
            approximate_drawdowns[hotkey] = approximate_drawdown

            effective_drawdowns[hotkey] = LedgerUtils.effective_drawdown(recent_drawdown, approximate_drawdown)
            drawdown_penalties[hotkey] = LedgerUtils.risk_normalization(miner_checkpoints)

            ledger_daily_consistency_ratios[hotkey] = LedgerUtils.daily_consistency_ratio(miner_checkpoints)
            daily_consistency_penalty[hotkey] = LedgerUtils.daily_consistency_penalty(miner_checkpoints)

            ledger_biweekly_consistency_ratios[hotkey] = LedgerUtils.biweekly_consistency_ratio(miner_checkpoints)
            biweekly_consistency_penalty[hotkey] = LedgerUtils.biweekly_consistency_penalty(miner_checkpoints)

            drawdown_abnormality_penalties[hotkey] = LedgerUtils.drawdown_abnormality(miner_checkpoints)
            max_drawdown_threshold_penalties[hotkey] = LedgerUtils.max_drawdown_threshold_penalty(miner_checkpoints)

            # Positional consistency ratios
            positional_realized_returns_ratios[hotkey] = PositionPenalties.returns_ratio(miner_lookback_positions)
            positional_realized_returns_penalties[hotkey] = PositionPenalties.returns_ratio_penalty(miner_lookback_positions)

            positional_return_time_consistency_ratios[hotkey] = PositionPenalties.time_consistency_ratio(miner_lookback_positions)
            positional_return_time_consistency_penalty = PositionPenalties.time_consistency_penalty(miner_lookback_positions)
            positional_return_time_consistency_penalties[hotkey] = positional_return_time_consistency_penalty

            # Now for the ledger statistics
            n_checkpoints[hotkey] = len([x for x in miner_checkpoints if x.open_ms > 0])
            checkpoint_durations[hotkey] = sum([x.open_ms for x in miner_checkpoints])

            # Now for the full positions statistics
            n_positions[hotkey] = len(miner_positions)
            positional_return[hotkey] = Metrics.base_return(miner_returns)
            positional_duration[hotkey] = PositionUtils.total_duration(miner_positions)

        # Cumulative ledger, for printing
        cumulative_return_ledger = LedgerUtils.cumulative(filtered_ledger)

        # This is when we only want to look at the successful miners
        successful_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=challengeperiod_success_hotkeys)
        successful_positions, _ = self.position_manager.filtered_positions_for_scoring(hotkeys=challengeperiod_success_hotkeys)

        # successful_ledger, successful_positions = subtensor_weight_setter.sync_ledger_positions(
        #     successful_ledger,
        #     successful_positions
        # )

        checkpoint_results = Scoring.compute_results_checkpoint(
            successful_ledger,
            successful_positions,
            evaluation_time_ms=time_now,
            verbose=False
        )

        challengeperiod_scores = [
            (x, ValiConfig.CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_testing_hotkeys
        ]

        scoring_results = checkpoint_results + challengeperiod_scores
        weights = dict(scoring_results)

        # Rankings
        weights_rank = self.rank_dictionary(weights)
        weights_percentile = self.percentile_rank_dictionary(weights)

        # Rankings on Traditional Metrics
        omega_rank = self.rank_dictionary(omega_dict)
        omega_percentile = self.percentile_rank_dictionary(omega_dict)

        sortino_rank = self.rank_dictionary(sortino_dict)
        sortino_percentile = self.percentile_rank_dictionary(sortino_dict)

        sharpe_rank = self.rank_dictionary(sharpe_dict)
        sharpe_percentile = self.percentile_rank_dictionary(sharpe_dict)

        short_return_rank = self.rank_dictionary(short_return_dict)
        short_return_percentile = self.percentile_rank_dictionary(short_return_dict)

        return_rank = self.rank_dictionary(return_dict)
        return_percentile = self.percentile_rank_dictionary(return_dict)

        statistical_confidence_rank = self.rank_dictionary(statistical_confidence_dict)
        statistical_confidence_percentile = self.percentile_rank_dictionary(statistical_confidence_dict)

        short_risk_adjusted_return_rank = self.rank_dictionary(short_risk_adjusted_return_dict)
        short_risk_adjusted_return_percentile = self.percentile_rank_dictionary(short_risk_adjusted_return_dict)

        risk_adjusted_return_rank = self.rank_dictionary(risk_adjusted_return_dict)
        risk_adjusted_return_percentile = self.percentile_rank_dictionary(risk_adjusted_return_dict)

        # Rankings on Penalized Metrics
        omega_penalized_dict = self.apply_penalties(omega_dict, miner_penalties)
        omega_penalized_rank = self.rank_dictionary(omega_penalized_dict)
        omega_penalized_percentile = self.percentile_rank_dictionary(omega_penalized_dict)

        sharpe_penalized_dict = self.apply_penalties(sharpe_dict, miner_penalties)
        sharpe_penalized_rank = self.rank_dictionary(sharpe_penalized_dict)
        sharpe_penalized_percentile = self.percentile_rank_dictionary(sharpe_penalized_dict)

        sortino_penalized_dict = self.apply_penalties(sortino_dict, miner_penalties)
        sortino_penalized_rank = self.rank_dictionary(sortino_penalized_dict)
        sortino_penalized_percentile = self.percentile_rank_dictionary(sortino_penalized_dict)

        short_risk_adjusted_return_penalized_dict = self.apply_penalties(short_risk_adjusted_return_dict, miner_penalties)
        short_risk_adjusted_return_penalized_rank = self.rank_dictionary(short_risk_adjusted_return_penalized_dict)
        short_risk_adjusted_return_penalized_percentile = self.percentile_rank_dictionary(short_risk_adjusted_return_penalized_dict)

        risk_adjusted_return_penalized_dict = self.apply_penalties(risk_adjusted_return_dict, miner_penalties)
        risk_adjusted_return_penalized_rank = self.rank_dictionary(risk_adjusted_return_penalized_dict)
        risk_adjusted_return_penalized_percentile = self.percentile_rank_dictionary(risk_adjusted_return_penalized_dict)

        statistical_confidence_penalized_dict = self.apply_penalties(statistical_confidence_dict, miner_penalties)
        statistical_confidence_penalized_rank = self.rank_dictionary(statistical_confidence_penalized_dict)
        statistical_confidence_penalized_percentile = self.percentile_rank_dictionary(statistical_confidence_penalized_dict)

        # Here is the full list of data in frontend format

        # Get scores of successful miners for each metric
        success_scores_dict = Scoring.score_miners(ledger_dict=successful_ledger,
                                                   positions=successful_positions,
                                                   evaluation_time_ms=time_now)

        combined_data = []
        for miner_id in selected_miner_hotkeys:
            # challenge period specific data
            challengeperiod_specific = {}

            if miner_id in sorted_challengeperiod_testing:
                challengeperiod_testing_time = sorted_challengeperiod_testing[miner_id]
                chellengeperiod_end_time = challengeperiod_testing_time + ValiConfig.CHALLENGE_PERIOD_MS
                remaining_time = chellengeperiod_end_time - time_now
                challengeperiod_specific = {
                    "status": "testing",
                    "start_time_ms": challengeperiod_testing_time,
                    "remaining_time_ms": remaining_time,
                }
                inspection_positions = {miner_id: lookback_positions.get(miner_id, None)}

                # Get individual scoring dict for inspection
                inspection_ledger = {miner_id: filtered_ledger.get(miner_id, None)}

                # Get the scores for this miner for each metric
                inspection_scoring_dict = Scoring.score_miners(
                    ledger_dict=inspection_ledger,
                    positions=inspection_positions,
                    evaluation_time_ms=time_now)


                # Calculate percentiles for each metric
                percentile_dict = self.percentiles_config(scores_dict=success_scores_dict, inspection_dict=inspection_scoring_dict)

                # Update penalties for inspection miner
                percentile_dict["penalties"].update(inspection_scoring_dict["penalties"])

                # Combine scores and apply penalties
                combined_scores = Scoring.combine_scores(scoring_dict=percentile_dict)

                challengeperiod_trial_percentiles = self.percentile_rank_dictionary(dict(combined_scores))

                challengeperiod_trial_score = combined_scores.get(miner_id, 0)
                challengeperiod_trial_percentile = challengeperiod_trial_percentiles.get(miner_id, 0)

                challengeperiod_passing = bool(challengeperiod_trial_percentile >= ValiConfig.CHALLENGE_PERIOD_PERCENTILE_THRESHOLD)

                challengeperiod_specific["scores"] = {}
                for config_name, config in percentile_dict["metrics"].items():

                    inspection_metrics = inspection_scoring_dict["metrics"]
                    testing_score = inspection_metrics[config_name]["scores"]

                    # If not the right number of scores for any reason, skip metric since there was some error earlier
                    if len(testing_score) != 1:
                        continue

                    # There is only one score in the inspection_scoring_dict: the testing miner
                    value = testing_score[0][1]

                    # Show value and percentile for each metric
                    challengeperiod_specific["scores"][config_name] = {
                        "value": value,
                        "percentile": config["percentiles"].get(miner_id, 0),
                        "target_score": config["target_score"]
                    }
                # Show stats for overall score
                challengeperiod_specific["scores"]["overall"] = {
                        "value": challengeperiod_trial_score,
                        "percentile": challengeperiod_trial_percentile,
                        "target_percentile": ValiConfig.CHALLENGE_PERIOD_PERCENTILE_THRESHOLD,
                        "target_score": percentile_dict["overall_target_score"],
                        "passing": challengeperiod_passing,
                    }

            elif miner_id in sorted_challengeperiod_success:
                challengeperiod_success_time = sorted_challengeperiod_success[miner_id]
                challengeperiod_specific = {
                    "status": "success",
                    "start_time": challengeperiod_success_time,
                }

            # checkpoint specific data
            miner_cumulative_return_ledger = cumulative_return_ledger.get(miner_id)
            miner_standard_ledger = filtered_ledger.get(miner_id)

            if miner_standard_ledger is None:
                continue

            miner_data = {
                "hotkey": miner_id,
                "weight": {
                    "value": weights.get(miner_id),
                    "rank": weights_rank.get(miner_id),
                    "percentile": weights_percentile.get(miner_id),
                },
                "challengeperiod": challengeperiod_specific,
                "penalties": {
                    "drawdown_threshold": max_drawdown_threshold_penalties.get(miner_id),
                    "drawdown_abnormality": drawdown_abnormality_penalties.get(miner_id),
                    "martingale": miner_martingale_penalties.get(miner_id),
                    "total": miner_penalties.get(miner_id, 0.0),
                },
                "volatility": {
                    "annual": annual_volatility.get(miner_id),
                    "annual_downside": annual_downside_volatility.get(miner_id),
                },
                "drawdowns": {
                    "recent": recent_drawdowns.get(miner_id),
                    "approximate": approximate_drawdowns.get(miner_id),
                    "effective": effective_drawdowns.get(miner_id),
                },
                "scores": {
                    "omega": {
                        "value": omega_dict.get(miner_id),
                        "rank": omega_rank.get(miner_id),
                        "percentile": omega_percentile.get(miner_id),
                        "overall_contribution": omega_percentile.get(miner_id) * ValiConfig.SCORING_OMEGA_WEIGHT,
                    },
                    "sharpe": {
                        "value": sharpe_dict.get(miner_id),
                        "rank": sharpe_rank.get(miner_id),
                        "percentile": sharpe_percentile.get(miner_id),
                        "overall_contribution": sharpe_percentile.get(miner_id) * ValiConfig.SCORING_SHARPE_WEIGHT,
                    },
                    "sortino": {
                        "value": sortino_dict.get(miner_id),
                        "rank": sortino_rank.get(miner_id),
                        "percentile": sortino_percentile.get(miner_id),
                        "overall_contribution": sortino_percentile.get(miner_id) * ValiConfig.SCORING_SORTINO_WEIGHT,
                    },
                    "statistical_confidence": {
                        "value": statistical_confidence_dict.get(miner_id),
                        "rank": statistical_confidence_rank.get(miner_id),
                        "percentile": statistical_confidence_percentile.get(miner_id),
                        "overall_contribution": statistical_confidence_percentile.get(miner_id) * ValiConfig.SCORING_STATISTICAL_CONFIDENCE_WEIGHT,
                    },
                    "short_return": {
                        "value": short_return_dict.get(miner_id),
                        "rank": short_return_rank.get(miner_id),
                        "percentile": short_return_percentile.get(miner_id),
                        "overall_contribution": 0,
                    },
                    "return": {
                        "value": return_dict.get(miner_id),
                        "rank": return_rank.get(miner_id),
                        "percentile": return_percentile.get(miner_id),
                        "overall_contribution": 0,
                    },
                    "short-calmar": {
                        "value": short_risk_adjusted_return_dict.get(miner_id),
                        "rank": short_risk_adjusted_return_rank.get(miner_id),
                        "percentile": short_risk_adjusted_return_percentile.get(miner_id),
                        "overall_contribution": short_risk_adjusted_return_percentile.get(miner_id) * ValiConfig.SCORING_SHORT_RETURN_LOOKBACK_WEIGHT,
                    },
                    "calmar": {
                        "value": risk_adjusted_return_dict.get(miner_id),
                        "rank": risk_adjusted_return_rank.get(miner_id),
                        "percentile": risk_adjusted_return_percentile.get(miner_id),
                        "overall_contribution": risk_adjusted_return_percentile.get(miner_id) * ValiConfig.SCORING_LONG_RETURN_LOOKBACK_WEIGHT,
                    }
                },
                "penalized_scores": {
                    "omega": {
                        "value": omega_penalized_dict.get(miner_id),
                        "rank": omega_penalized_rank.get(miner_id),
                        "percentile": omega_penalized_percentile.get(miner_id),
                        "overall_contribution": omega_penalized_percentile.get(miner_id) * ValiConfig.SCORING_OMEGA_WEIGHT,
                    },
                    "sharpe": {
                        "value": sharpe_penalized_dict.get(miner_id),
                        "rank": sharpe_penalized_rank.get(miner_id),
                        "percentile": sharpe_penalized_percentile.get(miner_id),
                        "overall_contribution": sharpe_penalized_percentile.get(miner_id) * ValiConfig.SCORING_SHARPE_WEIGHT,
                    },
                    "sortino": {
                        "value": sortino_penalized_rank.get(miner_id),
                        "rank": sortino_penalized_rank.get(miner_id),
                        "percentile": sortino_penalized_percentile.get(miner_id),
                        "overall_contribution": sortino_penalized_percentile.get(miner_id) * ValiConfig.SCORING_SORTINO_WEIGHT,
                    },
                    "statistical_confidence": {
                        "value": statistical_confidence_penalized_dict.get(miner_id),
                        "rank": statistical_confidence_penalized_rank.get(miner_id),
                        "percentile": statistical_confidence_penalized_percentile.get(miner_id),
                        "overall_contribution": statistical_confidence_penalized_percentile.get(miner_id) * ValiConfig.SCORING_STATISTICAL_CONFIDENCE_WEIGHT,
                    },
                    "short-calmar": {
                        "value": short_risk_adjusted_return_penalized_dict.get(miner_id),
                        "rank": short_risk_adjusted_return_penalized_rank.get(miner_id),
                        "percentile": short_risk_adjusted_return_penalized_percentile.get(miner_id),
                        "overall_contribution": short_risk_adjusted_return_penalized_percentile.get(miner_id) * ValiConfig.SCORING_SHORT_RETURN_LOOKBACK_WEIGHT,
                    },
                    "calmar": {
                        "value": risk_adjusted_return_penalized_dict.get(miner_id),
                        "rank": risk_adjusted_return_penalized_rank.get(miner_id),
                        "percentile": risk_adjusted_return_penalized_percentile.get(miner_id),
                        "overall_contribution": risk_adjusted_return_penalized_percentile.get(miner_id) * ValiConfig.SCORING_LONG_RETURN_LOOKBACK_WEIGHT,
                    }
                },
                "engagement": {
                    "n_checkpoints": n_checkpoints.get(miner_id),
                    "n_positions": n_positions.get(miner_id),
                    "position_duration": positional_duration.get(miner_id),
                    "checkpoint_durations": checkpoint_durations.get(miner_id),
                    "minimum_days_boolean": minimum_days_threshold_dict.get(miner_id),
                },
                "plagiarism": plagiarism.get(miner_id),
                "martingale": miner_martingale_scores.get(miner_id),
            }

            miner_checkpoints = {
                "checkpoints": miner_cumulative_return_ledger.get('cps', [])
            }

            if checkpoints:
                miner_data = {**miner_data, **miner_checkpoints}

            combined_data.append(miner_data)

        # Now pipe the vali_config data into the final dictionary
        ldconfig_data = dict(ValiConfig.__dict__)
        ldconfig_printable = {
            key: value for key, value in ldconfig_data.items()
            if isinstance(value, (int, float, str))
               and key not in ['BASE_DIR', 'base_directory']
        }

        # Filter out invalid entries
        valid_data = [item for item in combined_data if item is not None and
                      item.get('weight') is not None and
                      item['weight'].get('rank') is not None]

        # If there's no valid data, return an empty dict or handle accordingly
        if not valid_data:
            return {
                'version': ValiConfig.VERSION,
                'created_timestamp_ms': time_now,
                'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
                'data': [],  # empty list as there's no valid data
                'constants': ldconfig_printable,
            }

        final_dict = {
            'version': ValiConfig.VERSION,
            'created_timestamp_ms': time_now,
            'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
            'data': sorted(valid_data, key=lambda x: x['weight']['rank']),
            'constants': ldconfig_printable,
        }

        return final_dict


    def generate_request_minerstatistics(self, time_now: int, checkpoints: bool = True):

        final_dict = self.generate_miner_statistics_data(time_now, checkpoints)

        output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "minerstatistics.json"
        ValiBkpUtils.write_file(
            output_file_path,
            final_dict,
        )


if __name__ == "__main__":
    perf_ledger_manager = PerfLedgerManager(None)
    elimination_manager = EliminationManager(None, None, None)
    position_manager = PositionManager(None, None, elimination_manager=elimination_manager,
                                       challengeperiod_manager=None,
                                       perf_ledger_manager=perf_ledger_manager)
    challengeperiod_manager = ChallengePeriodManager(None, None, position_manager=position_manager)

    elimination_manager.position_manager = position_manager
    position_manager.challengeperiod_manager = challengeperiod_manager
    elimination_manager.challengeperiod_manager = challengeperiod_manager
    challengeperiod_manager.position_manager = position_manager
    perf_ledger_manager.position_manager = position_manager
    subtensor_weight_setter = SubtensorWeightSetter(
        config=None,
        wallet=None,
        metagraph=None,
        running_unit_tests=False,
        position_manager=position_manager,
    )
    plagiarism_detector = PlagiarismDetector(None, None, position_manager=position_manager)
    msm = MinerStatisticsManager(position_manager, subtensor_weight_setter, plagiarism_detector)
    msm.generate_request_minerstatistics(1628572800000, True)
