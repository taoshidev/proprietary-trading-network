import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
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


# ---------------------------------------------------------------------------
# Enums and Dataclasses
# ---------------------------------------------------------------------------
class ScoreType(Enum):
    """Enum for different types of scores that can be calculated"""
    BASE = "base"
    PENALIZED = "penalized"
    AUGMENTED = "augmented"


@dataclass
class ScoreMetric:
    """Class to hold metric calculation configuration"""
    name: str
    metric_func: callable
    weight: float = 1.0
    requires_penalties: bool = False
    requires_weighting: bool = False


class ScoreResult:
    """Class to hold score calculation results"""

    def __init__(self, value: float, rank: int, percentile: float, overall_contribution: float = 0):
        self.value = value
        self.rank = rank
        self.percentile = percentile
        self.overall_contribution = overall_contribution

    def to_dict(self) -> Dict[str, float]:
        return {
            "value": self.value,
            "rank": self.rank,
            "percentile": self.percentile,
            "overall_contribution": self.overall_contribution
        }


# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------
class MetricsCalculator:
    """Class to handle all metrics calculations"""

    def __init__(self):
        # Add or remove metrics as desired. Excluding short-term metrics as requested.
        self.metrics = {
            "omega": ScoreMetric(
                name="omega",
                metric_func=Metrics.omega,
                weight=ValiConfig.SCORING_OMEGA_WEIGHT
            ),
            "sharpe": ScoreMetric(
                name="sharpe",
                metric_func=Metrics.sharpe,
                weight=ValiConfig.SCORING_SHARPE_WEIGHT
            ),
            "sortino": ScoreMetric(
                name="sortino",
                metric_func=Metrics.sortino,
                weight=ValiConfig.SCORING_SORTINO_WEIGHT
            ),
            "statistical_confidence": ScoreMetric(
                name="statistical_confidence",
                metric_func=Metrics.statistical_confidence,
                weight=ValiConfig.SCORING_STATISTICAL_CONFIDENCE_WEIGHT
            ),
            "risk_adjusted_return": ScoreMetric(
                name="calmar",
                metric_func=Metrics.drawdown_adjusted_return,
                weight=ValiConfig.SCORING_LONG_RETURN_LOOKBACK_WEIGHT
            ),
        }

    def calculate_metric(
        self,
        metric: ScoreMetric,
        data: Dict[str, Dict[str, Any]],
        penalties: Optional[Dict[str, float]] = None,
        weighting: bool = False
    ) -> Dict[str, float]:
        """
        Calculate a single metric for all miners.
        Optionally applies penalty or weighting based on the metric's flags and ScoreType.
        """
        scores = {}
        for hotkey, miner_data in data.items():
            log_returns = miner_data.get("log_returns", [])
            checkpoints = miner_data.get("checkpoints", [])
            # Compute raw metric
            value = 0.0
            try:
                value = metric.metric_func(
                    log_returns=log_returns,
                    checkpoints=checkpoints
                )
            except Exception:
                # If an error occurs in the metric function, set to 0
                value = 0.0

            # Penalty
            if penalties and metric.requires_penalties:
                value *= penalties.get(hotkey, 1.0)

            # Weighting
            if weighting and metric.requires_weighting:
                value *= metric.weight

            scores[hotkey] = value
        return scores


# ---------------------------------------------------------------------------
# MinerStatisticsManager
# ---------------------------------------------------------------------------
class MinerStatisticsManager:
    def __init__(
        self,
        position_manager: PositionManager,
        subtensor_weight_setter: SubtensorWeightSetter,
        plagiarism_detector: PlagiarismDetector
    ):
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.elimination_manager = position_manager.elimination_manager
        self.challengeperiod_manager = position_manager.challengeperiod_manager
        self.subtensor_weight_setter = subtensor_weight_setter
        self.plagiarism_detector = plagiarism_detector

        self.metrics_calculator = MetricsCalculator()

    # -------------------------------------------
    # Ranking / Percentile Helpers
    # -------------------------------------------
    def rank_dictionary(self, d: Dict[str, float], ascending: bool = False) -> Dict[str, int]:
        """Rank the values in a dictionary (descending by default)."""
        sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=not ascending)
        return {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}

    def percentile_rank_dictionary(self, d: Dict[str, float], ascending: bool = False) -> Dict[str, float]:
        """Calculate percentile ranks for dictionary values."""
        keys = list(d.keys())
        values = list(d.values())
        percentiles = []
        for val in values:
            p = percentileofscore(values, val, kind='rank') / 100.0
            percentiles.append(p)
        return dict(zip(keys, percentiles))

    # -------------------------------------------
    # Gather Extra Stats (drawdowns, volatility, etc.)
    # -------------------------------------------
    def gather_extra_data(self, hotkey: str, ledger_dict: Dict[str, Any], positions_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers additional data such as volatility, drawdowns, engagement stats,
        ignoring short-term metrics.
        """
        miner_ledger = ledger_dict.get(hotkey)
        miner_cps = miner_ledger.cps if miner_ledger else []
        miner_positions = positions_dict.get(hotkey, [])
        # ledger_returns_log returns {hotkey: [returns...], ...}
        miner_returns = LedgerUtils.ledger_returns_log(ledger_dict).get(hotkey, [])

        # Volatility
        ann_volatility = min(Metrics.ann_volatility(miner_returns), 100)
        ann_downside_volatility = min(Metrics.ann_downside_volatility(miner_returns), 100)

        # Drawdowns
        recent_dd = LedgerUtils.recent_drawdown(miner_cps)
        approx_dd = LedgerUtils.approximate_drawdown(miner_cps)
        effective_dd = LedgerUtils.effective_drawdown(recent_dd, approx_dd)

        # Engagement: positions
        n_positions = len(miner_positions)
        pos_duration = PositionUtils.total_duration(miner_positions)
        percentage_profitable = self.position_manager.get_percent_profitable_positions(miner_positions)

        # Engagement: checkpoints
        n_checkpoints = len([cp for cp in miner_cps if cp.open_ms > 0])
        checkpoint_durations = sum(cp.open_ms for cp in miner_cps)

        # Minimum days boolean
        meets_min_days = (len(miner_returns) >= ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N)

        # Martingale
        martingale_score = PositionPenalties.martingale_score(miner_positions)

        return {
            "annual_volatility": ann_volatility,
            "annual_downside_volatility": ann_downside_volatility,
            "drawdowns": {
                "recent": recent_dd,
                "approximate": approx_dd,
                "effective": effective_dd
            },
            "positions_info": {
                "n_positions": n_positions,
                "positional_duration": pos_duration,
                "percentage_profitable": percentage_profitable
            },
            "checkpoints_info": {
                "n_checkpoints": n_checkpoints,
                "checkpoint_durations": checkpoint_durations
            },
            "minimum_days_boolean": meets_min_days,
            "martingale_score": martingale_score
        }

    # -------------------------------------------
    # Prepare data for metric calculations
    # -------------------------------------------
    def prepare_miner_data(self, hotkey: str, filtered_ledger: Dict[str, Any], filtered_positions: Dict[str, Any], time_now: int) -> Dict[str, Any]:
        """
        Combines the minimal fields needed for the metrics plus the extra data.
        """
        miner_ledger = filtered_ledger.get(hotkey)
        miner_returns = LedgerUtils.ledger_returns_log(filtered_ledger).get(hotkey, [])
        miner_cps = miner_ledger.cps if miner_ledger else []
        miner_positions = filtered_positions.get(hotkey, [])

        extra_data = self.gather_extra_data(hotkey, filtered_ledger, filtered_positions)

        return {
            "log_returns": miner_returns,
            "checkpoints": miner_cps,
            "positions": miner_positions,
            "ledger": miner_ledger,
            "evaluation_time": time_now,
            "extra_data": extra_data
        }

    # -------------------------------------------
    # Penalties: store them individually so we can show them
    # -------------------------------------------
    def calculate_penalties_breakdown(self, miner_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Returns a dict:
            {
               hotkey: {
                  "drawdown_threshold": ...,
                  "martingale": ...,
                  "total": ...
               }
            }
        """
        results = {}
        for hotkey, data in miner_data.items():
            cps = data.get("checkpoints", [])
            positions = data.get("positions", [])

            drawdown_threshold_penalty = LedgerUtils.max_drawdown_threshold_penalty(cps)
            martingale_penalty = PositionPenalties.martingale_penalty(positions)

            total_penalty = drawdown_threshold_penalty * martingale_penalty

            results[hotkey] = {
                "drawdown_threshold": drawdown_threshold_penalty,
                "martingale": martingale_penalty,
                "total": total_penalty
            }
        return results

    # -------------------------------------------
    # Compute a single penalty factor per miner for "PENALIZED" metrics
    # (the "total" from the breakdown).
    # -------------------------------------------
    def calculate_penalties(self, miner_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        breakdown = self.calculate_penalties_breakdown(miner_data)
        return {hk: breakdown[hk]["total"] for hk in breakdown}

    # -------------------------------------------
    # Main scoring wrapper
    # -------------------------------------------
    def calculate_all_scores(
            self,
            miner_data: Dict[str, Dict[str, Any]],
            score_type: ScoreType = ScoreType.BASE
    ) -> Dict[str, Dict[str, ScoreResult]]:
        """Calculate all metrics for all miners (BASE, PENALIZED, AUGMENTED)."""
        # Initialize flags
        penalties = None
        weighting = False

        # Reset all flags first
        for metric in self.metrics_calculator.metrics.values():
            metric.requires_penalties = False
            metric.requires_weighting = False

        # Set flags based on score type
        if score_type == ScoreType.PENALIZED:
            penalties = self.calculate_penalties(miner_data)
            weighting = True
            for metric in self.metrics_calculator.metrics.values():
                metric.requires_penalties = True
                metric.requires_weighting = True
        elif score_type == ScoreType.AUGMENTED:
            weighting = True
            for metric in self.metrics_calculator.metrics.values():
                metric.requires_weighting = True

        # Calculate for each metric
        metric_results = {}
        for metric_name, metric in self.metrics_calculator.metrics.items():
            numeric_scores = self.metrics_calculator.calculate_metric(
                metric, miner_data, penalties=penalties, weighting=weighting
            )
            ranks = self.rank_dictionary(numeric_scores)
            percentiles = self.percentile_rank_dictionary(numeric_scores)

            # Build ScoreResult objects
            metric_results[metric_name] = {
                hotkey: ScoreResult(
                    value=numeric_scores[hotkey],
                    rank=ranks[hotkey],
                    percentile=percentiles[hotkey],
                    overall_contribution=percentiles[hotkey] * metric.weight
                )
                for hotkey in numeric_scores
            }

        return metric_results

    # -------------------------------------------
    # Generate final data
    # -------------------------------------------
    def generate_miner_statistics_data(
        self,
        time_now: int = None,
        checkpoints: bool = True,
        selected_miner_hotkeys: List[str] = None
    ) -> Dict[str, Any]:

        if time_now is None:
            time_now = TimeUtil.now_in_millis()

        # ChallengePeriod: success + testing
        challengeperiod_testing_dict = self.challengeperiod_manager.get_challengeperiod_testing()
        challengeperiod_success_dict = self.challengeperiod_manager.get_challengeperiod_success()

        sorted_challengeperiod_testing = dict(sorted(challengeperiod_testing_dict.items(), key=lambda x: x[1]))
        sorted_challengeperiod_success = dict(sorted(challengeperiod_success_dict.items(), key=lambda x: x[1]))

        challengeperiod_testing_hotkeys = list(sorted_challengeperiod_testing.keys())
        challengeperiod_success_hotkeys = list(sorted_challengeperiod_success.keys())

        all_miner_hotkeys = list(set(challengeperiod_testing_hotkeys + challengeperiod_success_hotkeys))
        if selected_miner_hotkeys is None:
            selected_miner_hotkeys = all_miner_hotkeys

        # Filter ledger/positions
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(all_miner_hotkeys)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(all_miner_hotkeys)

        # For weighting logic: gather "successful" checkpoint-based results
        successful_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(challengeperiod_success_hotkeys)
        successful_positions, _ = self.position_manager.filtered_positions_for_scoring(challengeperiod_success_hotkeys)

        # Compute the checkpoint-based weighting for successful miners
        checkpoint_results = Scoring.compute_results_checkpoint(
            successful_ledger, successful_positions,
            evaluation_time_ms=time_now, verbose=False, weighting=True
        )  # returns list of (hotkey, weightVal)

        # For testing miners, we might just give them a default "CHALLENGE_PERIOD_WEIGHT"
        challengeperiod_scores = [
            (hk, ValiConfig.CHALLENGE_PERIOD_WEIGHT) for hk in challengeperiod_testing_hotkeys
        ]

        # Combine them
        combined_weights_list = checkpoint_results + challengeperiod_scores

        ######## TEMPORARY LOGIC FOR BLOCK REMOVALS ON MINERS - REMOVE WHEN CLEARED
        dtao_registration_bug_registrations = set(['5Dvep8Psc5ASQf6jGJHz5qsi8x1HS2sefRbkKxNNjPcQYPfH', '5DnViSacXqrP8FnQMtpAFGyahUPvU2A6pbrX7wcexb3bmVjb', '5Grgb5e4aHrGzhAd1ZSFQwUHQSM5yaJw5Dp7T7ss7yLY17jB',
         '5FbaR3qjbbnYpkDCkuh4TUqqen1UMSscqjmhoDWQgGRh189o', '5FqSBwa7KXvv8piHdMyVbcXQwNWvT9WjHZGHAQwtoGVQD3vo', '5F25maVPbzV4fojdABw5Jmawr43UAc5uNRJ3VjgKCUZrYFQh',
         '5DjqgrgQcKdrwGDg7RhSkxjnAVWwVgYTBodAdss233s3zJ6T', '5FpypsPpSFUBpByFXMkJ34sV88PRjAKSSBkHkmGXMqFHR19Q', '5CXsrszdjWooHK3tfQH4Zk6spkkSsduFrEHzMemxU7P2wh7H',
         '5EFbAfq4dsGL6Fu6Z4jMkQUF3WiGG7XczadUvT48b9U7gRYW', '5GyBmAHFSFRca5BYY5yHC3S8VEcvZwgamsxyZTXep5prVz9f', '5EXWvBCADJo1JVv6jHZPTRuV19YuuJBnjG3stBm3bF5cR9oy',
         '5HDjwdba5EvQy27CD6HksabaHaPP4NSHLLaH2o9CiD3aA5hv', '5EWSKDmic7fnR89AzVmqLL14YZbJK53pxSc6t3Y7qbYm5SaV', '5DQ1XPp8KuDEwGP1eC9eRacpLoA1RBLGX22kk5vAMBtp3kGj',
         '5ERorZ39jVQJ7cMx8j8osuEV8dAHHCbpx8kGZP4Ygt5dxf93', '5GsNcT3ENpxQdNnM2LTSC5beBneEddZjpUhNVCcrdUbicp1w'])

        combined_weights_dict = dict(combined_weights_list)
        for hotkey, w_val in combined_weights_dict.items():
            if hotkey in dtao_registration_bug_registrations:
                combined_weights_dict[hotkey] = 0.0

        # Rebuild the list
        combined_weights_list = list(combined_weights_dict.items())
        #################################

        weights_dict = dict(combined_weights_list)
        weights_rank = self.rank_dictionary(weights_dict)
        weights_percentile = self.percentile_rank_dictionary(weights_dict)

        # Load plagiarism once
        plagiarism_scores = self.plagiarism_detector.get_plagiarism_scores_from_disk()

        # Prepare data for each miner
        miner_data = {}
        for hotkey in selected_miner_hotkeys:
            miner_data[hotkey] = self.prepare_miner_data(hotkey, filtered_ledger, filtered_positions, time_now)

        # Compute the base, penalized, and augmented scores
        base_scores = self.calculate_all_scores(miner_data, ScoreType.BASE)
        penalized_scores = self.calculate_all_scores(miner_data, ScoreType.PENALIZED)
        augmented_scores = self.calculate_all_scores(miner_data, ScoreType.AUGMENTED)

        # Also compute penalty breakdown (for display in final "penalties" dict).
        penalty_breakdown = self.calculate_penalties_breakdown(miner_data)

        # Build the final list
        results = []
        for hotkey in selected_miner_hotkeys:

            # ChallengePeriod info
            challengeperiod_info = {}
            if hotkey in sorted_challengeperiod_testing:
                cp_start = sorted_challengeperiod_testing[hotkey]
                cp_end = cp_start + ValiConfig.CHALLENGE_PERIOD_MS
                remaining = cp_end - time_now
                challengeperiod_info = {
                    "status": "testing",
                    "start_time_ms": cp_start,
                    "remaining_time_ms": max(remaining, 0)
                }
            elif hotkey in sorted_challengeperiod_success:
                cp_start = sorted_challengeperiod_success[hotkey]
                challengeperiod_info = {
                    "status": "success",
                    "start_time_ms": cp_start
                }

            # Build a small function to extract ScoreResult -> dict for each metric
            def build_scores_dict(metric_set: Dict[str, Dict[str, ScoreResult]]) -> Dict[str, Dict[str, float]]:
                out = {}
                for metric_name, hotkey_map in metric_set.items():
                    sr = hotkey_map.get(hotkey)
                    if sr is not None:
                        out[metric_name] = sr.to_dict()
                    else:
                        out[metric_name] = {}
                return out

            base_dict = build_scores_dict(base_scores)
            penalized_dict = build_scores_dict(penalized_scores)
            augmented_dict = build_scores_dict(augmented_scores)

            # Extra data
            extra = miner_data[hotkey].get("extra_data", {})

            # Volatility
            volatility_subdict = {
                "annual": extra.get("annual_volatility"),
                "annual_downside": extra.get("annual_downside_volatility"),
            }
            # Drawdowns
            drawdowns_subdict = extra.get("drawdowns", {})
            # Engagement
            engagement_subdict = {
                "n_checkpoints": extra.get("checkpoints_info", {}).get("n_checkpoints"),
                "n_positions": extra.get("positions_info", {}).get("n_positions"),
                "position_duration": extra.get("positions_info", {}).get("positional_duration"),
                "checkpoint_durations": extra.get("checkpoints_info", {}).get("checkpoint_durations"),
                "minimum_days_boolean": extra.get("minimum_days_boolean"),
                "percentage_profitable": extra.get("positions_info", {}).get("percentage_profitable"),
            }
            # Plagiarism
            plagiarism_val = plagiarism_scores.get(hotkey)
            # Martingale (score)
            martingale_val = extra.get("martingale_score")

            # Weight
            w_val = weights_dict.get(hotkey)
            w_rank = weights_rank.get(hotkey)
            w_pct = weights_percentile.get(hotkey)

            # Penalties breakdown for display
            pen_break = penalty_breakdown.get(hotkey, {})
            # e.g. {"drawdown_threshold": x, "martingale": y, "total": z}

            final_miner_dict = {
                "hotkey": hotkey,
                "challengeperiod": challengeperiod_info,
                "scores": base_dict,
                "penalized_scores": penalized_dict,
                "augmented_scores": augmented_dict,
                "volatility": volatility_subdict,
                "drawdowns": drawdowns_subdict,
                "plagiarism": plagiarism_val,
                "martingale": martingale_val,
                "engagement": engagement_subdict,
                "penalties": {
                    "drawdown_threshold": pen_break.get("drawdown_threshold", 1.0),
                    "martingale": pen_break.get("martingale", 1.0),
                    "total": pen_break.get("total", 1.0),
                },
                "weight": {
                    "value": w_val,
                    "rank": w_rank,
                    "percentile": w_pct,
                }
            }

            # Optionally attach actual checkpoints (like the original first script)
            if checkpoints:
                ledger_obj = miner_data[hotkey].get("ledger")
                if ledger_obj and hasattr(ledger_obj, "cps"):
                    final_miner_dict["checkpoints"] = ledger_obj.cps

            results.append(final_miner_dict)

        # (Optional) sort by weight rank if you want the final data sorted in that manner:
        # Filter out any miners lacking a weight, then sort.
        # If you want to keep them all, remove this filtering:
        results_with_weight = [r for r in results if r["weight"]["rank"] is not None]
        # Sort by ascending rank
        results_sorted = sorted(results_with_weight, key=lambda x: x["weight"]["rank"])

        # If you'd prefer not to filter out those without weight, you could keep them at the end
        # Or you can unify them in a single list. For simplicity, let's keep it consistent:
        final_dict = {
            'version': ValiConfig.VERSION,
            'created_timestamp_ms': time_now,
            'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
            'data': results_sorted,
            'constants': self.get_printable_config()
        }
        return final_dict

    # -------------------------------------------
    # Printable config
    # -------------------------------------------
    def get_printable_config(self) -> Dict[str, Any]:
        """Get printable configuration values."""
        config_data = dict(ValiConfig.__dict__)
        return {
            key: value for key, value in config_data.items()
            if isinstance(value, (int, float, str))
               and key not in ['BASE_DIR', 'base_directory']
        }

    # -------------------------------------------
    # Write to disk
    # -------------------------------------------
    def generate_request_minerstatistics(self, time_now: int, checkpoints: bool = True):
        final_dict = self.generate_miner_statistics_data(time_now, checkpoints)
        output_file_path = os.path.join(ValiBkpUtils.get_vali_outputs_dir(), "minerstatistics.json")
        ValiBkpUtils.write_file(output_file_path, final_dict)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    perf_ledger_manager = PerfLedgerManager(None)
    elimination_manager = EliminationManager(None, None, None)
    position_manager = PositionManager(
        None, None,
        elimination_manager=elimination_manager,
        challengeperiod_manager=None,
        perf_ledger_manager=perf_ledger_manager
    )
    challengeperiod_manager = ChallengePeriodManager(None, None, position_manager=position_manager)

    # Cross-wire references
    elimination_manager.position_manager = position_manager
    position_manager.challengeperiod_manager = challengeperiod_manager
    elimination_manager.challengeperiod_manager = challengeperiod_manager
    challengeperiod_manager.position_manager = position_manager
    perf_ledger_manager.position_manager = position_manager

    subtensor_weight_setter = SubtensorWeightSetter(
        config=None,
        metagraph=None,
        running_unit_tests=False,
        position_manager=position_manager,
    )
    plagiarism_detector = PlagiarismDetector(None, None, position_manager=position_manager)

    msm = MinerStatisticsManager(position_manager, subtensor_weight_setter, plagiarism_detector)
    msm.generate_request_minerstatistics(1628572800000, True)

