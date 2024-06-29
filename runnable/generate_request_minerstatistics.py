import json
import traceback
import uuid
import time
import copy
import math
from scipy.stats import percentileofscore

from json import JSONDecodeError
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.scoring.scoring import Scoring, ScoringUnit
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager

def rank_dictionary(d, ascending=False):
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

def percentile_rank_dictionary(d, ascending=False) -> dict:
    """
    Rank the values in a dictionary as a percentile. Higher values get lower ranks by default.
    
    Args:
    d (dict): The dictionary to rank.
    ascending (bool): If True, ranks in ascending order. Default is False.
    
    Returns:
    dict: A dictionary with the same keys and ranked values.
    """
    # Sort the dictionary by value
    minernames = list(d.keys())
    scores = list(d.values())

    percentiles = percentileofscore( scores, scores ) / 100
    miner_percentiles = dict(zip(minernames, percentiles))
    
    return miner_percentiles


def generate_request_minerstatistics(time_now:int):
    time_now = TimeUtil.now_in_millis()
    subtensor_weight_setter = SubtensorWeightSetter(
        config=None,
        wallet=None,
        metagraph=None,
        running_unit_tests=False
    )

    ## Collect information from the disk and populate variables in memory
    subtensor_weight_setter._refresh_eliminations_in_memory()
    subtensor_weight_setter._refresh_challengeperiod_in_memory()

    # Get the dictionaries
    challengeperiod_testing_dictionary = subtensor_weight_setter.challengeperiod_testing
    challengeperiod_success_dictionary = subtensor_weight_setter.challengeperiod_success

    # Sort dictionaries by value
    sorted_challengeperiod_testing = dict(sorted(challengeperiod_testing_dictionary.items(), key=lambda item: item[1]))
    sorted_challengeperiod_success = dict(sorted(challengeperiod_success_dictionary.items(), key=lambda item: item[1]))

    challengeperiod_testing_hotkeys = list(challengeperiod_testing_dictionary.keys())
    challengeperiod_success_hotkeys = list(challengeperiod_success_dictionary.keys())

    ## get plagiarism scores
    plagiarism = subtensor_weight_setter.get_plagiarism_scores_from_disk()
    
    try:
        all_miner_hotkeys:list = ValiBkpUtils.get_directories_in_dir(
            ValiBkpUtils.get_miner_dir()
        )
    except FileNotFoundError:
        raise Exception(
            f"directory for miners doesn't exist "
            f"[{ValiBkpUtils.get_miner_dir()}]. Skip run for now."
        )
    
    # Perf Ledger Calculations
    n_checkpoints = {}
    checkpoint_durations = {}
    volume_threshold_count = {}
    omega_cps = {}
    inverted_sortino_cps = {}
    return_cps = {}
    consistency_penalties = {}

    ## full ledger of all miner hotkeys
    all_miner_hotkeys = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys

    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=all_miner_hotkeys)

    ## Penalties
    consistency_penalties = {}
    drawdown_penalties = {}
    recent_drawdowns = {}

    for hotkey, hotkey_ledger in filtered_ledger.items():
        recent_drawdown = PositionUtils.compute_recent_drawdown(hotkey_ledger.cps)
        consistency_penalty = PositionUtils.compute_consistency_penalty_cps(hotkey_ledger.cps)
        drawdown_penalty = PositionUtils.compute_drawdown_penalty_cps(hotkey_ledger.cps)
        recent_drawdowns[hotkey] = recent_drawdown
        consistency_penalties[hotkey] = consistency_penalty
        drawdown_penalties[hotkey] = drawdown_penalty

    ## Non-augmented values for everything
    for hotkey, miner_ledger in filtered_ledger.items():
        scoringunit = ScoringUnit.from_perf_ledger(miner_ledger)
        return_cps[hotkey] = math.exp(Scoring.return_cps(scoringunit))
        omega_cps[hotkey] = Scoring.omega_cps(scoringunit)
        inverted_sortino_cps[hotkey] = Scoring.inverted_sortino_cps(scoringunit)
        n_checkpoints[hotkey] = len([ x for x in miner_ledger.cps if x.open_ms > 0 ])
        checkpoint_durations[hotkey] = sum([ x.open_ms for x in miner_ledger.cps ])
        volume_threshold_count[hotkey] = Scoring.checkpoint_volume_threshold_count(scoringunit)

    ## Now all of the augmented terms
    # there are two augmented ledgers
    short_return_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS_SHORT
    long_return_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS_LONG
    risk_adjusted_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RISKMETRIC

    augmented_omega_cps = {}
    augmented_inverted_sortino_cps = {}
    augmented_return_short_cps = {}
    augmented_return_long_cps = {}

    cumulative_returns = PositionManager.cumulative_returns(filtered_ledger)
    returns_ledger_short = PositionManager.augment_perf_ledger(
        ledger=filtered_ledger,
        evaluation_time_ms=time_now,
        time_decay_coefficient=short_return_decay_coefficient
    )

    returns_ledger_long = PositionManager.augment_perf_ledger(
        ledger=filtered_ledger,
        evaluation_time_ms=time_now,
        time_decay_coefficient=long_return_decay_coefficient
    )

    risk_adjusted_ledger = PositionManager.augment_perf_ledger(
        ledger=filtered_ledger,
        evaluation_time_ms=time_now,
        time_decay_coefficient=risk_adjusted_decay_coefficient
    )

    for hotkey, miner_ledger in returns_ledger_short.items():
        scoringunit = ScoringUnit.from_perf_ledger(miner_ledger)
        augmented_return_short_cps[hotkey] = math.exp(Scoring.return_cps(scoringunit))

    for hotkey, miner_ledger in returns_ledger_long.items():
        scoringunit = ScoringUnit.from_perf_ledger(miner_ledger)
        augmented_return_long_cps[hotkey] = math.exp(Scoring.return_cps(scoringunit))

    for hotkey, miner_ledger in risk_adjusted_ledger.items():
        scoringunit = ScoringUnit.from_perf_ledger(miner_ledger)
        augmented_omega_cps[hotkey] = Scoring.omega_cps(scoringunit)
        augmented_inverted_sortino_cps[hotkey] = Scoring.inverted_sortino_cps(scoringunit)

    ## This is when we only want to look at the successful miners
    successful_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=challengeperiod_success_hotkeys)
    checkpoint_results = Scoring.compute_results_checkpoint(successful_ledger, evaluation_time_ms=time_now, verbose=False)
    challengeperiod_scores = [ (x, ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_testing_hotkeys ]
    scoring_results = checkpoint_results + challengeperiod_scores
    weights = dict(scoring_results)

    ## now also ranking each of the miners
    weights_rank = rank_dictionary(weights)
    weights_percentile = percentile_rank_dictionary(weights)

    # standard terms
    omega_rank = rank_dictionary(omega_cps)
    omega_percentile = percentile_rank_dictionary(omega_cps)

    inverted_sortino_rank = rank_dictionary(inverted_sortino_cps)
    inverted_sortino_percentile = percentile_rank_dictionary(inverted_sortino_cps)

    return_rank = rank_dictionary(return_cps)
    return_percentile = percentile_rank_dictionary(return_cps)

    # now the augmented terms
    augmented_omega_rank = rank_dictionary(augmented_omega_cps)
    augmented_omega_percentile = percentile_rank_dictionary(augmented_omega_cps)

    augmented_inverted_sortino_rank = rank_dictionary(augmented_inverted_sortino_cps)
    augmented_inverted_sortino_percentile = percentile_rank_dictionary(augmented_inverted_sortino_cps)

    augmented_return_short_rank = rank_dictionary(augmented_return_short_cps)
    augmented_return_short_percentile = percentile_rank_dictionary(augmented_return_short_cps)

    augmented_return_long_rank = rank_dictionary(augmented_return_long_cps)
    augmented_return_long_percentile = percentile_rank_dictionary(augmented_return_long_cps)

    ## Here is the full list of data in frontend format
    combined_data = []
    for miner_id in all_miner_hotkeys:
        ## challenge period specific data
        challengeperiod_specific = {}

        if miner_id in sorted_challengeperiod_testing:
            challengeperiod_testing_time = sorted_challengeperiod_testing[miner_id]
            chellengeperiod_end_time = challengeperiod_testing_time + ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS
            remaining_time = chellengeperiod_end_time - time_now
            challengeperiod_specific = {
                "status": "testing",
                "start_time_ms": challengeperiod_testing_time,
                "remaining_time_ms": remaining_time,
            }

        elif miner_id in sorted_challengeperiod_success:
            challengeperiod_success_time = sorted_challengeperiod_success[miner_id]
            challengeperiod_specific = {
                "status": "success",
                "start_time": challengeperiod_success_time,
            }

        ## checkpoint specific data
        miner_cumulative_returns = cumulative_returns.get(miner_id)
        miner_standard_ledger = filtered_ledger.get(miner_id)
        miner_returns_short_ledger = returns_ledger_short.get(miner_id)
        miner_returns_long_ledger = returns_ledger_long.get(miner_id)
        miner_risk_ledger = risk_adjusted_ledger.get(miner_id)

        if miner_standard_ledger is None or miner_returns_short_ledger is None or miner_returns_long_ledger is None or miner_risk_ledger is None:
            continue

        miner_data = {
            "hotkey": miner_id,
            "weight": {
                "value": weights.get(miner_id),
                "rank": weights_rank.get(miner_id),
                "percentile": weights_percentile.get(miner_id),
            },
            "challengeperiod": challengeperiod_specific,
            "recent_drawdown": recent_drawdowns.get(miner_id),
            "penalties": {
                "consistency": consistency_penalties.get(miner_id),
                "drawdown": drawdown_penalties.get(miner_id),
            },
            "scores":{
                "omega": {
                    "value": omega_cps.get(miner_id),
                    "rank": omega_rank.get(miner_id),
                    "percentile": omega_percentile.get(miner_id),
                },
                "inverted_sortino_cps": {
                    "value": inverted_sortino_cps.get(miner_id),
                    "rank": inverted_sortino_rank.get(miner_id),
                    "percentile": inverted_sortino_percentile.get(miner_id),
                },
                "return_cps": {
                    "value": return_cps.get(miner_id),
                    "rank": return_rank.get(miner_id),
                    "percentile": return_percentile.get(miner_id),
                }
            },
            "augmented_scores":{
                "omega": {
                    "value": augmented_omega_cps.get(miner_id),
                    "rank": augmented_omega_rank.get(miner_id),
                    "percentile": augmented_omega_percentile.get(miner_id),
                },
                "inverted_sortino_cps": {
                    "value": augmented_inverted_sortino_cps.get(miner_id),
                    "rank": augmented_inverted_sortino_rank.get(miner_id),
                    "percentile": augmented_inverted_sortino_percentile.get(miner_id),
                },
                "returns_short_cps": {
                    "value": augmented_return_short_cps.get(miner_id),
                    "rank": augmented_return_short_rank.get(miner_id),
                    "percentile": augmented_return_short_percentile.get(miner_id),
                },
                "returns_long_cps": {
                    "value": augmented_return_long_cps.get(miner_id),
                    "rank": augmented_return_long_rank.get(miner_id),
                    "percentile": augmented_return_long_percentile.get(miner_id),
                }
            },
            "engagement": {
                "n_checkpoints": n_checkpoints.get(miner_id),
                "checkpoint_durations": checkpoint_durations.get(miner_id),
                "volume_threshold_count": volume_threshold_count.get(miner_id),
            },
            "plagiarism": plagiarism.get(miner_id),
            "cumulative": miner_cumulative_returns,
            "checkpoints": {
                "standard": miner_standard_ledger.cps,
                "returns_short_augmented": miner_returns_short_ledger.cps,
                "returns_long_augmented": miner_returns_long_ledger.cps,
                "risk_augmented": miner_risk_ledger.cps,
            }
        }
        combined_data.append(miner_data)
    
    final_dict = {
        'version': ValiConfig.VERSION,
        'created_timestamp_ms': time_now,
        'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
        "constants":{
            "return_short_cps_weight": ValiConfig.SCORING_RETURN_CPS_SHORT_WEIGHT,
            "return_long_cps_weight": ValiConfig.SCORING_RETURN_CPS_LONG_WEIGHT,
            "omega_cps_weight": ValiConfig.SCORING_OMEGA_CPS_WEIGHT,
            "inverted_sortino_cps_weight": ValiConfig.SCORING_SORTINO_CPS_WEIGHT,
            "set_weight_lookback_range_days": ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS,
            "set_weight_minimum_positions": ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS,
            "set_weight_minimum_position_duration_ms": ValiConfig.SET_WEIGHT_MINIMUM_POSITION_DURATION_MS,
            "omega_minimum_denominator": ValiConfig.OMEGA_MINIMUM_DENOMINATOR,
            "omega_ratio_threshold": ValiConfig.OMEGA_RATIO_THRESHOLD,
            "probabilistic_sharpe_ratio_threshold": ValiConfig.PROBABILISTIC_LOG_SHARPE_RATIO_THRESHOLD,
            "annual_risk_free_rate": ValiConfig.ANNUAL_RISK_FREE_RATE,
            "lookback_range_days_risk_free_rate": ValiConfig.LOOKBACK_RANGE_DAYS_RISK_FREE_RATE,
            "max_total_drawdown": ValiConfig.MAX_TOTAL_DRAWDOWN,
            "max_daily_drawdown": ValiConfig.MAX_DAILY_DRAWDOWN,
            "checkpoint_length_threshold": ValiConfig.CHECKPOINT_LENGTH_THRESHOLD,
            "checkpoint_duration_threshold": ValiConfig.CHECKPOINT_DURATION_THRESHOLD,
            "challengeperiod_volume_checkpoints": ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_VOLUME_CHECKPOINTS,
            "challengeperiod_volume_threshold": ValiConfig.CHECKPOINT_VOLUME_THRESHOLD,
            "challengeperiod_gracevalue": ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT,
            "challengeperiod_return_minimum": ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_RETURN_CPS_PERCENT,
            "challengeperiod_omega_minimum": ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_OMEGA_CPS,
            "challengeperiod_sortino_minimum": ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_SORTINO_CPS,
        },
        "data": combined_data,
    }

    output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "minerstatistics.json"
    ValiBkpUtils.write_file(
        output_file_path,
        final_dict,
    )