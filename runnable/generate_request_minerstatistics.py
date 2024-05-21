import json
import traceback
import uuid
import time
import copy
import math

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

    # Convert to readable format
    challengeperiod_testing_readable = {k: TimeUtil.millis_to_formatted_date_str(v) for k, v in sorted_challengeperiod_testing.items()}
    challengeperiod_success_readable = {k: TimeUtil.millis_to_formatted_date_str(v) for k, v in sorted_challengeperiod_success.items()}

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
    for hotkey, hotkey_ledger in filtered_ledger.items():
        consistency_penalty = PositionUtils.compute_consistency_penalty_cps(hotkey_ledger.cps)
        consistency_penalties[hotkey] = consistency_penalty

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
    return_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS
    risk_adjusted_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RISKMETRIC

    augmented_omega_cps = {}
    augmented_inverted_sortino_cps = {}
    augmented_return_cps = {}

    returns_ledger = PositionManager.augment_perf_ledger(
        ledger=filtered_ledger,
        evaluation_time_ms=time_now,
        decay_coefficient=return_decay_coefficient
    )

    risk_adjusted_ledger = PositionManager.augment_perf_ledger(
        ledger=filtered_ledger,
        evaluation_time_ms=time_now,
        decay_coefficient=risk_adjusted_decay_coefficient
    )

    for hotkey, miner_ledger in returns_ledger.items():
        scoringunit = ScoringUnit.from_perf_ledger(miner_ledger)
        augmented_return_cps[hotkey] = Scoring.return_cps(scoringunit)

    for hotkey, miner_ledger in risk_adjusted_ledger.items():
        scoringunit = ScoringUnit.from_perf_ledger(miner_ledger)
        augmented_omega_cps[hotkey] = Scoring.omega_cps(scoringunit)
        augmented_inverted_sortino_cps[hotkey] = Scoring.inverted_sortino_cps(scoringunit)

    ## This is when we only want to look at the successful miners
    successful_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=challengeperiod_success_hotkeys)
    checkpoint_results = Scoring.compute_results_checkpoint(successful_ledger, evaluation_time_ms=time_now)
    challengeperiod_scores = [ (x, ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_testing_hotkeys ]
    scoring_results = checkpoint_results + challengeperiod_scores
    
    final_dict = {
        'version': ValiConfig.VERSION,
        'created_timestamp_ms': time_now,
        'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
        'weights': dict(scoring_results),
        'challengeperiod': {
            "testing": challengeperiod_testing_readable,
            "success": challengeperiod_success_readable
        },
        'penalties': {
            "consistency": consistency_penalties,
        },
        "metrics": {
            "omega_cps": omega_cps,
            "inverted_sortino_cps": inverted_sortino_cps,
            "return_cps": return_cps,
            "n_checkpoints": n_checkpoints,
            "checkpoint_durations": checkpoint_durations,
            "volume_threshold_count": volume_threshold_count,
        },
        "augmented_metrics": {
            "augmented_omega_cps": augmented_omega_cps,
            "augmented_inverted_sortino_cps": augmented_inverted_sortino_cps,
            "augmented_return_cps": augmented_return_cps
        },
        "constants":{
            "return_cps_weight": ValiConfig.SCORING_RETURN_CPS_WEIGHT,
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
        },
        'plagiarism': plagiarism,
    }

    output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "minerstatistics.json"
    ValiBkpUtils.write_file(
        output_file_path,
        final_dict,
    )