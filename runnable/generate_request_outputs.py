import json
import traceback
import uuid
import time
import copy
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


def generate_request_outputs(write_legacy:bool, write_validator_checkpoint:bool):
    # let's set the time first to try and make it as close as possible to the original
    time_now = TimeUtil.now_in_millis()

    position_manager = PositionManager(
        config=None,
        metagraph=None,
        running_unit_tests=False
    )

    subtensor_weight_setter = SubtensorWeightSetter(
        config=None,
        wallet=None,
        metagraph=None,
        running_unit_tests=False
    )

    eliminations = position_manager.get_eliminations_from_disk()
    eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
    plagiarism = position_manager.get_plagiarism_scores_from_disk()

    ## Collect information from the disk and populate variables in memory
    subtensor_weight_setter._refresh_eliminations_in_memory()
    subtensor_weight_setter._refresh_challengeperiod_in_memory()

    challengeperiod_testing_dictionary = subtensor_weight_setter.challengeperiod_testing
    challengeperiod_testing_hotkeys = list(challengeperiod_testing_dictionary.keys())

    challengeperiod_success_dictionary = subtensor_weight_setter.challengeperiod_success
    challengeperiod_success_hotkeys = list(challengeperiod_success_dictionary.keys())
    
    challengeperiod_eliminated_hotkeys = [ x['hotkey'] for x in subtensor_weight_setter.eliminations ]

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
    perf_ledgers = PerfLedgerManager.load_perf_ledgers_from_disk() if write_validator_checkpoint else {}

    omega_cps = {}
    inverted_sortino_cps = {}
    return_cps = {}
    consistency_penalties = {}

    filtered_ledger = subtensor_weight_setter.filtered_ledger(hotkeys=challengeperiod_success_hotkeys)

    return_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RETURNS
    risk_adjusted_decay_coefficient = ValiConfig.HISTORICAL_DECAY_COEFFICIENT_RISKMETRIC

    ## consistency penalties
    consistency_penalties = {}
    for hotkey, hotkey_ledger in filtered_ledger.items():
        consistency_penalty = PositionUtils.compute_consistency_penalty_cps(hotkey_ledger.cps)
        consistency_penalties[hotkey] = consistency_penalty

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
        return_cps[hotkey] = Scoring.return_cps(scoringunit)

    for hotkey, miner_ledger in risk_adjusted_ledger.items():
        scoringunit = ScoringUnit.from_perf_ledger(miner_ledger)
        omega_cps[hotkey] = Scoring.omega_cps(scoringunit)
        inverted_sortino_cps[hotkey] = Scoring.inverted_sortino_cps(scoringunit)

    checkpoint_results = Scoring.compute_results_checkpoint(filtered_ledger, evaluation_time_ms=time_now)
    challengeperiod_scores = [ (x, ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_testing_hotkeys ]
    scoring_results = checkpoint_results + challengeperiod_scores

    # we won't be able to query for eliminated hotkeys from challenge period
    hotkey_positions = position_manager.get_all_miner_positions_by_hotkey(
        all_miner_hotkeys,
        sort_positions=True
    )

    acceptable_position_end_ms = TimeUtil.timestamp_to_millis(
        TimeUtil.generate_start_timestamp(
            ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
        ))

    time_now = TimeUtil.now_in_millis()
    dict_hotkey_position_map = {}

    youngest_order_processed_ms = float("inf")
    oldest_order_processed_ms = 0
    for k, original_positions in hotkey_positions.items():
        dict_hotkey_position_map[k] = {
            "positions": [],
            "thirty_day_returns": 1.0,
        }
        positions_30_days = [
            position
            for position in original_positions
            if position.open_ms > acceptable_position_end_ms
        ]

        if k not in eliminated_hotkeys:
            ps_30_days = subtensor_weight_setter._filter_positions(positions_30_days)
            filter_miner_boolean = subtensor_weight_setter._filter_miner(ps_30_days, time_now)

            return_per_position = position_manager.get_return_per_closed_position(ps_30_days)
            if len(return_per_position) > 0:
                curr_return = return_per_position[len(return_per_position) - 1]
                dict_hotkey_position_map[k]["thirty_day_returns"] = curr_return

            if not filter_miner_boolean:
                ## also get the augmented returns
                return_per_position_augmented: list[float] = position_manager.get_return_per_closed_position_augmented(
                    ps_30_days,
                    evaluation_time_ms=time_now,
                )

                if len(return_per_position_augmented) > 0:
                    curr_return_augmented = return_per_position_augmented
                    dict_hotkey_position_map[k][
                        "thirty_day_returns_augmented"
                    ] = curr_return_augmented

        for p in original_positions:
            youngest_order_processed_ms = min(youngest_order_processed_ms,
                                              min(p.orders, key=lambda o: o.processed_ms).processed_ms)
            oldest_order_processed_ms = max(oldest_order_processed_ms,
                                            max(p.orders, key=lambda o: o.processed_ms).processed_ms)
            if p.close_ms is None:
                p.close_ms = 0
            dict_hotkey_position_map[k]["positions"].append(
                json.loads(str(p), cls=GeneralizedJSONDecoder)
            )

    miner_finalscores = {
        k: v['thirty_day_returns_augmented']
        for k, v in dict_hotkey_position_map.items()
        if 'thirty_day_returns_augmented' in v and
        k not in eliminated_hotkeys and
        k not in challengeperiod_eliminated_hotkeys and
        k not in challengeperiod_success_hotkeys
    }

    filtered_results = list(miner_finalscores.items())
    ## Can also start tracking some of the other metrics here
    omega_list = {}
    augmented_return_list = {}
    sharpe_ratio_list = {}
    probabilistic_sharpe_ratio_list = {}

    for miner_id, returns in filtered_results:
        if miner_id in eliminated_hotkeys:
            continue

        omega_list[miner_id] = Scoring.omega(returns)
        augmented_return_list[miner_id] = Scoring.total_return(returns)
        sharpe_ratio_list[miner_id] = Scoring.sharpe_ratio(returns)
        probabilistic_sharpe_ratio_list[miner_id] = Scoring.probabilistic_sharpe_ratio(returns)

    ord_dict_hotkey_position_map = dict(
        sorted(
            dict_hotkey_position_map.items(),
            key=lambda item: item[1]["thirty_day_returns"],
            reverse=True,
        )
    )

    n_orders_original = 0
    for positions in hotkey_positions.values():
        n_orders_original += sum([len(position.orders) for position in positions])

    n_positions_new = 0
    for data in ord_dict_hotkey_position_map.values():
        positions = data['positions']
        n_positions_new += sum([len(p['orders']) for p in positions])

    assert n_orders_original == n_positions_new, f"n_orders_original: {n_orders_original}, n_positions_new: {n_positions_new}"
    now_ms = time_now
    final_dict = {
        'version': ValiConfig.VERSION,
        'created_timestamp_ms': now_ms,
        'created_date': TimeUtil.millis_to_formatted_date_str(now_ms),
        'weights': dict(scoring_results),
        'consistency_penalties': consistency_penalties,
        "metrics": {
            "omega": omega_list,
            "augmented_return": augmented_return_list,
            "sharpe_ratio": sharpe_ratio_list,
            "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio_list,
            "omega_cps": omega_cps,
            "inverted_sortino_cps": inverted_sortino_cps,
            "return_cps": return_cps
        },
        "constants":{
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
        },
        'challengeperiod': {
            "testing": challengeperiod_testing_dictionary,
            "success": challengeperiod_success_dictionary
        },
        'eliminations': eliminations,
        'plagiarism': plagiarism,
        'youngest_order_processed_ms': youngest_order_processed_ms,
        'oldest_order_processed_ms': oldest_order_processed_ms,
        'positions': ord_dict_hotkey_position_map,
        'perf_ledgers': perf_ledgers
    }

    paths = []
    if write_validator_checkpoint:
        output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "validator_checkpoint.json"
        #logger.debug("Writing to output_file_path:" + output_file_path)
        ValiBkpUtils.write_file(
            output_file_path,
            final_dict,
        )
        paths.append(output_file_path)

    if write_legacy:
        # Support for legacy output file
        legacy_output_file_path = ValiConfig.BASE_DIR + "/validation/outputs/output.json"
        ValiBkpUtils.write_file(
            legacy_output_file_path,
            ord_dict_hotkey_position_map,
        )
        paths.append(legacy_output_file_path)

    #logger.info(f"backed up data to {paths}")


if __name__ == "__main__":
    logger = LoggerUtils.init_logger("generate_request_outputs")
    last_write_time = time.time()
    n_updates = 0
    try:
        while True:
            current_time = time.time()
            write_legacy = False
            write_validator_checkpoint = False
            # Check if it's time to write the legacy output
            if current_time - last_write_time >= 15:
                write_legacy = True
                write_validator_checkpoint = True

            if write_legacy or write_validator_checkpoint:
                # Generate the request outputs
                generate_request_outputs(write_legacy=write_legacy,
                                         write_validator_checkpoint=write_validator_checkpoint)

            if write_legacy:
                last_legacy_write_time = current_time
            if write_validator_checkpoint:
                last_validator_checkpoint_time = current_time

            # Log completion duration
            if write_legacy or write_validator_checkpoint:
                n_updates += 1
                if n_updates % 10 == 0:
                    logger.info("Completed writing outputs in " + str(time.time() - current_time) + " seconds. n_updates: " + str(n_updates))
    except (JSONDecodeError, ValiBkpCorruptDataException) as e:
        logger.error("error occurred trying to decode position json. Probably being written to simultaneously.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    # Sleep for a short time to prevent tight looping, adjust as necessary
    time.sleep(1)
