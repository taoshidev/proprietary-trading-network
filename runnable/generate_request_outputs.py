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
from vali_objects.scoring.scoring import Scoring
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager


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

    try:
        all_miner_hotkeys:list = ValiBkpUtils.get_directories_in_dir(
            ValiBkpUtils.get_miner_dir()
        )
    except FileNotFoundError:
        raise Exception(
            f"directory for miners doesn't exist "
            f"[{ValiBkpUtils.get_miner_dir()}]. Skip run for now."
        )
    
    # This one is retroactive, so we aren't modifying the eliminations on disk in the outputs file
    challengeperiod_miners = []
    challengeperiod_eliminations = []

    full_hotkey_positions = position_manager.get_all_miner_positions_by_hotkey(
        all_miner_hotkeys,
        sort_positions=True,
        eliminations=eliminations,
        acceptable_position_end_ms=None,
    )

    for original_hotkey, original_positions in full_hotkey_positions.items():
        challenge_check_logic = subtensor_weight_setter._challengeperiod_check(original_positions, time_now)
        if challenge_check_logic is False:
            challengeperiod_eliminations.append(original_hotkey)
        if challenge_check_logic is None:
            challengeperiod_miners.append(original_hotkey)

    omitted_miners = challengeperiod_miners + challengeperiod_eliminations

    # Perf Ledger Calculations
    perf_ledgers = PerfLedgerManager.load_perf_ledgers_from_disk() if write_validator_checkpoint else {}
    ledger = copy.deepcopy(perf_ledgers)

    omega_cps = {}
    inverted_sortino_cps = {}
    consistency_penalties = {}
    augmented_ledger = {}
    for hotkey, miner_ledger in ledger.items():
        if omitted_miners is not None and hotkey in omitted_miners:
            continue

        if hotkey in eliminated_hotkeys:
            continue

        checkpoint_meets_criteria = subtensor_weight_setter._filter_checkpoint_list(miner_ledger.cps)
        if not checkpoint_meets_criteria:
            continue

        augmented_ledger[hotkey] = miner_ledger
        augmented_ledger[hotkey].cps = position_manager.augment_perf_checkpoint(
            miner_ledger.cps,
            time_now
        )

        # Consistency penalty
        consistency_penalty = PositionUtils.compute_consistency_penalty_cps(
            miner_ledger.cps,
            time_now
        )
        consistency_penalties[hotkey] = consistency_penalty

        gains = [cp.gain for cp in augmented_ledger[hotkey].cps]
        losses = [cp.loss for cp in augmented_ledger[hotkey].cps]
        n_updates = [cp.n_updates for cp in augmented_ledger[hotkey].cps]
        open_durations = [cp.open_ms for cp in augmented_ledger[hotkey].cps]

        # Omega
        omega_cps[hotkey] = Scoring.omega_cps(
            gains,
            losses,
            n_updates,
            open_durations
        )

        # Inverted Sortino
        inverted_sortino_cps[hotkey] = Scoring.inverted_sortino_cps(
            gains,
            losses,
            n_updates,
            open_durations
        )

    checkpoint_results = Scoring.compute_results_checkpoint(augmented_ledger)
    challengeperiod_scores = [ (x, ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT) for x in challengeperiod_miners ]
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
        k not in challengeperiod_eliminations and
        k not in challengeperiod_miners
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
        'challengeperiod_miners': challengeperiod_miners,
        'eliminations': eliminations,
        'plagiarism': plagiarism,
        'youngest_order_processed_ms': youngest_order_processed_ms,
        'oldest_order_processed_ms': oldest_order_processed_ms,
        'positions': ord_dict_hotkey_position_map,
        'perf_ledgers': perf_ledgers
    }

    if write_validator_checkpoint:
        output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "validator_checkpoint.json"
        #logger.debug("Writing to output_file_path:" + output_file_path)
        ValiBkpUtils.write_file(
            output_file_path,
            final_dict,
        )
        logger.info(f"backed up validator checkpoint to {output_file_path}")

    if write_legacy:
        # Support for legacy output file
        legacy_output_file_path = ValiConfig.BASE_DIR + "/validation/outputs/output.json"
        ValiBkpUtils.write_file(
            legacy_output_file_path,
            ord_dict_hotkey_position_map,
        )

if __name__ == "__main__":
    logger = LoggerUtils.init_logger("generate_request_outputs")
    last_legacy_write_time = time.time()
    last_validator_checkpoint_time = time.time()
    try:
        while True:
            current_time = time.time()
            write_legacy = False
            write_validator_checkpoint = False
            msg = ""
            # Check if it's time to write the legacy output
            if current_time - last_legacy_write_time >= 15:
                msg += "Writing output.json. "
                write_legacy = True

            # Check if it's time to write the validator checkpoint
            if current_time - last_validator_checkpoint_time >= 180:
                msg += "Writing validator validator_checkpoint.json. "
                write_validator_checkpoint = True

            if write_legacy or write_validator_checkpoint:
                logger.info(msg)
                # Generate the request outputs
                generate_request_outputs(write_legacy=write_legacy, write_validator_checkpoint=write_validator_checkpoint)

            if write_legacy:
                last_legacy_write_time = current_time
            if write_validator_checkpoint:
                last_validator_checkpoint_time = current_time

            # Log completion duration
            if write_legacy or write_validator_checkpoint:
                logger.info("Completed writing outputs in " + str(time.time() - current_time) + " seconds.")
    except (JSONDecodeError, ValiBkpCorruptDataException) as e:
        logger.error("error occurred trying to decode position json. Probably being written to simultaneously.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    # Sleep for a short time to prevent tight looping, adjust as necessary
    time.sleep(1)
