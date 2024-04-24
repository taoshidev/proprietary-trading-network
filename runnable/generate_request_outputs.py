import json
import os
import traceback
import uuid
import time
from json import JSONDecodeError
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.scoring.scoring import Scoring

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
    consistency_penalties = {}

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

                if len(return_per_position_augmented) > 0:
                    consistency_penalty = PositionUtils.compute_consistency_penalty(
                        ps_30_days, time_now
                    )
                    consistency_penalties[k] = consistency_penalty

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

    scaled_transformed_list = Scoring.transform_and_scale_results(filtered_results)
    challengeperiod_weights = [ (x, ValiConfig.SET_WEIGHT_MINER_GRACE_PERIOD_VALUE) for x in challengeperiod_miners ]

    # going to just overwrite the computed weights with the challengeperiod weights
    transformed_list =  scaled_transformed_list + challengeperiod_weights
    transformed_dict = dict(transformed_list)

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
        'weights': transformed_dict,
        'consistency_penalties': consistency_penalties,
        "metrics": {
            "omega": omega_list,
            "augmented_return": augmented_return_list,
            "sharpe_ratio": sharpe_ratio_list,
            "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio_list,
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
    }

    if write_validator_checkpoint:
        output_file_path = ValiBkpUtils.get_vali_outputs_dir() + f"validator_checkpoint_{int(time.time())}.json"
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


def manage_checkpoint_files(base_dir, current_time):
    checkpoint_dir = ValiBkpUtils.get_vali_outputs_dir()

    twenty_four_hours_ago = current_time - 86400
    forty_eight_hours_ago = current_time - 172800

    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("validator_checkpoint_")
    ]

    newest_valid_checkpoint = None
    newest_valid_time = 0
    for filename in checkpoint_files:
        filepath = os.path.join(checkpoint_dir, filename)
        filetime = int(filename.split("_")[-1].split(".")[0])

        if filetime < forty_eight_hours_ago:
            os.remove(filepath)
        elif filetime < twenty_four_hours_ago and filetime > newest_valid_time:
            newest_valid_checkpoint = filepath
            newest_valid_time = filetime

    if newest_valid_checkpoint:
        main_checkpoint_path = os.path.join(base_dir, "validator_checkpoint.json")
        with open(newest_valid_checkpoint, 'r') as new_file, open(main_checkpoint_path, 'w') as main_file:
            json_data = json.load(new_file)
            json.dump(json_data, main_file)


if __name__ == "__main__":
    base_dir = ValiConfig.BASE_DIR
    last_legacy_write_time = time.time()
    last_validator_checkpoint_time = time.time()
    logger = LoggerUtils.init_logger("generate_request_outputs")
    try:
        while True:
            current_time = time.time()
            write_legacy = False
            write_validator_checkpoint = False
            msg = ""

            if current_time - last_legacy_write_time >= 15:
                msg += "Writing output.json. "
                write_legacy = True

            if current_time - last_validator_checkpoint_time >= 180:
                msg += "Writing validator validator_checkpoint.json. "
                write_validator_checkpoint = True

            if write_legacy or write_validator_checkpoint:
                logger.info(msg)
                generate_request_outputs(write_legacy, write_validator_checkpoint)

            manage_checkpoint_files(base_dir, current_time)

            if write_legacy:
                last_legacy_write_time = current_time
            if write_validator_checkpoint:
                last_validator_checkpoint_time = current_time

            logger.info("Completed writing outputs in " + str(time.time() - current_time) + " seconds.")
            time.sleep(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        time.sleep(1)
