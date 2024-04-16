import json
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

    hotkey_positions = position_manager.get_all_miner_positions_by_hotkey(
        all_miner_hotkeys,
        sort_positions=True,
        acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
            TimeUtil.generate_start_timestamp(
                ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
            )
        ),
    )

    time_now = TimeUtil.now_in_millis()
    dict_hotkey_position_map = {}
    graceperiod_miners = set()
    consistency_penalties = {}

    youngest_order_processed_ms = float("inf")
    oldest_order_processed_ms = 0
    for k, original_positions in hotkey_positions.items():
        dict_hotkey_position_map[k] = {
            "positions": [],
            "thirty_day_returns": 1.0,
        }

        if k not in eliminated_hotkeys:
            ps = subtensor_weight_setter._filter_positions(original_positions)
            filter_miner_boolean = subtensor_weight_setter._filter_miner(ps, time_now)

            return_per_position = position_manager.get_return_per_closed_position(ps)
            if len(return_per_position) > 0:
                curr_return = return_per_position[len(return_per_position) - 1]
                dict_hotkey_position_map[k]["thirty_day_returns"] = curr_return

            if not filter_miner_boolean:
                ## also get the augmented returns
                return_per_position_augmented: list[float] = position_manager.get_return_per_closed_position_augmented(
                    ps,
                    evaluation_time_ms=TimeUtil.now_in_millis(),
                )

                if len(return_per_position_augmented) > 0:
                    curr_return_augmented = return_per_position_augmented
                    dict_hotkey_position_map[k][
                        "thirty_day_returns_augmented"
                    ] = curr_return_augmented

                if len(return_per_position_augmented) > 0:
                    consistency_penalty = PositionUtils.compute_consistency_penalty(
                        ps, time_now
                    )
                    consistency_penalties[k] = consistency_penalty

                if len(return_per_position_augmented) < ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
                    graceperiod_miners.add(k)

        for p in original_positions:
            youngest_order_processed_ms = min(youngest_order_processed_ms,
                                              min(p.orders, key=lambda o: o.processed_ms).processed_ms)
            oldest_order_processed_ms = max(oldest_order_processed_ms,
                                            max(p.orders, key=lambda o: o.processed_ms).processed_ms)

            if k in eliminated_hotkeys:
                if p.is_open_position:
                    logger.warning(
                        "This should not happen anymore. Please alert the team if you see this."
                        "position was not closed. Will check last order and "
                        "see if its closed. If not, will note and add."
                    )
                    if p.orders[len(p.orders) - 1].order_type != OrderType.FLAT:
                        logger.warning("order was not closed. Will add close.")
                        # price shouldn't matter, the miner already added to elims
                        # price is only used for return purposes which would be irrelevant
                        p.orders.append(
                            Order(
                                order_type=OrderType.FLAT,
                                leverage=0.0,
                                price=0.0,
                                trade_pair=p.trade_pair,
                                processed_ms=TimeUtil.now_in_millis(),
                                order_uuid=str(uuid.uuid4()),
                            )

                        )

            if p.close_ms is None:
                p.close_ms = 0
            dict_hotkey_position_map[k]["positions"].append(
                json.loads(str(p), cls=GeneralizedJSONDecoder)
            )

    miner_finalscores = {
        k: v['thirty_day_returns_augmented']
        for k, v in dict_hotkey_position_map.items()
        if 'thirty_day_returns_augmented' in v
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

    scaled_transformed_list = dict(Scoring.transform_and_scale_results(filtered_results))
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

    #logger.debug(f"n_orders_original: {n_orders_original}, n_positions_new: {n_positions_new}")
    assert n_orders_original == n_positions_new, f"n_orders_original: {n_orders_original}, n_positions_new: {n_positions_new}"

    now_ms = TimeUtil.now_in_millis()
    final_dict = {
        'version': ValiConfig.VERSION,
        'created_timestamp_ms': now_ms,
        'created_date': TimeUtil.millis_to_formatted_date_str(now_ms),
        'weights': scaled_transformed_list,
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
        'graceperiod_miners': list(graceperiod_miners),
        'eliminations': eliminations,
        'plagiarism': plagiarism,
        'youngest_order_processed_ms': youngest_order_processed_ms,
        'oldest_order_processed_ms': oldest_order_processed_ms,
        'positions': ord_dict_hotkey_position_map,
    }

    if write_validator_checkpoint:
        output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "validator_checkpoint.json"
        #logger.debug("Writing to output_file_path:" + output_file_path)
        ValiBkpUtils.write_file(
            output_file_path,
            final_dict,
        )

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
    except JSONDecodeError:
        logger.error("error occurred trying to decode position json. Probably being written to simultaneously.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    # Sleep for a short time to prevent tight looping, adjust as necessary
    time.sleep(1)
