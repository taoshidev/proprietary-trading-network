import json
import traceback
import uuid
from datetime import datetime
import time

from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order


def generate_request_outputs():
    position_manager = PositionManager(
        config=None,
        metagraph=None,
        running_unit_tests=False
    )
    def get_eliminations_from_disk():
        location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=False)
        cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
        logger.info(f"Loaded [{len(cached_eliminations)}] eliminations from disk: {cached_eliminations}. Dir: {location}")
        return cached_eliminations

    def get_plagiarism_scores_from_disk():
        location = ValiBkpUtils.get_plagiarism_scores_file_location(running_unit_tests=False)
        ans = ValiUtils.get_vali_json_file(location)
        logger.info(f"Loaded [{len(ans)}] plagiarism scores from disk: {ans}. Dir: {location}")
        return ans
    
    eliminations = get_eliminations_from_disk()
    eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
    plagiarism = get_plagiarism_scores_from_disk()

    try:
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

        dict_hotkey_position_map = {}
        youngest_order_processed_ms = float("inf")
        oldest_order_processed_ms = 0
        for k, ps in hotkey_positions.items():
            dict_hotkey_position_map[k] = {
                "positions": [],
                "thirty_day_returns": 1.0,
            }

            return_per_position = position_manager.get_return_per_closed_position(ps)

            ## also get the augmented returns
            return_per_position_augmented = position_manager.get_return_per_closed_position_augmented(
                ps,
                evaluation_time_ms=TimeUtil.now_in_millis(),
            )

            if len(return_per_position) > 0:
                curr_return = return_per_position[len(return_per_position) - 1]
                dict_hotkey_position_map[k]["thirty_day_returns"] = curr_return

            if len(return_per_position_augmented) > 0:
                curr_return_augmented = return_per_position_augmented[
                    len(return_per_position_augmented) - 1
                ]
                dict_hotkey_position_map[k][
                    "thirty_day_returns_augmented"
                ] = curr_return_augmented

            for p in ps:
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

        ord_dict_hotkey_position_map = dict(
            sorted(
                dict_hotkey_position_map.items(),
                key=lambda item: item[1]["thirty_day_returns"],
                reverse=True,
            )
        )

        now_ms = TimeUtil.now_in_millis()
        final_dict = {
            'positions': ord_dict_hotkey_position_map,
            'eliminations': eliminations,
            'plagiarism': plagiarism,
            'youngest_order_processed_ms': youngest_order_processed_ms,
            'oldest_order_processed_ms': oldest_order_processed_ms,
            'created_timestamp_ms': now_ms,
            'created_date': TimeUtil.millis_to_datetime(now_ms).strftime("%Y-%m-%d %H:%M:%S"),
        }

        output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "validator_checkpoint.json"
        logger.info("Writing to output_file_path:" + output_file_path)
        ValiBkpUtils.write_file(
            output_file_path,
            final_dict,
        )
        logger.info("successfully outputted request output.")
    except Exception:
        logger.error("error occurred trying generate request outputs.")
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    logger = LoggerUtils.init_logger("generate_request_outputs")
    logger.info("generate request outputs")
    while True:
        now = datetime.utcnow()
        if True:
            logger.info(f"{now}: outputting request output")
            generate_request_outputs()
            time.sleep(15)
