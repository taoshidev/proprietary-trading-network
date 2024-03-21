import json
import traceback
import uuid
from datetime import datetime
import time
import logging

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
    try:
        eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )
    except Exception:
        logger.warning("couldn't get eliminations file.")
        eliminations = None
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
        
        positionmanager = PositionManager(
            config=None, 
            metagraph=all_miner_hotkeys, 
            running_unit_tests=False
        )

        hotkey_positions = positionmanager.get_all_miner_positions_by_hotkey(
            all_miner_hotkeys,
            sort_positions=True,
            acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                TimeUtil.generate_start_timestamp(
                    ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
                )
            ),
        )

        dict_hotkey_position_map = {}

        for k, ps in hotkey_positions.items():
            dict_hotkey_position_map[k] = {
                "positions": [],
                "thirty_day_returns": 1.0,
            }

            return_per_position = positionmanager.get_return_per_closed_position(ps)

            ## also get the augmented returns
            return_per_position_augmented = positionmanager.get_return_per_closed_position_augmented(
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
                if eliminations is not None and k in eliminations:
                    if p.is_closed_position is False:
                        logger.warning(
                            "position was not closed. Will check last order and "
                            "see if its closed. If not, will note and add."
                        )
                        if p.orders[len(p.orders) - 1].order_type != OrderType.FLAT:
                            logger.warning("order was not closed. Will add close.")
                            # price shouldn't matter, the miner already added to elims
                            # price is only used for return purposes which would be irrelevant
                            p.orders.append(
                                Order(
                                    OrderType.FLAT,
                                    0.0,
                                    0.0,
                                    p.trade_pair,
                                    TimeUtil.now_in_millis(),
                                    str(uuid.uuid4()),
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

        ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_outputs_dir())

        ValiBkpUtils.write_file(
            ValiBkpUtils.get_vali_outputs_dir() + "output.json",
            ord_dict_hotkey_position_map,
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
