import json
import traceback
from datetime import datetime
import time

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


def generate_request_outputs():
    eliminations = ValiUtils.get_vali_json_file(
        ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
    )
    try:
        try:
            all_miner_hotkeys = ValiBkpUtils.get_directories_in_dir(
                ValiBkpUtils.get_miner_dir()
            )
        except FileNotFoundError:
            raise Exception(
                f"directory for miners doesn't exist "
                f"[{ValiBkpUtils.get_miner_dir()}]. Skip run for now."
            )

        hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
            all_miner_hotkeys,
            sort_positions=True,
            eliminations=eliminations,
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
            return_per_position = PositionUtils.get_return_per_closed_position(ps)
            if len(return_per_position) > 0:
                curr_return = return_per_position[len(return_per_position) - 1]
                dict_hotkey_position_map[k]["thirty_day_returns"] = curr_return

            for p in ps:
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
        print("successfully outputted request output.")
    except Exception:
        print("error occurred trying generate request outputs.")
        print(traceback.format_exc())


if __name__ == "__main__":
    print("generate request outputs")
    while True:
        now = datetime.utcnow()
        if True:
            print(f"{now}: outputting request output")
            generate_request_outputs()
            time.sleep(15)
