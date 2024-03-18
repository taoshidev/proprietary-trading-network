import json
import os
from datetime import datetime

from vali_objects.position import Position
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


def get_file(f):
    output_json_path = os.path.abspath(os.path.join(f))
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as file:
            data = file.read()
        return json.loads(data)
    else:
        return None


def regenerate_miner_positions():
    position_manager = PositionManager()
    miner_positions = "miner_positions.json"
    data = get_file(miner_positions)
    if data is None:
        logger.warning(f"necessary file doesn't exist [{miner_positions}]")
        return False
    for muid, all_ps in data.items():
        ValiBkpUtils.make_dir(ValiBkpUtils.get_miner_position_dir(muid))
        for p_dict in all_ps["positions"]:
            p = Position.from_dict(p_dict)
            position_manager.save_miner_position_to_disk(p)
    return True


if __name__ == "__main__":
    logger = LoggerUtils.init_logger("regen_miner_positions")
    logger.info("regenerating miner positions")
    now = datetime.utcnow()
    logger.info(f"{now}: request timestamp")
    passed = regenerate_miner_positions()
    if passed:
        logger.info("regeneration complete")
    else:
        logger.fatal("regeneration failed")
