import json
import os
from datetime import datetime

from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import bittensor as bt

def get_file(f):
    output_json_path = os.path.abspath(os.path.join(f))
    print('path:', output_json_path)
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as file:
            data = file.read()
        return json.loads(data)
    else:
        return None


def regenerate_miner_positions():
    miner_positions = "validator_checkpoint.json"
    data = get_file(miner_positions)
    if data is None:
        bt.logging.warning(f"necessary file doesn't exist [{miner_positions}]")
        return False

    bt.logging.info(f"Found validator backup file with 'created_date' {data['created_date']} and 'created_timestamp_ms' {data['created_timestamp_ms']}")

    position_manager = PositionManager()
    # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
    # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup youngest timestamp is
    # older than the local youngest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
    youngest_disk_ms, oldest_disk_ms = (
        position_manager.get_extreme_position_order_processed_on_disk_ms())
    youngest_backup_ms = data['youngest_order_processed_ms']
    oldest_backup_ms = data['oldest_order_processed_ms']
    bt.logging.info(f"youngest_disk_order_timestamp: {youngest_disk_ms}, "
          f"youngest_backup_order_timestamp: {youngest_backup_ms}, "
          f"oldest_disk_order_timestamp: {oldest_disk_ms}, "
          f"oldest_backup_order_timestamp: {oldest_backup_ms}")

    if youngest_disk_ms >= youngest_backup_ms and oldest_disk_ms <= oldest_backup_ms:
        pass  # Ready for update!
    elif oldest_disk_ms > oldest_backup_ms:
        bt.logging.error("Please re-pull the backup file before restoring. Backup appears to be older than the disk.")
        return False
    elif youngest_disk_ms < youngest_backup_ms:
        bt.logging.error("Your local filesystem has older orders than the backup. Please reach out to the team ASAP before regenerating. You may be holding irrecoverable data!")
        return False
    else:
        bt.logging.error("Problem with backup file detected. Please reach out to the team ASAP")
        return False


    n_existing_position = position_manager.get_number_of_miners_with_any_positions()
    n_existing_eliminations = position_manager.get_number_of_eliminations()
    n_existing_plagiarism_scores = position_manager.get_number_of_plagiarism_scores()
    msg = (f"Detected {n_existing_position} hotkeys with positions, {n_existing_eliminations} eliminations, "
           f"and {n_existing_plagiarism_scores} plagiarism scores.")
    bt.logging.info(msg)
    if n_existing_position or n_existing_eliminations or n_existing_plagiarism_scores:
        msg = ("This script will overwrite all existing positions, eliminations, and plagiarism scores. "
               "Are you sure you want to continue? (y/n)\n")
        bt.logging.warning(msg)
        if input(msg).lower() != 'y':
            return False

    bt.logging.info(f"regenerating {len(data['positions'].keys())} hotkeys")
    position_manager.clear_all_miner_positions_from_disk()
    for hotkey, json_positions in data['positions'].items():
        # sort positions by close_ms otherwise, writing a closed position after an open position for the same
        # trade pair will delete the open position
        positions = [Position(**json_positions_dict) for json_positions_dict in json_positions['positions']]
        assert len(positions) > 0, f"no positions for hotkey {hotkey}"
        positions.sort(key=position_manager.sort_by_close_ms)
        ValiBkpUtils.make_dir(ValiBkpUtils.get_miner_all_positions_dir(hotkey))
        for p_obj in positions:
            bt.logging.info(f'creating position {p_obj}')
            position_manager.save_miner_position_to_disk(p_obj)

        # Validate that the positions were written correctly
        disk_positions = position_manager.get_all_miner_positions(hotkey, sort_positions=True)
        bt.logging.info(f'disk_positions: {disk_positions}, positions: {positions}')
        assert disk_positions == positions, f"disk_positions: {disk_positions}, positions: {positions}"

    bt.logging.info(f"regenerating {len(data['eliminations'])} eliminations")
    position_manager.write_eliminations_to_disk(data['eliminations'])

    bt.logging.info(f"regenerating {len(data['plagiarism'])} plagiarism scores")
    position_manager.write_plagiarism_scores_to_disk(data['plagiarism'])
    return True

if __name__ == "__main__":
    bt.logging.info("regenerating miner positions")
    now = datetime.utcnow()
    bt.logging.info(f"{now}: request timestamp")
    passed = regenerate_miner_positions()
    if passed:
        bt.logging.info("regeneration complete")
    else:
        bt.logging.error("regeneration failed")
