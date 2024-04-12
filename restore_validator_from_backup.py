import argparse
import json
import shutil
import time
from datetime import datetime

from vali_objects.position import Position
#from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import bittensor as bt

#from vali_objects.utils.vali_utils import ValiUtils


def backup_validation_directory():
    dir_to_backup = ValiBkpUtils.get_vali_dir()
    # Write to the backup location. Make sure it is a function of the date. No dashes. Days and months get 2 digits.
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_location = ValiBkpUtils.get_vali_bkp_dir() + date_str + '/'
    # Sync directory to the backup location using python shutil
    shutil.copytree(dir_to_backup, backup_location)
    bt.logging.info(f"backed up {dir_to_backup} to {backup_location}")


def regenerate_miner_positions(perform_backup=True):
    backup_file_path = "validator_checkpoint.json"
    try:
        data = json.loads(ValiBkpUtils.get_file(backup_file_path))
        if isinstance(data, str):  # TODO: Why is the data being double serialized?
            data = json.loads(data)
    except Exception as e:
        bt.logging.error(f"Unable to read validator checkpoint file. {e}")
        return False

    bt.logging.info(f"Found validator backup file with the following attributes:")
    # Log every key and value apir in the data except for positions, eliminations, and plagiarism scores
    for key, value in data.items():
        # Check is the value is of type dict or list. If so, print the size of the dict or list
        if isinstance(value, dict) or isinstance(value, list):
            # Log the size of the positions, eliminations, and plagiarism scores
            bt.logging.info(f"    {key}: {len(value)} entries")
        else:
            bt.logging.info(f"    {key}: {value}")

    # TESTING
    #secrets = ValiUtils.get_secrets()
    #live_price_fetcher = LivePriceFetcher(secrets=secrets)
    #position_manager = PositionManager(live_price_fetcher=live_price_fetcher)
    position_manager = PositionManager()
    position_manager.init_cache_files()
    #position_manager.perform_price_recalibration(time_per_batch_s=10000000)
    # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
    # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup smallest timestamp is
    # older than the local smallest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
    smallest_disk_ms, largest_disk_ms = (
        position_manager.get_extreme_position_order_processed_on_disk_ms())
    smallest_backup_ms = data['youngest_order_processed_ms']
    largest_backup_ms = data['oldest_order_processed_ms']
    bt.logging.info(f"smallest_disk_order_timestamp: {smallest_disk_ms}, "
          f"smallest_backup_order_timestamp: {smallest_backup_ms}, "
          f"oldest_disk_order_timestamp: {largest_disk_ms}, "
          f"oldest_backup_order_timestamp: {largest_backup_ms}")

    if smallest_disk_ms >= smallest_backup_ms and largest_disk_ms <= largest_backup_ms:
        pass  # Ready for update!
    elif largest_disk_ms > largest_backup_ms:
        bt.logging.error("Please re-pull the backup file before restoring. Backup appears to be older than the disk.")
        return False
    elif smallest_disk_ms < smallest_backup_ms:
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

    bt.logging.info("Backing up and overwriting all existing positions, eliminations, and plagiarism scores.")
    if perform_backup:
        backup_validation_directory()

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
            #bt.logging.info(f'creating position {p_obj}')
            position_manager.save_miner_position_to_disk(p_obj)

        # Validate that the positions were written correctly
        disk_positions = position_manager.get_all_miner_positions(hotkey, sort_positions=True)
        #bt.logging.info(f'disk_positions: {disk_positions}, positions: {positions}')
        n_disk_positions = len(disk_positions)
        n_memory_positions = len(positions)
        memory_p_uuids = set([p.position_uuid for p in positions])
        disk_p_uuids = set([p.position_uuid for p in disk_positions])
        assert n_disk_positions == n_memory_positions, f"n_disk_positions: {n_disk_positions}, n_memory_positions: {n_memory_positions}"
        assert memory_p_uuids == disk_p_uuids, f"memory_p_uuids: {memory_p_uuids}, disk_p_uuids: {disk_p_uuids}"


    bt.logging.info(f"regenerating {len(data['eliminations'])} eliminations")
    position_manager.write_eliminations_to_disk(data['eliminations'])

    bt.logging.info(f"regenerating {len(data['plagiarism'])} plagiarism scores")
    position_manager.write_plagiarism_scores_to_disk(data['plagiarism'])
    return True

if __name__ == "__main__":
    t0 = time.time()
    # Check commandline arg "disable_backup" to disable backup.
    parser = argparse.ArgumentParser(description="Regenerate miner positions with optional backup disabling.")
    # Add disable_backup argument, default is 0 (False), change type to int
    parser.add_argument('--backup', type=int, default=1,
                        help='Set to 0 to disable backup during regeneration process.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Use the disable_backup argument to control backup
    perform_backup = bool(args.backup)
    bt.logging.info("regenerating miner positions")
    if not perform_backup:
        bt.logging.warning("backup disabled")
    passed = regenerate_miner_positions(perform_backup)
    if passed:
        bt.logging.info("regeneration complete in %.2f seconds" % (time.time() - t0))
    else:
        bt.logging.error("regeneration failed")
