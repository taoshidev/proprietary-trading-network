import argparse
import json
import shutil
import time
from datetime import datetime

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import bittensor as bt

from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

DEBUG = 0

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
        if isinstance(data, str):
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
    backup_creation_time_ms = data['created_timestamp_ms']

    if DEBUG:
        secrets = ValiUtils.get_secrets()
        live_price_fetcher = LivePriceFetcher(secrets=secrets)
        position_manager = PositionManager(live_price_fetcher=live_price_fetcher, perform_price_adjustment=False,
                                           perform_order_corrections=True, generate_correction_templates=False,
                                           apply_corrections_template=False, perform_fee_structure_update=False)
        #position_manager.perform_price_recalibration(time_per_batch_s=10000000)
    else:
        position_manager = PositionManager()
        position_manager.init_cache_files()

    challengeperiod_manager = ChallengePeriodManager(config=None, metagraph=None)
    # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
    # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup smallest timestamp is
    # older than the local smallest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
    smallest_disk_ms, largest_disk_ms = (
        position_manager.get_extreme_position_order_processed_on_disk_ms())
    smallest_backup_ms = data['youngest_order_processed_ms']
    largest_backup_ms = data['oldest_order_processed_ms']
    try:
        formatted_backup_creation_time = TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)
        formatted_disk_date_largest = TimeUtil.millis_to_formatted_date_str(largest_disk_ms)
        formatted_backup_date_largest = TimeUtil.millis_to_formatted_date_str(largest_backup_ms)
        formatted_disk_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_disk_ms)
        formatted_backup_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_backup_ms)
    except:
        formatted_backup_creation_time = backup_creation_time_ms
        formatted_disk_date_largest = largest_disk_ms
        formatted_backup_date_largest = largest_backup_ms
        formatted_disk_date_smallest = smallest_disk_ms
        formatted_backup_date_smallest = smallest_backup_ms

    bt.logging.info(f"Timestamp analysis of backup vs disk (UTC):")
    bt.logging.info(f"    backup_creation_time: {formatted_backup_creation_time}")
    bt.logging.info(f"    smallest_disk_order_timestamp: {formatted_disk_date_smallest}")
    bt.logging.info(f"    smallest_backup_order_timestamp: {formatted_backup_date_smallest}")
    bt.logging.info(f"    oldest_disk_order_timestamp: {formatted_disk_date_largest}")
    bt.logging.info(f"    oldest_backup_order_timestamp: {formatted_backup_date_largest}")

    if smallest_disk_ms >= smallest_backup_ms and largest_disk_ms <= backup_creation_time_ms:
        pass  # Ready for update!
    elif largest_disk_ms > backup_creation_time_ms:
        bt.logging.error(f"Please re-pull the backup file before restoring. Backup {formatted_backup_creation_time} appears to be older than the disk {formatted_disk_date_largest}.")
        return False
    elif smallest_disk_ms < smallest_backup_ms:
        #bt.logging.error("Your local filesystem has older orders than the backup. Please reach out to the team ASAP before regenerating. You may be holding irrecoverable data!")
        #return False
        pass  # Deregistered miners can trip this check. We will allow the regeneration to proceed.
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

    perf_ledgers = data.get('perf_ledgers', {})
    bt.logging.info(f"regenerating {len(perf_ledgers)} perf ledgers")
    PerfLedgerManager.save_perf_ledgers_to_disk(perf_ledgers)

    bt.logging.info(f"regenerating {len(data['plagiarism'])} plagiarism scores")
    position_manager.write_plagiarism_scores_to_disk(data['plagiarism'])

    ## Now sync challenge period with the disk
    challengeperiod = data.get('challengeperiod', {})
    challengeperiod_manager.challengeperiod_testing = challengeperiod.get('testing', {})  
    challengeperiod_manager.challengeperiod_success = challengeperiod.get('success', {})

    current_timestamp = TimeUtil.now_in_millis()
    challengeperiod_manager.challengeperiod_testing = { **challengeperiod_manager.challengeperiod_testing, **{
        k: current_timestamp for k in ['5CPwzn7JDkh9kjjqpC4zFHnZNBjL6JrBnY91cwLjyBwVUKy5', '5FxmH6iry6LaBxrBxnbUpQdZBMM6eZuFyPk4pj7EZMrqjie3', '5EySUmguNd7eoWoTABxdLDnoCsGHekAPVfr7s43ooPMQr8nJ', '5Cfx8PtVZxXcdVLBW6suwyvU8QmnZCHom5fVPfexJhkQh16U', '5G9yGe6TEDxx7wD2mpM2XSu8P7gVt6wwUvuXRrHGAYJDA955', '5DDBXwjobdGeiM3svUPUogVB6T3YVNC7nmin9p7EQGBrK2hA', '5DaHdgTLPrGCdiNMosKq9GEpDmA6pPaMvNopXtnG28AtYghm', '5GRFAJ3iwukm1CQpDTBsxTh2xjm227KtbVC1za8ukk6WqeyN', '5CrGaMAv5guzzoyef6XBUPiBGhsrnox7nxPayV8DPzZh1zQL', '5GzYKUYSD5d7TJfK4jsawtmS2bZDgFuUYw8kdLdnEDxSykTU', '5DnViSacXqrP8FnQMtpAFGyahUPvU2A6pbrX7wcexb3bmVjb', '5HMmbZ7UUPnfCgvBcXpUqJe1XopEZVPGoFDFSs1MRgtV3GLK', '5E7DEGmFUewdJTnSh829jGc3SpSd295hhvUgNcNiQST6bw4A', '5Exax1W9RiNbARDejrthf4SK1FQ2u9DPUhCq9jm58gUysTy4', '5Gdr6mT7Ssyr9ByvenBKie4wgTqci3B7fN4kQ7cwVkBKPfqa', '5D7ZcGnnzT3yzwkZd94oGYXdHbCkrkrn7XELaXdR5dDHrtJX', '5DoCFr2EoW1CGuYCEXhsuQdWRsgiUMuxGwNt4Xqb5TCptcBW', '5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X']
    }}
                                                                                
    challengeperiod_manager.challengeperiod_success = { k: current_timestamp for k in ['5HDmzyhrEco9w6Jv8eE3hDMcXSE4AGg1MuezPR4u2covxKwZ', '5HgBDcx8Z9oEWrGQm7obH4aPE2M5YXWA6S6MP1HtFHguUqek', '5DX8tSyGrx1QuoR1wL99TWDusvmmWgQW5su3ik2Sc8y8Mqu3', '5H8niLrzmxZUhAzRM29GNcnDyJPWEwujw5nbENuWcDV889W4', '5C8Wegdus2cAcwSNU47MdiLXwZdewFkSv93xUWQP3wn32QJV', '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx', '5EPDfdoeYhygXYEJ9xo8DV6kLuQZnrZgvH87sqci7WDM2j4g', '5Da5hqCMSVgeGWmzeEnNrime3JKfgTpQmh7dXsdMP58dgeBd', '5FqSBwa7KXvv8piHdMyVbcXQwNWvT9WjHZGHAQwtoGVQD3vo', '5EAS8w6A4Nwc4quVQUs6xEDdhNSCNFgJ2ZzkHtJm83KthJaN', '5HBCKWiy27ncsMzX3aF1hP4yPqPJy86knbAoedeS1XymfSpn', '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw', '5Ec93qtHkKprEaA5EWXrmPmWppMeMiwaY868bpxfkH5ocBxi', '5Eh9p81ioCeoTArv7kSa1PWcaXw33UdRjVQfLQsFPpn474GC', '5CnuyazUFWumVNTxoqMPc3bWk3FBj6eUbRjHDm8jsrEqaDkS', '5Dxqzduahnqw8q3XSUfTcEZGU7xmAsfJubhHZwvXVLN9fSjR', '5DcgKr6s8z75sE4c69iMSM8adfRVex7A8BZe2mouVwMVRis4', '5Fjz3ENZwDkn2txvryhPofbn2T3DbyHferTxvsastmmggFFb', '5His3c7GyUKpWgRpuWAZfHKxtszZLQuTSMaEWM4NbkS1wsNm', '5CSHrvBiEJAFAAj7YAr5y8jzmpFajRsV9PahphPGi7P8PZrA', '5DqmvEK7Viv2NpEEJGJVuYaQEGpeSW6HAVxrNvV18CLxKve5', '5EF393sRCV3Q6SFNTpKQed8m3QDGRgfDvke8sUoH3kbLqGZS', '5Df8YED2EoxY65B2voeCHzY9rn1R76DXB8Cq9f62CsGVRoU5', '5EZoATFyB3FdCEqEBuWSSDpdqFc8pePm6n5fMVRTuKpLu6Dr', '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t', '5EPhd4PXgdQtxSXBUfB6FodJ2Uxy7TeVf8ZVGoP8gfGyCuqW', '5Ct1J2jNxb9zeHpsj547BR1nZk4ZD51Bb599tzEWnxyEr4WR', '5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ', '5HY6RF4RVDnX8CQ5JrcnzjM2rixQXZpW2eh48aNLUYT1V9LW', '5F1sxW5apTPEYfDJUoHTRG4kGaUmkb3YVi5hwt5A9Fu8Gi6a', '5FpypsPpSFUBpByFXMkJ34sV88PRjAKSSBkHkmGXMqFHR19Q'] }

    challengeperiod_manager._write_challengeperiod_from_memory_to_disk()
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
