import argparse
import time
import requests
import json

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
import bittensor as bt


def compute_delta(mothership_json, min_time_ms):

    bt.logging.info("Found mothership data with the following attributes:")
    # Log every key and value apir in the data except for positions, eliminations, and plagiarism scores
    for key, value in mothership_json.items():
        # Check is the value is of type dict or list. If so, print the size of the dict or list
        if isinstance(value, dict) or isinstance(value, list):
            # Log the size of the positions, eliminations, and plagiarism scores
            bt.logging.info(f"    {key}: {len(value)} entries")
        else:
            bt.logging.info(f"    {key}: {value}")
    backup_creation_time_ms = mothership_json['created_timestamp_ms']

    elimination_manager = EliminationManager(None, None, None)
    position_manager = PositionManager(perform_order_corrections=True,
                                       challengeperiod_manager=None,
                                       elimination_manager=elimination_manager)

    # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
    # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup smallest timestamp is
    # older than the local smallest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
    smallest_disk_ms, largest_disk_ms = (
        position_manager.get_extreme_position_order_processed_on_disk_ms())
    smallest_backup_ms = mothership_json['youngest_order_processed_ms']
    largest_backup_ms = mothership_json['oldest_order_processed_ms']
    try:
        formatted_backup_creation_time = TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)
        formatted_disk_date_largest = TimeUtil.millis_to_formatted_date_str(largest_disk_ms)
        formatted_backup_date_largest = TimeUtil.millis_to_formatted_date_str(largest_backup_ms)
        formatted_disk_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_disk_ms)
        formatted_backup_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_backup_ms)
    except:  # noqa: E722
        formatted_backup_creation_time = backup_creation_time_ms
        formatted_disk_date_largest = largest_disk_ms
        formatted_backup_date_largest = largest_backup_ms
        formatted_disk_date_smallest = smallest_disk_ms
        formatted_backup_date_smallest = smallest_backup_ms

    bt.logging.info("Timestamp analysis of mothership vs disk (UTC):")
    bt.logging.info(f"    mothership_creation_time: {formatted_backup_creation_time}")
    bt.logging.info(f"    smallest_disk_order_timestamp: {formatted_disk_date_smallest}")
    bt.logging.info(f"    smallest_mothership_order_timestamp: {formatted_backup_date_smallest}")
    bt.logging.info(f"    oldest_disk_order_timestamp: {formatted_disk_date_largest}")
    bt.logging.info(f"    oldest_mothership_order_timestamp: {formatted_backup_date_largest}")


    n_existing_position = position_manager.get_number_of_miners_with_any_positions()
    n_existing_eliminations = position_manager.get_number_of_eliminations()
    msg = (f"Detected {n_existing_position} existing hotkeys with positions, {n_existing_eliminations} existing eliminations")
    bt.logging.info(msg)

    bt.logging.info("Computing delta positions")

    delta_positions = []
    position_uuids_added = set()
    delta_orders = []
    delta_order_positions = []
    for hotkey, json_positions in mothership_json['positions'].items():
        # sort positions by close_ms otherwise, writing a closed position after an open position for the same
        # trade pair will delete the open position
        mothership_positions = [Position(**json_positions_dict) for json_positions_dict in json_positions['positions']]
        if not mothership_positions:
            continue
        disk_positions = position_manager.get_positions_for_one_hotkey(hotkey)
        if min_time_ms:
            disk_positions = [p for p in disk_positions if any(o.processed_ms >= min_time_ms for o in p.orders)]
            mothership_positions = [p for p in mothership_positions if any(o.processed_ms >= min_time_ms for o in p.orders)]
        for mp in mothership_positions:
            corresponding_position = None
            for dp in disk_positions:
                if mp.position_uuid == dp.position_uuid:
                    corresponding_position = dp
                    break
            if corresponding_position is None:
                delta_positions.append(mp)
                position_uuids_added.add(mp.position_uuid)
                continue

            for mo in mp.orders:
                corresponding_order = None
                for do in corresponding_position.orders:
                    if mo.order_uuid == do.order_uuid:
                        corresponding_order = do
                        break
                if corresponding_order is None:
                    delta_orders.append((corresponding_position.orders, mp.orders))
                    if mp.position_uuid not in position_uuids_added:
                        delta_order_positions.append(mp)
                        position_uuids_added.add(mp.position_uuid)
                    break

    print(f"Found {len(delta_positions)} positions to snap to and {len(delta_orders)} order deltas to snap to ")
    for position in delta_positions:
        print(position.to_copyable_str(), '\n\n')

    for position in delta_order_positions:
        print(position.to_copyable_str(), '\n\n')

def get_mothership_checkpoint(url, api_key):
    data = {
        'api_key': api_key
    }
    json_data = json.dumps(data)
    headers = {
        'Content-Type': 'application/json',
    }
    test = requests.get(url, data=json_data, headers=headers)
    return test.json()


if __name__ == "__main__":
    bt.logging.enable_info()
    parser = argparse.ArgumentParser(description="Compute a delta with mothership in a format that can be used with PM.")
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--url', type=str, default=None)
    parser.add_argument('--min-time-ms', type=int, default=None)

    t0 = time.time()
    # Parse command-line arguments
    args = parser.parse_args()



    # Use the disable_backup argument to control backup
    api_key = args.api_key
    url = args.url
    min_time_ms = args.min_time_ms
    bt.logging.info(f"Computing delta. api_key: {api_key}, url: {url}, min_time_ms: {min_time_ms}")

    mothership_json = get_mothership_checkpoint(url, api_key)
    #with open('validator_checkpoint.json', 'r') as f:
    #    mothership_json = json.loads(f.read())

    compute_delta(mothership_json, min_time_ms)
    bt.logging.info("Delta complete in %.2f seconds" % (time.time() - t0))
