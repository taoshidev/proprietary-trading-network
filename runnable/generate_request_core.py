import io
import json
import os
import zipfile

import bittensor as bt
from google.cloud import storage

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager


def upload_checkpoint_to_gcloud(final_dict):
    """
    The idea is to upload a zipped, time lagged validator checkpoint to google cloud for auto restoration
    on other validators as well as transparency with the community.

    """
    # Only upload the checkpoint at 5:09:00, 5:19:00, ... 5:59:00 UTC
    datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
    #TODO: Reinstate original condition
    #if not (datetime_now.hour == 6 and datetime_now.minute < 9 and datetime_now.second < 30):
    if not (datetime_now.minute == 22 and datetime_now.second < 30):
        return

    # check if file exists
    KEY_PATH = ValiConfig.BASE_DIR + '/gcloud.json'
    if not os.path.exists(KEY_PATH):
        return

    # Path to your service account key file
    key_path = KEY_PATH
    key_info = json.load(open(key_path))

    # Initialize a storage client using your service account key
    client = storage.Client.from_service_account_info(key_info)

    # Name of the bucket you want to write to
    bucket_name = 'validator_checkpoint'

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Name for the new blob
    # blob_name = 'validator_checkpoint.json'
    blob_name = 'validator_checkpoint.zip'

    # Create a new blob and upload data
    blob = bucket.blob(blob_name)

    positions = final_dict['positions']
    # 24 hours in milliseconds
    time_lag = 1000 * 60 * 60 * 24
    for hotkey, positions in positions.items():
        new_positions = []
        for position in positions['positions']:
            new_orders = []
            for order in position['orders']:
                if order['processed_ms'] < time_lag:
                    new_orders.append(order)
            if len(new_orders):
                position.orders = new_orders
                position.rebuild_position_with_updated_orders()
                new_positions.append(position)
            else:
                # if no orders are left, remove the position
                pass
        positions[hotkey] = new_positions

    str = json.dumps(final_dict, cls=CustomEncoder)

    # Create a zip file in memory
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add the json file to the zip file
            zip_file.writestr('validator_checkpoint.json', str)
            #zip_file.write(ValiBkpUtils.get_vcp_output_path(), arcname='validator_checkpoint.json')

        # Rewind the buffer's file pointer to the beginning so you can read its content
        zip_buffer.seek(0)

        # Upload the content of the zip_buffer to Google Cloud Storage
        blob.upload_from_file(zip_buffer)

    print(f'Uploaded {blob_name} to {bucket_name}')

def generate_request_core(time_now:int):
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

    ## Collect information from the disk and populate variables in memory
    subtensor_weight_setter._refresh_eliminations_in_memory()
    subtensor_weight_setter._refresh_challengeperiod_in_memory()

    challengeperiod_testing_dictionary = subtensor_weight_setter.challengeperiod_testing
    challengeperiod_success_dictionary = subtensor_weight_setter.challengeperiod_success

    try:
        all_miner_hotkeys:list = ValiBkpUtils.get_directories_in_dir(
            ValiBkpUtils.get_miner_dir()
        )
    except FileNotFoundError:
        raise Exception(
            f"directory for miners doesn't exist "
            f"[{ValiBkpUtils.get_miner_dir()}]. Skip run for now."
        )
    
    # Perf Ledger Calculations
    perf_ledgers = PerfLedgerManager.load_perf_ledgers_from_disk()

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

    youngest_order_processed_ms = float("inf")
    oldest_order_processed_ms = 0

    for k, original_positions in hotkey_positions.items():
        dict_hotkey_position_map[k] = {
            "positions": [],
            "thirty_day_returns": 1.0,
            "all_time_returns": 1.0,
            "n_positions": 0,
            "percentage_profitable": 0.0
        }
        positions_30_days = [
            position
            for position in original_positions
            if position.open_ms > acceptable_position_end_ms
        ]

        if k not in eliminated_hotkeys:
            ps_30_days = subtensor_weight_setter._filter_positions(positions_30_days)
            return_per_position = position_manager.get_return_per_closed_position(ps_30_days)
            if len(return_per_position) > 0:
                curr_return = return_per_position[len(return_per_position) - 1]
                dict_hotkey_position_map[k]["thirty_day_returns"] = curr_return

            ps_all_time = subtensor_weight_setter._filter_positions(original_positions)
            return_per_position = position_manager.get_return_per_closed_position(ps_all_time)
            if len(return_per_position) > 0:
                curr_return = return_per_position[len(return_per_position) - 1]
                dict_hotkey_position_map[k]["all_time_returns"] = curr_return
                dict_hotkey_position_map[k]["n_positions"] = len(ps_all_time)
                dict_hotkey_position_map[k]["percentage_profitable"] = position_manager.get_percent_profitable_positions(ps_all_time)

        for p in original_positions:
            youngest_order_processed_ms = min(youngest_order_processed_ms,
                                              min(p.orders, key=lambda o: o.processed_ms).processed_ms)
            oldest_order_processed_ms = max(oldest_order_processed_ms,
                                            max(p.orders, key=lambda o: o.processed_ms).processed_ms)
            if p.close_ms is None:
                p.close_ms = 0

            position_manager.strip_old_price_sources(p, time_now)

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

    n_orders_original = 0
    for positions in hotkey_positions.values():
        n_orders_original += sum([len(position.orders) for position in positions])

    n_positions_new = 0
    for data in ord_dict_hotkey_position_map.values():
        positions = data['positions']
        n_positions_new += sum([len(p['orders']) for p in positions])

    assert n_orders_original == n_positions_new, f"n_orders_original: {n_orders_original}, n_positions_new: {n_positions_new}"
    final_dict = {
        'version': ValiConfig.VERSION,
        'created_timestamp_ms': time_now,
        'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
        'challengeperiod': {
            "testing": challengeperiod_testing_dictionary,
            "success": challengeperiod_success_dictionary
        },
        'eliminations': eliminations,
        'youngest_order_processed_ms': youngest_order_processed_ms,
        'oldest_order_processed_ms': oldest_order_processed_ms,
        'positions': ord_dict_hotkey_position_map,
        'perf_ledgers': perf_ledgers
    }

    vcp_output_file_path = ValiBkpUtils.get_vcp_output_path()
    ValiBkpUtils.write_file(
        vcp_output_file_path,
        final_dict,
    )

    miner_positions_output_file_path = ValiBkpUtils.get_miner_positions_output_path()
    ValiBkpUtils.write_file(
        miner_positions_output_file_path,
        ord_dict_hotkey_position_map,
    )

    upload_checkpoint_to_gcloud(final_dict)