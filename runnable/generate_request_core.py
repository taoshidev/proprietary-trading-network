import json
import traceback
import uuid
import time
import copy
from json import JSONDecodeError
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.scoring.scoring import Scoring, ScoringUnit
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager


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
    one_week_ago_ms = time_now - 1000 * 60 * 60 * 24 * 7
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

            # Remove price sources (debug info) for old data to make the checkpoint smaller.
            if p.is_closed_position and p.close_ms < one_week_ago_ms:
                for o in p.orders:
                    o.price_sources = []

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

    vcp_output_file_path = ValiBkpUtils.get_vali_outputs_dir() + "validator_checkpoint.json"
    ValiBkpUtils.write_file(
        vcp_output_file_path,
        final_dict,
    )

    miner_positions_output_file_path = ValiConfig.BASE_DIR + "/validation/outputs/output.json"
    ValiBkpUtils.write_file(
        miner_positions_output_file_path,
        ord_dict_hotkey_position_map,
    )