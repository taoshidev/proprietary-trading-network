# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import json
import random
from inspect import isclass
from pickle import UnpicklingError
from typing import Dict, List, Type, Tuple

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.cmw.cmw_objects.cmw import CMW
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.dataclasses.base_objects.new_request_dataclass import NewRequestDataClass
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.dataclasses.prediction_request import PredictionRequest
from vali_objects.dataclasses.training_request import TrainingRequest

from vali_objects.dataclasses.position import Position
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.exceptions.vali_memory_missing_exception import ValiMemoryMissingException
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.scaling.scaling import Scaling
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_memory_utils import ValiMemoryUtils


class ValiUtils:
    @staticmethod
    def get_vali_records() -> CMW:
        try:
            return ValiUtils.get_vali_memory_json()
        except ValiMemoryMissingException:
            ValiUtils.set_memory_with_bkp()
            return ValiUtils.get_vali_records()

    @staticmethod
    def get_vali_bkp_json() -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            vbkp = ValiBkpUtils.get_vali_file(ValiBkpUtils.get_vali_bkp_dir()
                                              + ValiBkpUtils.get_vali_data_file())
        except FileNotFoundError:
            init_cmw = CMWUtil.initialize_cmw()
            ValiUtils.set_vali_bkp(init_cmw)
            return init_cmw
        else:
            return json.loads(vbkp)

    @staticmethod
    def get_vali_weights_json() -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            vweights = ValiBkpUtils.get_vali_file(ValiBkpUtils.get_vali_weights_dir()
                                              + ValiBkpUtils.get_vali_weights_file())
        except FileNotFoundError:
            return {}
        else:
            return json.loads(vweights)

    @staticmethod
    def get_vali_memory_json() -> CMW:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            vm = ValiMemoryUtils.get_vali_memory()
        except KeyError:
            raise ValiMemoryMissingException("loading vali memory failed")
        else:
            if vm is None:
                raise ValiMemoryMissingException("vm is none")
            return CMWUtil.load_cmw(json.loads(vm))

    @staticmethod
    def check_memory_matches_bkp() -> bool:
        vbkp = ValiUtils.get_vali_bkp_json()
        vm = CMWUtil.dump_cmw(ValiUtils.get_vali_memory_json())
        if vbkp != vm:
            raise ValiRecordsMisalignmentException("misalignment of bkp and memory")
        else:
            return True

    @staticmethod
    def set_memory_with_bkp() -> None:
        ValiMemoryUtils.set_vali_memory(json.dumps(ValiUtils.get_vali_bkp_json()))

    @staticmethod
    def save_cmw_results(request_uuid: str, content: Dict | object):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_bkp_dir(),
                                     ValiBkpUtils.get_cmw_filename(request_uuid),
                                     content)

    @staticmethod
    def save_predictions_request(request_uuid: str, content: Dict | object):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_predictions_dir(),
                                     ValiBkpUtils.get_response_filename(request_uuid),
                                     content, True)

    @staticmethod
    def set_vali_memory_and_bkp(vali_records: Dict):
        ValiMemoryUtils.set_vali_memory(json.dumps(vali_records))
        ValiUtils.set_vali_bkp(vali_records)

    @staticmethod
    def set_vali_bkp(vali_records: Dict):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_bkp_dir(),
                                     ValiBkpUtils.get_vali_data_file(),
                                     vali_records)

    @staticmethod
    def set_vali_weights_bkp(vali_weights: Dict):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_weights_dir(),
                                     ValiBkpUtils.get_vali_weights_file(),
                                     vali_weights)

    @staticmethod
    def get_all_miner_positions_and_orders(miner_hotkey, only_open_positions=False):
        def sort_by_end(item):
            return item.open_ms

        miner_dir = ValiBkpUtils.get_miner_dir(miner_hotkey)
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)

        positions = [ValiUtils.get_miner_positions(file) for file in all_files]

        if only_open_positions:
            positions = [position for position in positions if position.close_ms is None]

        sorted_pred_requests = sorted(positions, key=sort_by_end, reverse=True)
        return sorted_pred_requests



    @staticmethod
    def get_predictions_to_complete() -> List[PredictionRequest]:
        def sort_by_end(item):
            return item.df.end

        all_files = ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_vali_predictions_dir())
        request_to_complete = {}
        for file in all_files:
            unpickled_df = ValiUtils.get_miner_positions(file)
            # need to add a buffer of 24 hours to ensure the data is available via api requests
            if TimeUtil.now_in_millis() > unpickled_df.end + TimeUtil.hours_in_millis(1):
                if unpickled_df.vmins is not None \
                        and unpickled_df.vmaxs is not None \
                        and unpickled_df.decimal_places is not None:
                    unpickled_unscaled_data_structure = Scaling.unscale_values(unpickled_df.vmins[0],
                                                                               unpickled_df.vmaxs[0],
                                                                               unpickled_df.decimal_places[0],
                                                                               unpickled_df.predictions)
                else:
                    unpickled_unscaled_data_structure = unpickled_df.predictions
                if unpickled_df.request_uuid not in request_to_complete:
                    # keeping as a dict to easily add new files to ref
                    request_to_complete[unpickled_df.request_uuid] = PredictionRequest(
                        request_uuid=unpickled_df.request_uuid,
                        df=unpickled_df,
                        files=[],
                        predictions={}
                    )
                request_to_complete[
                    unpickled_df.request_uuid].predictions[unpickled_df.miner_uid] = unpickled_unscaled_data_structure
                request_to_complete[
                    unpickled_df.request_uuid].files.append(file)
        pred_requests = [pred_request for pred_request in request_to_complete.values()]
        sorted_pred_requests = sorted(pred_requests, key=sort_by_end, reverse=True)
        return sorted_pred_requests

    @staticmethod
    def generate_standard_request(request: Type[NewRequestDataClass]):
        # templated for now as we trade only btc
        stream_type = "BTCUSD-5m"
        topic_id = 1
        schema_id = 1
        feature_ids = [0.001, 0.002, 0.003, 0.004, 0.005]
        prediction_size = int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX))
        additional_details = {
            "tf": 5,
            "trade_pair": "BTCUSD"
        }

        if request == TrainingRequest:
            return TrainingRequest(
                stream_type=stream_type,
                topic_id=topic_id,
                schema_id=schema_id,
                feature_ids=feature_ids,
                prediction_size=prediction_size,
                additional_details=additional_details
            )
        elif request == ClientRequest:
            return ClientRequest(
                stream_type=stream_type,
                topic_id=topic_id,
                schema_id=schema_id,
                feature_ids=feature_ids,
                prediction_size=prediction_size,
                additional_details=additional_details
            )
        else:
            raise Exception("not a recognizable client request")

    @staticmethod
    def randomize_days(historical_lookback: bool) -> (int, int, List[Tuple[int, int]]):
        days = int(random.uniform(ValiConfig.HISTORICAL_DATA_LOOKBACK_DAYS_MIN,
                                  ValiConfig.HISTORICAL_DATA_LOOKBACK_DAYS_MAX))
        # if 1 then historical lookback, otherwise live
        if historical_lookback:
            start = int(random.uniform(30, 1460))
        else:
            start = days
        TimeUtil.generate_start_timestamp(start)
        return TimeUtil.generate_start_timestamp(start), \
               TimeUtil.generate_start_timestamp(start - days), \
               TimeUtil.convert_range_timestamps_to_millis(
                   TimeUtil.generate_range_timestamps(
                       TimeUtil.generate_start_timestamp(start), days))

    @staticmethod
    def get_standardized_ds() -> List[List]:
        # close time, close, high, low, volume
        return [[], [], [], [], []]

    @staticmethod
    def get_miner_positions(file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            return ValiBkpUtils.get_vali_file(file, True)
        except FileNotFoundError:
            raise ValiFileMissingException("Vali position file is missing")
        except UnpicklingError:
            raise ValiBkpCorruptDataException("position data is not pickled")

    @staticmethod
    def get_secrets() -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_vali_file(ValiBkpUtils.get_secrets_dir())
            return json.loads(secrets)
        except FileNotFoundError:
            raise ValiFileMissingException("Vali secrets file is missing")

    @staticmethod
    def save_miner_position(miner_hotkey: str, position: str, content: Dict | object):
        ValiBkpUtils.write_vali_file(ValiBkpUtils.get_vali_predictions_dir(),
                                     ValiBkpUtils.get_miner_position_dir(miner_hotkey, position),
                                     content, True)
