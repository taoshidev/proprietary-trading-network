# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import json
import os
import pickle
from datetime import datetime

from vali_config import ValiConfig
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile


class ValiBkpUtils:

    @staticmethod
    def get_miner_dir(miner_hotkey) -> str:
        return ValiConfig.BASE_DIR + f'/validation/{miner_hotkey}'

    @staticmethod
    def get_miner_position_dir(miner_hotkey, position) -> str:
        return ValiConfig.BASE_DIR + f'/validation/{miner_hotkey}/{position}'

    @staticmethod
    def get_miner_order_dir(hotkey, position, order) -> str:
        return ValiConfig.BASE_DIR + f'/validation/{hotkey}/{position}/{order}'

    @staticmethod
    def get_secrets_dir():
        return ValiConfig.BASE_DIR + f'/validation/secrets.json'

    @staticmethod
    def get_vali_bkp_dir() -> str:
        return ValiConfig.BASE_DIR + '/validation/backups/'

    @staticmethod
    def get_vali_outputs_dir() -> str:
        return ValiConfig.BASE_DIR + '/validation/outputs/'

    @staticmethod
    def get_vali_weights_dir() -> str:
        return ValiConfig.BASE_DIR + '/validation/weights/'

    @staticmethod
    def get_vali_data_file() -> str:
        return 'valirecords.json'

    @staticmethod
    def get_vali_weights_file() -> str:
        return 'valiweights.json'

    @staticmethod
    def get_vali_predictions_dir() -> str:
        return ValiConfig.BASE_DIR + '/validation/predictions/'

    @staticmethod
    def get_response_filename(request_uuid: str) -> str:
        return str(request_uuid) + ".pickle"

    @staticmethod
    def get_cmw_filename(request_uuid: str) -> str:
        return str(request_uuid) + ".json"

    @staticmethod
    def make_dir(vali_dir: str) -> None:
        if not os.path.exists(vali_dir):
            os.makedirs(vali_dir)

    @staticmethod
    def get_write_type(is_pickle: bool) -> str:
        return 'wb' if is_pickle else 'w'

    @staticmethod
    def get_read_type(is_pickle: bool) -> str:
        return 'rb' if is_pickle else 'r'

    @staticmethod
    def write_to_vali_dir(vali_file: str, vali_data: dict | object, is_pickle: bool = False) -> None:
        with open(vali_file, ValiBkpUtils.get_write_type(is_pickle)) as f:
            pickle.dump(vali_data, f) if is_pickle else f.write(json.dumps(vali_data))
        f.close()

    @staticmethod
    def write_vali_file(vali_dir: str, file_name: str, vali_data: dict | object, is_pickle: bool = False) -> None:
        # will concat dir and file name
        ValiBkpUtils.make_dir(vali_dir)
        ValiBkpUtils.write_to_vali_dir(vali_dir + file_name, vali_data, is_pickle)

    @staticmethod
    def get_vali_file(vali_file, is_pickle: bool = False) -> str | PredictionDataFile:
        with open(vali_file, ValiBkpUtils.get_read_type(is_pickle)) as f:
            return pickle.load(f) if is_pickle else f.read()

    @staticmethod
    def get_all_files_in_dir(vali_dir: str) -> list[str]:
        all_files = []
        if os.path.exists(vali_dir):
            for filename in os.listdir(vali_dir):
                if os.path.isfile(os.path.join(vali_dir, filename)):
                    all_files.append(vali_dir + filename)
        return all_files

    @staticmethod
    def delete_stale_files(vali_dir: str) -> None:
        current_date = datetime.now()
        if os.path.exists(vali_dir):
            for filename in os.listdir(vali_dir):
                file_path = os.path.join(vali_dir, filename)
                if os.path.isfile(file_path):
                    creation_timestamp = os.path.getctime(file_path)
                    creation_date = datetime.fromtimestamp(creation_timestamp)
                    age_in_days = (current_date - creation_date).days
                    if age_in_days > ValiConfig.DELETE_STALE_DATA:
                        os.remove(file_path)



