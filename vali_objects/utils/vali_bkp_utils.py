# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import json
import os
import pickle

from vali_config import ValiConfig

class ValiBkpUtils:
    @staticmethod
    def get_miner_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/miners/"

    @staticmethod
    def get_miner_position_dir(miner_hotkey, running_unit_tests=False) -> str:
        return f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}{miner_hotkey}/positions/"

    @staticmethod
    def get_eliminations_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/eliminations.json"

    @staticmethod
    def get_plagiarism_scores_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/plagiarism.json"

    @staticmethod
    def get_secrets_dir():
        return ValiConfig.BASE_DIR + f"/secrets.json"

    @staticmethod
    def get_vali_bkp_dir() -> str:
        return ValiConfig.BASE_DIR + "/validation/backups/"

    @staticmethod
    def get_vali_outputs_dir() -> str:
        return ValiConfig.BASE_DIR + "/validation/outputs/"

    @staticmethod
    def get_vali_weights_dir() -> str:
        return ValiConfig.BASE_DIR + "/validation/weights/"

    @staticmethod
    def get_vali_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/"

    @staticmethod
    def get_vali_data_file() -> str:
        return "valirecords.json"

    @staticmethod
    def get_vali_weights_file() -> str:
        return "valiweights.json"

    @staticmethod
    def get_vali_predictions_dir() -> str:
        return ValiConfig.BASE_DIR + "/validation/predictions/"

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
        return "wb" if is_pickle else "w"

    @staticmethod
    def get_read_type(is_pickle: bool) -> str:
        return "rb" if is_pickle else "r"

    @staticmethod
    def write_to_dir(
        vali_file: str, vali_data: dict | object, is_pickle: bool = False
    ) -> None:
        os.makedirs(os.path.dirname(vali_file), exist_ok=True)
        with open(vali_file, ValiBkpUtils.get_write_type(is_pickle)) as f:
            pickle.dump(vali_data, f) if is_pickle else f.write(json.dumps(vali_data))
        f.close()

    @staticmethod
    def write_file(
        vali_dir: str, vali_data: dict | object, is_pickle: bool = False
    ) -> None:
        ValiBkpUtils.write_to_dir(vali_dir, vali_data, is_pickle)

    @staticmethod
    def get_file(vali_file: str, is_pickle: bool = False) -> str | object:
        with open(vali_file, ValiBkpUtils.get_read_type(is_pickle)) as f:
            ans = pickle.load(f) if is_pickle else f.read()
            #bt.logging.info(f"vali_file: {vali_file}, ans: {ans}")
            return ans

    @staticmethod
    def get_all_files_in_dir(vali_dir: str) -> list[str]:
        all_files = []
        for dirpath, dirnames, filenames in os.walk(vali_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                all_files.append(filepath)
        return all_files

    @staticmethod
    def get_directories_in_dir(directory):
        return [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]
