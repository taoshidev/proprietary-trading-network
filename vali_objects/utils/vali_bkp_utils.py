# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import json
import os
import pickle

from vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import OrderStatus


class ValiBkpUtils:
    @staticmethod
    def get_miner_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/miners/"

    @staticmethod
    def get_miner_all_positions_dir(miner_hotkey, running_unit_tests=False) -> str:
        return f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}{miner_hotkey}/positions/"

    @staticmethod
    def get_eliminations_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/eliminations.json"

    @staticmethod
    def get_plagiarism_scores_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/plagiarism.json"

    @staticmethod
    def get_secrets_dir():
        return ValiConfig.BASE_DIR + f"/secrets.json"

    @staticmethod
    def get_plagiarism_blocklist_file_location():
        return ValiConfig.BASE_DIR + f"/miner_blocklist.json"
    
    @staticmethod
    def get_vali_bkp_dir() -> str:
        return ValiConfig.BASE_DIR + "/backups/"

    @staticmethod
    def get_vali_outputs_dir() -> str:
        return ValiConfig.BASE_DIR + "/runnable/"

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
            if isinstance(vali_data, Position):
                f.write(vali_data.to_json_string())
            elif is_pickle:
                pickle.dump(vali_data, f)
            else:
                f.write(json.dumps(vali_data))
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

    @staticmethod
    def get_partitioned_miner_positions_dir(miner_hotkey, trade_pair_id, order_status=OrderStatus.ALL,
                                            running_unit_tests=False) -> str:

        base_dir = (f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}"
               f"{miner_hotkey}/positions/{trade_pair_id}/")

        # Decide the subdirectory based on the order_status
        status_dir = {
            OrderStatus.OPEN: "open/",
            OrderStatus.CLOSED: "closed/",
            OrderStatus.ALL: ""
        }[order_status]

        return f"{base_dir}{status_dir}"
