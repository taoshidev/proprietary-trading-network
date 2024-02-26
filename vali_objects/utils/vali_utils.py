# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import json
from pickle import UnpicklingError
from typing import Dict, List

from vali_objects.position import Position
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import (
    ValiFileMissingException,
)
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


class ValiUtils:
    ELIMINATIONS = "eliminations"
    COPY_TRADING = "copy_trading"

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
    def save_miner_position(
        miner_hotkey: str, position_uuid: str, content: Dict | object
    ) -> None:
        ValiBkpUtils.write_vali_file(
            ValiBkpUtils.get_miner_position_dir(miner_hotkey) + position_uuid,
            content,
            True,
        )

    @staticmethod
    def get_vali_json_file(vali_dir: str, key: str = None) -> List | Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_vali_file(vali_dir)
            if key is not None:
                return json.loads(secrets)[key]
            else:
                return json.loads(secrets)
        except FileNotFoundError:
            print(f"no vali json file [{dir}], continuing")
            return []
