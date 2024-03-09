# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import datetime
import json
import os
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
            return ValiBkpUtils.get_file(file, True)
        except FileNotFoundError:
            raise ValiFileMissingException("Vali position file is missing")
        except UnpicklingError:
            raise ValiBkpCorruptDataException("position data is not pickled")

    @staticmethod
    def get_secrets() -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_file(ValiBkpUtils.get_secrets_dir())
            return json.loads(secrets)
        except FileNotFoundError:
            raise ValiFileMissingException("Vali secrets file is missing")

    @staticmethod
    def save_miner_position(
        miner_hotkey: str, position_uuid: str, content: Dict | object
    ) -> None:
        ValiBkpUtils.make_dir(ValiBkpUtils.get_miner_position_dir(miner_hotkey))
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_miner_position_dir(miner_hotkey) + position_uuid,
            content,
            True,
        )

    @staticmethod
    def get_vali_json_file(vali_dir: str, key: str = None) -> List | Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_file(vali_dir)
            if key is not None:
                return json.loads(secrets)[key]
            else:
                return json.loads(secrets)
        except FileNotFoundError:
            print(f"no vali json file [{dir}], continuing")
            return []
        
    @staticmethod
    def get_miner_eliminations_from_cache() -> Dict:
        return ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
    )
        
    @staticmethod
    def get_miner_copying_from_cache() -> Dict:
        return ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_copy_trading_dir(), ValiUtils.COPY_TRADING
    )

    @staticmethod
    def init_cache_files(metagraph: object):
         
        ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_dir())

        if len(ValiBkpUtils.get_miner_eliminations_from_cache()) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_eliminations_dir(), {ValiUtils.ELIMINATIONS: []}
            )

        if len(ValiUtils.get_vali_json_file(ValiBkpUtils.get_miner_copying_dir())) == 0:
            miner_copying_file = {hotkey: 0 for hotkey in metagraph.hotkeys}
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_miner_copying_dir(), miner_copying_file
            )


    @staticmethod
    def get_last_modified_time_miner_directory(directory_path):
        try:
            # Get the last modification time of the directory
            timestamp = os.path.getmtime(directory_path)
            
            # Convert the timestamp to a datetime object
            last_modified_date = datetime.datetime.fromtimestamp(timestamp)
            
            return last_modified_date
        except FileNotFoundError:
            # Handle the case where the directory does not exist
            print(f"The directory {directory_path} was not found.")
            return None
