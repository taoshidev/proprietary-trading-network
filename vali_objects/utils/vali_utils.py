# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import json
import subprocess
import traceback

from typing import Dict, List

from vali_objects.exceptions.vali_bkp_file_missing_exception import (
    ValiFileMissingException,
)
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import bittensor as bt

class ValiUtils:
    @staticmethod
    def get_secrets() -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_file(ValiBkpUtils.get_secrets_dir())
            return json.loads(secrets)
        except FileNotFoundError:
            raise ValiFileMissingException("Vali secrets file is missing")

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
            print(f"no vali json file [{vali_dir}], continuing")
            return []
        
    @staticmethod
    def get_vali_json_file_dict(vali_dir: str, key: str = None) -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_file(vali_dir)
            if key is not None:
                return json.loads(secrets)[key]
            else:
                return json.loads(secrets)
        except FileNotFoundError:
            print(f"no vali json file [{vali_dir}], continuing")
            return {}
        
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
            print(f"no vali json file [{vali_dir}], continuing")
            return []

