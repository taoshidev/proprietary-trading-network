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

    @staticmethod
    def force_validator_to_restore_from_checkpoint(validator_hotkey, metagraph, config, secrets):
        try:
            if "mothership" in secrets:
                bt.logging.warning(f"Validator {validator_hotkey} is the mothership. Not forcing restore.")
                return

            if config.subtensor.network == "test":  # Only need do this in mainnet
                bt.logging.warning("Not forcing validator to restore from checkpoint in testnet.")
                return

            hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in metagraph.neurons}
            my_trust = hotkey_to_v_trust.get(validator_hotkey)
            if my_trust is None:
                bt.logging.warning(f"Validator {validator_hotkey} not found in metagraph. Cannot determine trust.")
                return

            # Good enough
            if my_trust > 0.5:
                return

            bt.logging.warning(f"Validator {validator_hotkey} trust is {my_trust}. Forcing restore.")
            command = """
            curl https://dashboard.taoshi.io/api/validator-checkpoint -o validator_checkpoint.json &&
            sed -i 's/^{"checkpoint"://' validator_checkpoint.json &&
            sed -i 's/}$//' validator_checkpoint.json &&
            python3 restore_validator_from_backup.py
            """
            subprocess.run(command, shell=True, check=True)

        except Exception as e:
            bt.logging.error(f"Error forcing validator to restore from checkpoint: {e}")
            bt.logging.error(traceback.format_exc())

