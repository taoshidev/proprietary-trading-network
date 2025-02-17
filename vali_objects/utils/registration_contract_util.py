import bittensor as bt
import datetime

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController


class RegistrationContractUtil:
    def __init__(self):
        pass

    @staticmethod
    def get_registered() -> list[str]:
        registration_contract_address = ValiConfig.REGISTRATION_CONTRACT_ADDRESS
        # registered_hotkeys = contract_interaction(registration_contract_address)
        return []  # registered_hotkeys

    @staticmethod
    def deregister_hotkey(hotkey: str):
        # submit request to the contract to deregister the hotkey
        contract_address = ValiConfig.REGISTRATION_CONTRACT_ADDRESS

        # send address to contract
        # contract_interaction(contract_address, hotkey)
        bt.logging.info(f"Hotkey {hotkey} deregistered at time {datetime.datetime.fromtimestamp(TimeUtil.now_in_millis()/1000)}")

    @staticmethod
    def deregister_hotkeys(hotkeys: list[str]):
        for hotkey in hotkeys:
            RegistrationContractUtil.deregister_hotkey(hotkey)

        return True
