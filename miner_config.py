import os
from vali_objects.vali_config import ValiConfig


BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

class MinerConfig:
    HIGH_V_TRUST_THRESHOLD = 0.75
    STAKE_MIN = 1000  # Do not change from int
    AXON_NO_IP = "0.0.0.0"

    DASHBOARD_API_PORT = 41511
    COLLATERAL_API_PORT = 41521
    BASE_DIR = base_directory = BASE_DIR
    
    # Collateral system configuration
    COLLATERAL_ADMIN_VALIDATOR_HOTKEY_TESTNET = "5FbKk7fD33ceRDADDdpj5ULRKZMca3XW3Psoxg69iTnVjurS"
    COLLATERAL_ADMIN_VALIDATOR_VAULT_TESTNET = "5FbsFNnBTTyKbZPWXixk5yweSjYGrUVhUbWpmC2XZQhi8uco"

    @staticmethod
    def get_miner_received_signals_dir() -> str:
        return ValiConfig.BASE_DIR + "/mining/received_signals/"

    @staticmethod
    def get_miner_processed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + "/mining/processed_signals/"

    @staticmethod
    def get_miner_failed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + "/mining/failed_signals/"
    
    @staticmethod
    def get_primary_validator_hotkey(network: str) -> str:
        """Get collateral admin validator hotkey"""
        if network == "test":
            return MinerConfig.COLLATERAL_ADMIN_VALIDATOR_HOTKEY_TESTNET
    
    @staticmethod
    def get_primary_validator_vault(network: str) -> str:
        """Get collateral admin validator vault address"""
        if network == "test":
            return MinerConfig.COLLATERAL_ADMIN_VALIDATOR_VAULT_TESTNET
