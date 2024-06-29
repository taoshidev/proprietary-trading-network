from vali_config import ValiConfig


class MinerConfig:
    HIGH_V_TRUST_THRESHOLD = 0.75
    STAKE_MIN = 1000.0
    AXON_NO_IP = "0.0.0.0"

    @staticmethod
    def get_miner_received_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/received_signals/"

    @staticmethod
    def get_miner_processed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/processed_signals/"

    @staticmethod
    def get_miner_failed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/failed_signals/"