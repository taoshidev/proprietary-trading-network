from vali_config import ValiConfig


class MinerConfig:

    @staticmethod
    def get_miner_received_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/received_signals/"

    @staticmethod
    def get_miner_processed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/processed_signals/"

    @staticmethod
    def get_miner_failed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/failed_signals/"
