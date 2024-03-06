from vali_config import ValiConfig


class MinerConfig:

    @staticmethod
    def get_miner_received_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/received_signals/"

    @staticmethod
    def get_miner_sent_signals_dir() -> str:
        return ValiConfig.BASE_DIR + f"/mining/sent_signals/"
