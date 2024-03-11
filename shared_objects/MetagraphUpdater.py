import time

from vali_config import ValiConfig
from shared_objects.challenge_utils import ChallengeBase


import bittensor as bt

class MetagraphUpdater(ChallengeBase):
    def __init__(self, config, metagraph):
        super().__init__(config, metagraph)   

    def update_metagraph(self):
        if time.time() - self.get_last_update_time() < ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_S:
            return
        bt.logging.info("Updating metagraph.")
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info(f"Metagraph updated: {self.metagraph}")
        self.set_last_update_time()
