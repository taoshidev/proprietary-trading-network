# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time

from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController


import bittensor as bt

class MetagraphUpdater(CacheController):
    def __init__(self, config, metagraph):
        super().__init__(config, metagraph)   

    def update_metagraph(self):
        if not self.refresh_allowed(ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MS):
            return
        bt.logging.info("Updating metagraph.")
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info(f"Metagraph updated. Hotkeys: {self.metagraph.hotkeys}")
        self.set_last_update_time()
