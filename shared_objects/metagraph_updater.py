# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time

from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController

import bittensor as bt


class MetagraphUpdater(CacheController):
    def __init__(self, config, metagraph, hotkey, is_miner):
        super().__init__(config, metagraph)
        # Initialize likely validators and miners with empty dictionaries. This maps hotkey to timestamp.
        self.likely_validators = {}
        self.likely_miners = {}
        self.hotkey = hotkey
        self.is_miner = is_miner

    def _current_timestamp(self):
        return time.time()

    def _is_expired(self, timestamp):
        return (self._current_timestamp() - timestamp) > 86400  # 24 hours in seconds

    def estimate_number_of_validators(self):
        # Filter out expired validators
        self.likely_validators = {k: v for k, v in self.likely_validators.items() if not self._is_expired(v)}
        hotkeys_with_v_trust = set() if self.is_miner else {self.hotkey}
        for neuron in self.metagraph.neurons:
            if neuron.validator_trust > 0:
                hotkeys_with_v_trust.add(neuron.hotkey)
        return len(hotkeys_with_v_trust.union(set(self.likely_validators.keys())))

    def estimate_number_of_miners(self):
        # Filter out expired miners
        self.likely_miners = {k: v for k, v in self.likely_miners.items() if not self._is_expired(v)}
        hotkeys_with_incentive = {self.hotkey} if self.is_miner else set()
        for neuron in self.metagraph.neurons:
            if neuron.incentive > 0:
                hotkeys_with_incentive.add(neuron.hotkey)

        return len(hotkeys_with_incentive.union(set(self.likely_miners.keys())))

    def update_likely_validators(self, hotkeys):
        current_time = self._current_timestamp()
        for h in hotkeys:
            self.likely_validators[h] = current_time

    def update_likely_miners(self, hotkeys):
        current_time = self._current_timestamp()
        for h in hotkeys:
            self.likely_miners[h] = current_time

    def log_metagraph_state(self):
        n_validators = self.estimate_number_of_validators()
        n_miners = self.estimate_number_of_miners()
        if self.is_miner:
            n_miners = max(1, n_miners)
        else:
            n_validators = max(1, n_validators)

        bt.logging.info(f"Metagraph state (approximation): {n_validators} active validators, {n_miners} active miners, hotkeys: "
                        f"{len(self.metagraph.hotkeys)}")

    def update_metagraph(self, recently_acked_miners=None, recently_acked_validators=None):
        if not self.refresh_allowed(ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MS):
            return
        bt.logging.info("Updating metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)
        if recently_acked_miners:
            self.update_likely_miners(recently_acked_miners)
        if recently_acked_validators:
            self.update_likely_validators(recently_acked_validators)
        self.log_metagraph_state()
        self.set_last_update_time()