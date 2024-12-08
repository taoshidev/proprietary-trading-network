# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time
import traceback

from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController

import bittensor as bt


class MetagraphUpdater(CacheController):
    def __init__(self, config, metagraph, hotkey, is_miner, position_inspector=None, position_manager=None, shutdown_dict=None):
        super().__init__(config, metagraph)
        # Initialize likely validators and miners with empty dictionaries. This maps hotkey to timestamp.
        self.likely_validators = {}
        self.likely_miners = {}
        self.hotkey = hotkey
        if is_miner:
            assert position_inspector is not None, "Position inspector must be provided for miners"
        else:
            assert position_manager is not None, "Position manager must be provided for validators"
        self.is_miner = is_miner
        self.position_inspector = position_inspector
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict  # Flag to control the loop

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

    def run_update_loop(self):
        while not self.shutdown_dict:
            try:
                self.update_metagraph()
            except Exception as e:
                # Handle exceptions or log errors
                bt.logging.error(f"Error during metagraph update: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)
            time.sleep(1)  # Don't busy loop

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

    def update_metagraph(self):
        if not self.refresh_allowed(ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MS):
            return

        recently_acked_miners = None
        recently_acked_validators = None
        if self.is_miner:
            recently_acked_validators = self.position_inspector.get_recently_acked_validators()
        else:
            recently_acked_miners = self.position_manager.get_recently_updated_miner_hotkeys()

        hotkeys_before = self.metagraph.hotkeys
        bt.logging.info("Updating metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)
        hotkeys_after = self.metagraph.hotkeys
        if len(hotkeys_after) < len(hotkeys_before):
            bt.logging.error(f"Metagraph has lost hotkeys: {set(hotkeys_before) - set(hotkeys_after)}"
                             f". Verify this is due to normal deregistration. Otherwise, there could be a serious bug!")
        elif len(hotkeys_after) > len(hotkeys_before):
            bt.logging.info(f"Metagraph has gained hotkeys: {set(hotkeys_after) - set(hotkeys_before)}")
        else:
            bt.logging.info(f"Metagraph hotkeys remain the same. n = {len(hotkeys_after)}")
        if recently_acked_miners:
            self.update_likely_miners(recently_acked_miners)
        if recently_acked_validators:
            self.update_likely_validators(recently_acked_validators)
        self.log_metagraph_state()
        self.set_last_update_time()