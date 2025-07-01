# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time
import traceback

from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController

import bittensor as bt


class MetagraphUpdater(CacheController):
    def __init__(self, config, metagraph, hotkey, is_miner, position_inspector=None, position_manager=None,
                 shutdown_dict=None, slack_notifier=None):
        super().__init__(metagraph)
        self.config = config
        self.subtensor = bt.subtensor(config=self.config)
        # Initialize likely validators and miners with empty dictionaries. This maps hotkey to timestamp.
        self.likely_validators = {}
        self.likely_miners = {}
        self.hotkey = hotkey
        if is_miner:
            assert position_inspector is not None, "Position inspector must be provided for miners"
        self.is_miner = is_miner
        self.interval_wait_time_ms = ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MINER_MS if self.is_miner else \
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_VALIDATOR_MS
        self.position_inspector = position_inspector
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict  # Flag to control the loop
        self.slack_notifier = slack_notifier  # Add slack notifier for error reporting

        # Exponential backoff parameters
        self.min_backoff = 30  # 30 seconds minimum (changed from 1 second)
        self.max_backoff = 43200  # 12 hours maximum (12 * 60 * 60)
        self.backoff_factor = 2  # Double the wait time on each retry
        self.current_backoff = self.min_backoff
        self.consecutive_failures = 0

    def _current_timestamp(self):
        return time.time()

    def _is_expired(self, timestamp):
        return (self._current_timestamp() - timestamp) > 86400  # 24 hours in seconds

    def _get_compact_traceback(self, exception):
        """Get a compact version of the stack trace showing only the most relevant frames."""
        tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)

        # Filter out lines that are less relevant (like internal library calls)
        relevant_lines = []
        for line in tb_lines:
            # Keep the exception line and lines from our code
            if (line.strip().startswith('File') and
                    (any(keyword in line for keyword in ['metagraph', 'miner', 'validator', 'vali_', 'neurons/']) or
                     line.count('/') <= 3)):  # Keep shorter paths (likely our code)
                relevant_lines.append(line.strip())
            elif not line.strip().startswith('File'):
                relevant_lines.append(line.strip())

        # If we filtered too much, just take the last few lines
        if len(relevant_lines) < 3:
            relevant_lines = [line.strip() for line in tb_lines[-3:]]

        return ' | '.join(relevant_lines)

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
                # Reset backoff on successful update
                if self.consecutive_failures > 0:
                    bt.logging.info(
                        f"Metagraph update successful after {self.consecutive_failures} failures. Resetting backoff.")
                    if self.slack_notifier:
                        self.slack_notifier.send_message(
                            f"✅ Metagraph update recovered after {self.consecutive_failures} consecutive failures",
                            level="info"
                        )
                self.consecutive_failures = 0
                self.current_backoff = self.min_backoff
                time.sleep(1)  # Normal operation delay
            except Exception as e:
                self.consecutive_failures += 1
                # Calculate next backoff time
                self.current_backoff = min(self.current_backoff * self.backoff_factor, self.max_backoff)

                # Get compact traceback
                compact_trace = self._get_compact_traceback(e)

                # Log error with backoff information and compact stack trace
                error_msg = (f"Error during metagraph update (attempt #{self.consecutive_failures}): {e}. "
                             f"Next retry in {self.current_backoff} seconds.")
                bt.logging.error(error_msg)
                bt.logging.error(traceback.format_exc())

                if self.slack_notifier:
                    hours = self.current_backoff / 3600
                    self.slack_notifier.send_message(
                        f"❌ Metagraph update failing repeatedly!\n"
                        f"Consecutive failures: {self.consecutive_failures}\n"
                        f"Error: {str(e)}\n"
                        f"Trace: {compact_trace}\n"
                        f"Next retry in: {hours:.2f} hours\n"
                        f"Please check the miner logs!",
                        level="error"
                    )

                # Wait with exponential backoff
                time.sleep(self.current_backoff)

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

        bt.logging.info(
            f"metagraph state (approximation): {n_validators} active validators, {n_miners} active miners, hotkeys: "
            f"{len(self.metagraph.hotkeys)}")

    def sync_lists(self, shared_list, updated_list, brute_force=False):
        if brute_force:
            prev_memory_location = id(shared_list)
            shared_list[:] = updated_list  # Update the proxy list in place without changing the reference
            assert prev_memory_location == id(shared_list), f"Memory location changed after brute force update from {prev_memory_location} to {id(shared_list)}"
            return

        # Convert to sets for fast comparison
        current_set = set(shared_list)
        updated_set = set(updated_list)

        # Find items to remove (in current but not in updated)
        items_to_remove = current_set - updated_set
        # Find items to add (in updated but not in current)
        items_to_add = updated_set - current_set

        # Remove items no longer present
        for item in items_to_remove:
            shared_list.remove(item)

        # Add new items
        for item in items_to_add:
            shared_list.append(item)

    def update_metagraph(self):
        if not self.refresh_allowed(self.interval_wait_time_ms):
            return
        # Searching the BT forums suggests this is needed to recover from a metagraph update failure.
        if self.consecutive_failures > 0:
            self.subtensor = bt.subtensor(config=self.config)

        recently_acked_miners = None
        recently_acked_validators = None
        if self.is_miner:
            recently_acked_validators = self.position_inspector.get_recently_acked_validators()
        else:
            if self.position_manager:
                recently_acked_miners = self.position_manager.get_recently_updated_miner_hotkeys()
            else:
                recently_acked_miners = []

        hotkeys_before = set(self.metagraph.hotkeys)
        metagraph_clone = self.subtensor.metagraph(self.config.netuid)
        assert hasattr(metagraph_clone, 'hotkeys'), "Metagraph clone does not have hotkeys attribute"
        bt.logging.info("Updating metagraph...")
        # metagraph_clone.sync(subtensor=self.subtensor) The call to subtensor.metagraph() already syncs the metagraph.
        hotkeys_after = set(metagraph_clone.hotkeys)
        lost_hotkeys = hotkeys_before - hotkeys_after
        gained_hotkeys = hotkeys_after - hotkeys_before
        if lost_hotkeys:
            bt.logging.info(f"metagraph has lost hotkeys: {lost_hotkeys}")
        if gained_hotkeys:
            bt.logging.info(f"metagraph has gained hotkeys: {gained_hotkeys}")
        if not lost_hotkeys and not gained_hotkeys:
            bt.logging.info(f"metagraph hotkeys remain the same. n = {len(hotkeys_after)}")

        percent_lost = 100 * len(lost_hotkeys) / len(hotkeys_before) if lost_hotkeys else 0
        # failsafe condition to reject new metagraph
        if len(lost_hotkeys) > 10 and percent_lost >= 25:
            error_msg = (f"Too many hotkeys lost in metagraph update: {len(lost_hotkeys)} hotkeys lost, "
                         f"{percent_lost:.2f}% of total hotkeys. Rejecting new metagraph. ALERT A TEAM MEMBER ASAP...")
            bt.logging.error(error_msg)
            if self.slack_notifier:
                self.slack_notifier.send_message(
                    f"🚨 CRITICAL: {error_msg}",
                    level="error"
                )
        elif self.is_miner:
            self.metagraph = metagraph_clone
        else:
            # Multiprocessing (validator only)
            self.sync_lists(self.metagraph.neurons, list(metagraph_clone.neurons), brute_force=True)
            self.sync_lists(self.metagraph.uids, metagraph_clone.uids, brute_force=True)
            self.sync_lists(self.metagraph.hotkeys, metagraph_clone.hotkeys, brute_force=True)
            # Tuple doesn't support item assignment.
            self.sync_lists(self.metagraph.block_at_registration, metagraph_clone.block_at_registration,
                            brute_force=True)

        if recently_acked_miners:
            self.update_likely_miners(recently_acked_miners)
        if recently_acked_validators:
            self.update_likely_validators(recently_acked_validators)
        # self.log_metagraph_state()
        self.set_last_update_time()


# len([x for x in self.metagraph.axons if '0.0.0.0' not in x.ip]), len([x for x in self.metagraph.neurons if '0.0.0.0' not in x.axon_info.for ip])
if __name__ == "__main__":
    from neurons.miner import Miner
    from miner_objects.position_inspector import PositionInspector

    config = Miner.get_config()  # Must run this via commandline to populate correctly
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    position_inspector = PositionInspector(bt.wallet(config=config), metagraph, config)
    mgu = MetagraphUpdater(config, metagraph, "test", is_miner=True, position_inspector=position_inspector)
    while True:
        mgu.update_metagraph()
        time.sleep(60)