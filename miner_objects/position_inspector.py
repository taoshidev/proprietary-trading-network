from collections import defaultdict

import bittensor as bt
import time
from template.protocol import GetPositions


class PositionInspector:
    MAX_RETRIES = 1
    INITIAL_RETRY_DELAY = 3  # seconds
    UPDATE_INTERVAL_S = 5 * 60  # 5 minutes

    def __init__(self, dendrite, metagraph, config):
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.config = config
        self.last_update_time = 0
        self.recently_acked_validators = []

    def get_possible_validators(self):
        # Right now bittensor has no functionality to know if a hotkey 100% corresponds to a validator
        # Revisit this in the future.
        return self.metagraph.axons

    def query_positions(self, validators, hotkey_to_positions):
        remaining_validators_to_query = [v for v in validators if v.hotkey not in hotkey_to_positions]
        responses = self.dendrite.query(remaining_validators_to_query, GetPositions(), deserialize=True)
        return [(validator, response.positions) for validator, response in zip(remaining_validators_to_query, responses) if
                response.successfully_processed]

    def reconcile_validator_positions(self, hotkey_to_positions, validators):
        hotkey_to_validator = {v.hotkey: v for v in validators}
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph.neurons}
        orders_count = defaultdict(int)
        max_order_count = 0
        corresponding_positions = []
        corresponding_hotkey = None
        for hotkey, positions in hotkey_to_positions.items():
            for position in positions:
                orders_count[hotkey] += len(position['orders'])
                if orders_count[hotkey] > max_order_count:
                    max_order_count = orders_count[hotkey]
                    corresponding_positions = position
                    corresponding_hotkey = hotkey

        unique_counts = set(orders_count.values())

        if len(unique_counts) > 1:
            for hotkey, count in orders_count.items():
                axon_info = hotkey_to_validator[hotkey]
                bt.logging.warning(f"Validator {hotkey} has {count} orders with v_trust {hotkey_to_v_trust[hotkey]}, axon: {axon_info}. "
                                   f"Validators may be mis-synced.")

        # Return the position in hotkey_to_positions that has the most orders
        bt.logging.info(f"Validator with the most orders: {corresponding_hotkey}, n_orders: {max_order_count}, v_trust:"
                        f" {hotkey_to_v_trust.get(corresponding_hotkey, 0)}")
        return corresponding_positions

    def get_positions_with_retry(self, validators_to_query):
        attempts = 0
        delay = self.INITIAL_RETRY_DELAY
        hotkey_to_positions = {}
        while attempts < self.MAX_RETRIES and len(hotkey_to_positions) != len(validators_to_query):
            if attempts > 0:
                time.sleep(delay)
                delay *= 2  # Exponential backoff

            positions = self.query_positions(validators_to_query, hotkey_to_positions)
            for p in positions:
                hotkey_to_positions[p[0].hotkey] = p[1]

            attempts += 1

        bt.logging.info(f"Got positions from {len(hotkey_to_positions)} validators")

        self.recently_acked_validators = hotkey_to_positions.keys()
        position_most_orders = self.reconcile_validator_positions(hotkey_to_positions, validators_to_query)
        # Return the validator with the most orders
        return position_most_orders

    def refresh_allowed(self):
        return (time.time() - self.last_update_time) > self.UPDATE_INTERVAL_S

    def log_validator_positions(self):
        """
        Sends signals to the validators to get their time-sorted positions for this miner.
        This method may be used directly in your own logic to attempt to "fix" validator positions.
        Note: The rate limiter on validators will prevent repeated calls from succeeding if they are too frequent.
        """
        if not self.refresh_allowed():
            return

        validators_to_query = self.get_possible_validators()
        bt.logging.info(f"Querying {len(validators_to_query)} possible validators for positions")
        result = self.get_positions_with_retry(validators_to_query)

        if not result:
            bt.logging.info("No positions found.")

        self.last_update_time = time.time()
        bt.logging.success(f"PositionInspector successfully completed signal processing.")
