import bittensor as bt
import time
from template.protocol import GetPositions


class PositionInspector:
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 3  # seconds
    UPDATE_INTERVAL_S = 5 * 60  # 5 minutes
    TOP_N = 3  # Number of validators to query

    def __init__(self, dendrite, metagraph, config):
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.config = config
        self.last_update_time = 0

    def get_validators_by_stake(self):
        stakes = [x.item() for x in self.metagraph.stake]
        validators = self.metagraph.axons
        return sorted(validators, key=lambda v: stakes[validators.index(v)], reverse=True)[:self.TOP_N]

    def query_positions(self, top_validators, hotkey_to_positions):
        validators_to_query = [v for v in top_validators if v.hotkey not in hotkey_to_positions]
        responses = self.dendrite.query(validators_to_query, GetPositions(), deserialize=True)
        return [(validator, response.positions) for validator, response in zip(validators_to_query, responses) if
                response.successfully_processed]

    def get_validator_response_with_most_positions(self, hotkey_to_positions):
        if not hotkey_to_positions:
            return None

        return max(hotkey_to_positions.items(), key=lambda x: len(x[1]))

    def log_position_discrepancies(self, hotkey_to_positions, top_validators):
        hotkey_to_validator = {v.hotkey: v for v in top_validators}
        positions_count = {hotkey: len(positions) for hotkey, positions in hotkey_to_positions.items()}
        unique_counts = set(positions_count.values())

        if len(unique_counts) > 1:
            for hotkey, count in positions_count.items():
                axon_info = hotkey_to_validator[hotkey]
                bt.logging.warning(f"Validator {hotkey} has {count} positions, axon: {axon_info}. "
                                   f"Validators may be mis-synced.")

    def get_positions_with_retry(self, top_validators):
        attempts = 0
        delay = self.INITIAL_RETRY_DELAY
        hotkey_to_positions = {}
        while attempts < self.MAX_RETRIES and len(hotkey_to_positions) != self.TOP_N:
            if attempts > 0:
                time.sleep(delay)
                delay *= 2  # Exponential backoff

            positions = self.query_positions(top_validators, hotkey_to_positions)
            for p in positions:
                hotkey_to_positions[p[0].hotkey] = p[1]

            attempts += 1

        if len(hotkey_to_positions) != self.TOP_N:
            # Log how many validators failed to respond
            bt.logging.warning(
                f"Failed to get positions from {len(top_validators) - len(hotkey_to_positions)} out of {len(top_validators)} validators. Continuing...")

        self.log_position_discrepancies(hotkey_to_positions, top_validators)
        # Return the validator with the most positions
        return self.get_validator_response_with_most_positions(hotkey_to_positions)

    def refresh_allowed(self):
        return (time.time() - self.last_update_time) > self.UPDATE_INTERVAL_S

    def send_signals_with_cooldown(self):
        """
        Sends signals to the validators to get their time sorted positions for this miner. If you want to call this
        without a cooldown, use the method "get_positions_with_retry" directly. Note: The rate limiter on validators
        may prevent repeated calls from succeeding if they are too frequent.
        """
        if not self.refresh_allowed():
            return

        validators_to_query = self.get_validators_by_stake()
        bt.logging.info(f"Querying validators for positions: {validators_to_query}")
        result = self.get_positions_with_retry(validators_to_query)

        if result:
            validator, positions = result
            bt.logging.info(f"Validator with the most positions: {validator}")
            bt.logging.info(f"Positions: {positions}")
        else:
            bt.logging.info("No positions found.")

        self.last_update_time = time.time()
        bt.logging.success(f"PositionInspector successfully completed signal processing.")
