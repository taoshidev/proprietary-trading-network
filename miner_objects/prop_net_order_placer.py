# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
import os
import threading
import time
import bittensor as bt
from miner_config import MinerConfig
from template.protocol import SendSignal
from vali_objects.vali_config import TradePair
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
REPO_VERSION = 'unknown'
with open(ValiBkpUtils.get_meta_json_path(), 'r') as f:
    REPO_VERSION = json.loads(f.read()).get("subnet_version", "unknown")
class PropNetOrderPlacer:
    # Constants for retry logic with exponential backoff. After trying 3 times, there will be a delay of ~ 3 minutes.
    # This time is sufficient for validators to go offline, update, and come back online.
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY_SECONDS = 20

    def __init__(self, wallet, metagraph, config, is_testnet, position_inspector=None):
        self.wallet = wallet
        self.metagraph = metagraph
        self.config = config
        self.recently_acked_validators = []
        self.is_testnet = is_testnet
        self.trade_pair_id_to_last_order_send = {tp.trade_pair_id: 0 for tp in TradePair}
        self.used_miner_uuids = set()
        self.position_inspector = position_inspector

    def send_signals(self, signals, signal_file_names, recently_acked_validators: list[str]):
        """
        Initiates the process of sending signals to all validators in parallel.
        This method improves efficiency by leveraging concurrent processing,
        which is especially effective during the initial phase where most signal
        sending attempts are expected to succeed.
        """
        self.recently_acked_validators = recently_acked_validators

        threads = []
        for (signal_data, signal_file_path) in zip(signals, signal_file_names):
            thread = threading.Thread(target=self.process_a_signal, args=(signal_file_path, signal_data))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        #for thread in threads:
        #    thread.join()

        #time.sleep(3)

    def process_a_signal(self, signal_file_path, signal_data):
        """
        Processes a signal file by attempting to send it to the validators.
        Manages retry attempts and employs exponential backoff for failed attempts.
        """
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph.neurons}
        axons_to_try = self.position_inspector.get_possible_validators()
        axons_to_try.sort(key=lambda validator: hotkey_to_v_trust[validator.hotkey], reverse=True)

        validator_hotkey_to_axon = {}
        for axon in axons_to_try:
            assert axon.hotkey not in validator_hotkey_to_axon, f"Duplicate hotkey {axon.hotkey} in axons"
            validator_hotkey_to_axon[axon.hotkey] = axon


        retry_status = {
                'retry_attempts': 0,
                'retry_delay_seconds': self.INITIAL_RETRY_DELAY_SECONDS,
                'validators_needing_retry': axons_to_try,
                'validator_error_messages': {},
                'created_orders': {}
        }

        # Track the high-trust validators for special checking after processing
        high_trust_validators = self.get_high_trust_validators(axons_to_try, hotkey_to_v_trust)
        miner_order_uuid = signal_file_path.split('/')[-1]
        assert miner_order_uuid not in self.used_miner_uuids, f"Duplicate miner order uuid {miner_order_uuid}"
        self.used_miner_uuids.add(miner_order_uuid)
        send_signal_request = SendSignal(signal=signal_data, miner_order_uuid=miner_order_uuid, repo_version=REPO_VERSION)

        # Continue retrying until the max number of retries is reached or no validators need retrying
        while retry_status['retry_attempts'] < self.MAX_RETRIES and retry_status['validators_needing_retry']:
            self.attempt_to_send_signal(send_signal_request, retry_status, high_trust_validators, validator_hotkey_to_axon)

        # After retries, check if all high-trust validators have processed the signal successfully
        # This requires checking the current state of trust and response success
        high_trust_processed = True
        n_high_trust_validators = len(high_trust_validators)
        n_high_trust_validators_that_failed = 0
        for validator in high_trust_validators:
            if validator in retry_status['validators_needing_retry']:
                high_trust_processed = False
                n_high_trust_validators_that_failed += 1

        if self.is_testnet and retry_status['validator_error_messages']:
            high_trust_processed = False

        # If there were validators that failed to process the signal, we move the file to the failed directory
        if high_trust_processed:
            self.write_signal_to_processed_directory(signal_data, signal_file_path, retry_status)
        # If there is a validator that hasn't received our order after the max number of retries.
        elif self.config.write_failed_signal_logs:
            v_trust_floor = min([hotkey_to_v_trust[validator.hotkey] for validator in high_trust_validators])
            bt.logging.error(f"Signal file {signal_file_path} was not successfully processed by "
                 f"{n_high_trust_validators_that_failed}/{n_high_trust_validators} high-trust validators. (floor {v_trust_floor})"
                 f" Consider re-sending the signal if this is your first time seeing this error. If this error"
                 f" persists, your miner is eliminated or there is likely an issue with the relevant validator(s) "
                             f"and their vtrust should drop soon.")
            self.write_signal_to_failure_directory(signal_data, signal_file_path, retry_status)
        else:
            self.write_signal_to_processed_directory(signal_data, signal_file_path, retry_status)

        return signal_file_path

    def get_high_trust_validators(self, axons, hotkey_to_v_trust):
        """Returns a list of high-trust validators."""
        high_trust_validators = [ax for ax in axons if hotkey_to_v_trust[ax.hotkey] >= MinerConfig.HIGH_V_TRUST_THRESHOLD]
        if not high_trust_validators:
            if not self.is_testnet:
                bt.logging.error("No high-trust validators found. This is unexpected in mainnet. Please report to the team ASAP.")
            return axons
        else:
            return high_trust_validators


    def attempt_to_send_signal(self, send_signal_request: SendSignal, retry_status: dict, high_trust_validators: list, validator_hotkey_to_axon: dict):
        """
        Attempts to send a signal to the validators that need retrying, applying exponential backoff for each retry attempt.
        Logs the retry attempt number, and the number of validators that successfully responded out of the total number of original validators.
        """
        # Total number of axons being pinged this round. Used for percentage calculation
        # total_n_validators_this_round = len(retry_status['validators_needing_retry'])
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph.neurons}

        bt.logging.info(f"Attempt #{retry_status['retry_attempts']} for {send_signal_request.signal['trade_pair']['trade_pair_id']} uuid {send_signal_request.miner_order_uuid}."
                        f" Sending order to {len(retry_status['validators_needing_retry'])} hotkeys...")

        if retry_status['retry_attempts'] != 0:  # Apply exponential backoff after the first attempt
            time.sleep(retry_status['retry_delay_seconds'])
            retry_status['retry_delay_seconds'] *= 2  # Double the delay for the next attempt

        dendrite = bt.dendrite(wallet=self.wallet)
        validator_responses = dendrite.query(retry_status['validators_needing_retry'], send_signal_request)
        #validator_responses = dendrite(retry_status['validators_needing_retry'], send_signal_request)

        # Filtering validators for the next retry based on the current response.
        all_high_trust_validators_succeeded = True

        success_validators = set([response.validator_hotkey for response in validator_responses if
                                  response.successfully_processed and response.validator_hotkey])

        # Loop through responses for error messaging
        for response in validator_responses:
            acked_axon = validator_hotkey_to_axon.get(response.validator_hotkey)

            if response.successfully_processed:
                retry_status['created_orders'][acked_axon.hotkey] = response.order_json
                continue

            acked_axon = validator_hotkey_to_axon.get(response.validator_hotkey)
            vtrust = hotkey_to_v_trust.get(response.validator_hotkey)
            if acked_axon in high_trust_validators:
                all_high_trust_validators_succeeded = False
                if response.error_message:
                    msg = f"Error sending order to axon {acked_axon} with v_trust {vtrust}. Error message: {response.error_message}"
                    bt.logging.warning(msg)
                    if acked_axon.hotkey not in retry_status['validator_error_messages']:
                        retry_status['validator_error_messages'][acked_axon.hotkey] = []
                    retry_status['validator_error_messages'][acked_axon.hotkey].append(response.error_message)


        if all_high_trust_validators_succeeded:
            v_trust_floor = min([hotkey_to_v_trust[validator.hotkey] for validator in high_trust_validators])
            n_high_trust_validators = len(high_trust_validators)
            bt.logging.success(f"Signal file {send_signal_request.signal} was successfully processed by"
                               f" {n_high_trust_validators}/{n_high_trust_validators} high-trust validators with "
                               f"min v_trust {v_trust_floor}. "
                               f"Total n_validators: {len(retry_status['validators_needing_retry'])}. "
                               f" Created orders': {retry_status['created_orders']}")

        def _allow_retry(axon):
            if axon.hotkey in success_validators:
                return False
            # Do not retry if the validator has 0 trust and is not in the recently acked list.
            # Maybe another miner or inactive hotkey.
            if axon.hotkey in self.recently_acked_validators:
                return True
            return hotkey_to_v_trust[axon.hotkey] > 0

        new_validators_to_retry = [axon for axon in retry_status['validators_needing_retry'] if _allow_retry(axon)]
        # Sort the new list of axons needing retry by trust, highest to lowest to reduce possible lag
        new_validators_to_retry.sort(key=lambda validator: hotkey_to_v_trust[validator.hotkey], reverse=True)

        retry_status['validators_needing_retry'] = new_validators_to_retry

        # Calculating the number of successful responses
        #n_fails = len([response for response in validator_responses if not response.successfully_processed])
        retry_status['retry_attempts'] += 1  # Update the retry attempt count for this signal file

    def write_signal_to_processed_directory(self, signal_data, signal_file_path: str, retry_status: dict):
        """Moves a processed signal file to the processed directory."""
        signal_copy = signal_data.copy()
        signal_copy['trade_pair'] = signal_copy['trade_pair']['trade_pair_id']
        data_to_write = {'signal_data': signal_copy, 'created_orders': retry_status['created_orders']}
        self.write_signal_to_directory(MinerConfig.get_miner_processed_signals_dir(), signal_file_path, data_to_write, True)

    def write_signal_to_failure_directory(self, signal_data, signal_file_path: str, retry_status: dict):
        validators_needing_retry = retry_status['validators_needing_retry']
        error_messages_dict = retry_status['validator_error_messages']
        created_orders = retry_status['created_orders']
        # Append the failure information to the signal data.
        json_validator_data = [{'ip': validator.ip, 'port': validator.port, 'ip_type': validator.ip_type,
                                'hotkey': validator.hotkey, 'coldkey': validator.coldkey, 'protocol': validator.protocol}
                               for validator in validators_needing_retry]
        new_data = {'original_signal': signal_data,
                    'validators_needing_retry': json_validator_data,
                    'error_messages_dict': error_messages_dict,
                    'created_orders': created_orders}

        # Move signal file to the failed directory
        self.write_signal_to_directory(MinerConfig.get_miner_failed_signals_dir(), signal_file_path, signal_data, False)

        # Overwrite the file we just moved with the new data
        new_file_path = os.path.join(MinerConfig.get_miner_failed_signals_dir(), os.path.basename(signal_file_path))
        ValiBkpUtils.write_file(new_file_path, json.dumps(new_data))
        new_data_compact = {k: v for k, v in new_data.items() if k != 'validators_needing_retry'}
        bt.logging.info(f"Signal file modified to include failure information: {new_file_path}. Data dump: {new_data_compact}")

    def write_signal_to_directory(self, directory: str, signal_file_path, signal_data, success):
        ValiBkpUtils.make_dir(directory)
        new_path = os.path.join(directory, os.path.basename(signal_file_path))
        with open(new_path, 'w') as f:
            f.write(json.dumps(signal_data))
        filename = os.path.basename(signal_file_path)
        new_fullpath = os.path.join(directory, filename)
        msg = f"Signal file {signal_file_path} has been written to {new_fullpath} "
        if success:
            bt.logging.success(msg)
        else:
            bt.logging.error(msg)

