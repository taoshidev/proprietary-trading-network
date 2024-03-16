# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import bittensor as bt
from miner_config import MinerConfig
from template.protocol import SendSignal
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

class PropNetOrderPlacer:
    # Constants for retry logic with exponential backoff
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY_SECONDS = 3

    def __init__(self, dendrite, metagraph, config):
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.config = config

    def send_signals(self):
        """
        Initiates the process of sending signals to all validators in parallel.
        This method improves efficiency by leveraging concurrent processing,
        which is especially effective during the initial phase where most signal
        sending attempts are expected to succeed.
        """
        bt.logging.info(f"Number of validators detected: {len(self.metagraph.axons)}.")
        signal_files = ValiBkpUtils.get_all_files_in_dir(MinerConfig.get_miner_received_signals_dir())
        bt.logging.info(f"Total new signals to send: {len(signal_files)}.")

        # Use ThreadPoolExecutor to process signals in parallel
        with ThreadPoolExecutor() as executor:
            # Schedule the processing of each signal file
            futures = [executor.submit(self.process_each_signal, signal_file_path) for signal_file_path in signal_files]
            # as_completed() is used to handle the results as they become available
            for future in as_completed(futures):
                # Logging the completion of signal processing
                bt.logging.info(f"Signal processing completed for: {future.result()}")

        # Wait before the next batch if no signals are found, to avoid busy looping
        if len(signal_files) == 0:
            time.sleep(10)

    def process_each_signal(self, signal_file_path: str):
        """
        Processes each signal file by attempting to send it to the validators.
        Manages retry attempts and employs exponential backoff for failed attempts.
        """
        signal_data = self.load_signal_data(signal_file_path)
        retry_status = {
            signal_file_path: {
                'retry_attempts': 0,
                'retry_delay_seconds': self.INITIAL_RETRY_DELAY_SECONDS,
                'validators_needing_retry': self.metagraph.axons
            }
        }

        # Continue retrying until the max number of retries is reached or no validators need retrying
        while retry_status[signal_file_path]['retry_attempts'] < self.MAX_RETRIES and retry_status[signal_file_path]['validators_needing_retry']:
            self.attempt_to_send_signal(signal_data, signal_file_path, retry_status)

        # If there were validators that failed to process the signal, we move the file to the failed directory
        info = retry_status[signal_file_path]

        # If the config allows it and we have validators needing retry, we move the file to the failed directory
        if info['validators_needing_retry'] and self.config.write_failed_signal_logs:
            self.move_signal_to_failure_directory(signal_file_path, info['validators_needing_retry'])
        else:
            self.move_signal_to_processed_directory(signal_file_path)

        return signal_file_path

    def load_signal_data(self, signal_file_path: str):
        """Loads the signal data from a file."""
        return json.loads(ValiBkpUtils.get_file(signal_file_path), cls=GeneralizedJSONDecoder)

    def attempt_to_send_signal(self, signal_data: object, signal_file_path: str, retry_status: dict):
        """
        Attempts to send a signal to the validators that need retrying, applying exponential backoff for each retry attempt.
        Logs the retry attempt number, and the number of validators that successfully responded out of the total number of original validators.
        """
        current_attempt = retry_status[signal_file_path]['retry_attempts'] + 1  # Increment attempt counter for logging
        total_validators = len(self.metagraph.axons)  # Total number of validators for percentage calculation

        if current_attempt > 1:  # Apply exponential backoff after the first attempt
            time.sleep(retry_status[signal_file_path]['retry_delay_seconds'])
            retry_status[signal_file_path]['retry_delay_seconds'] *= 2  # Double the delay for the next attempt

        send_signal_request = SendSignal(signal=signal_data)
        validator_responses = self.dendrite.query(retry_status[signal_file_path]['validators_needing_retry'],
                                                  send_signal_request, deserialize=True)

        # Filtering validators for the next retry based on the current response
        retry_status[signal_file_path]['validators_needing_retry'] = [
            validator for validator, response in
            zip(retry_status[signal_file_path]['validators_needing_retry'], validator_responses)
            if not response.successfully_processed
        ]

        # Calculating the number of successful responses
        n_fails = len([response for response in validator_responses if not response.successfully_processed])
        # Logging detailed status including the retry attempt, successful responses, and total validators count
        bt.logging.info(
            f"Attempt {current_attempt}: Signal file {signal_file_path} was successfully processed by"
            f" {total_validators - n_fails}/{total_validators} validators.")

        retry_status[signal_file_path]['retry_attempts'] += 1  # Update the retry attempt count for this signal file

    def move_signal_to_processed_directory(self, signal_file_path: str):
        """Moves a processed signal file to the processed directory."""
        self.move_signal_to_directory(MinerConfig.get_miner_processed_signals_dir(), signal_file_path)
    def move_signal_to_failure_directory(self, signal_file_path: str, validators_needing_retry: list):
        # Append the failure information to the signal data.
        json_validator_data = [{'ip': validator.ip, 'port': validator.port, 'ip_type': validator.ip_type,
                                'hotkey': validator.hotkey, 'coldkey': validator.coldkey, 'protocol': validator.protocol}
                               for validator in validators_needing_retry]
        new_data = {'original_signal': self.load_signal_data(signal_file_path),
                    'validators_needing_retry': json_validator_data}

        # Move signal file to the failed directory
        self.move_signal_to_directory(MinerConfig.get_miner_failed_signals_dir(), signal_file_path)

        # Overwrite the file we just moved with the new data
        new_file_path = os.path.join(MinerConfig.get_miner_failed_signals_dir(), os.path.basename(signal_file_path))
        ValiBkpUtils.write_file(new_file_path, json.dumps(new_data))
        bt.logging.info(f"Signal file overwritten with failure information: {new_file_path}")

    def move_signal_to_directory(self, directory: str, signal_file_path):
        ValiBkpUtils.make_dir(directory)
        shutil.move(signal_file_path, os.path.join(directory, os.path.basename(signal_file_path)))
        bt.logging.info(f"Signal file {signal_file_path} has been moved to {directory} ")
