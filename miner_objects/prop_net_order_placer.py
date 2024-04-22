# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import bittensor as bt
from miner_config import MinerConfig
from template.protocol import SendSignal
from time_util.time_util import TimeUtil
from vali_config import TradePair, ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

class PropNetOrderPlacer:
    # Constants for retry logic with exponential backoff. After trying 3 times, there will be a delay of ~ 2 minutes.
    # This time is sufficient for validators to go offline, update, and come back online.
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY_SECONDS = 15

    def __init__(self, dendrite, metagraph, config):
        self.dendrite = dendrite
        self.metagraph = metagraph
        self.config = config
        self.recently_acked_validators = []
        self.trade_pair_id_to_last_order_send = {tp.trade_pair_id: 0 for tp in TradePair}

    def order_cooldown_check(self, signals):
        # Find out if any signals are for the same trade pair so an exception can be thrown

        for s in signals:
            trade_pair_id = s['trade_pair']['trade_pair_id']
            last_send_time_ms = self.trade_pair_id_to_last_order_send[trade_pair_id]
            if TimeUtil.now_in_millis() - last_send_time_ms < ValiConfig.ORDER_COOLDOWN_MS:
                time_to_wait_s = (ValiConfig.ORDER_COOLDOWN_MS - (TimeUtil.now_in_millis() - last_send_time_ms)) / 1000
                raise Exception(f"Cannot send signal for trade pair {trade_pair_id} yet. Last order sent at "
                                f"{TimeUtil.millis_to_formatted_date_str(last_send_time_ms)}"
                                f" (cooldown period: {ValiConfig.ORDER_COOLDOWN_MS} ms). "
                                f"Order can be sent in {time_to_wait_s} seconds. Retrying...")
            self.trade_pair_id_to_last_order_send[trade_pair_id] = TimeUtil.now_in_millis()

    def get_all_files_in_dir_no_duplicate_trade_pairs(self):
        # If there are duplicate trade pairs, only the most recent signal for that trade pair will be sent this round.
        all_files = ValiBkpUtils.get_all_files_in_dir(MinerConfig.get_miner_received_signals_dir())
        temp = {}
        n_files_being_suppressed_this_round = 0
        for f_name in all_files:
            signal = self.load_signal_data(f_name)
            time_of_signal_file = os.path.getmtime(f_name)
            trade_pair_id = signal['trade_pair']['trade_pair_id']
            if trade_pair_id not in temp:
                temp[trade_pair_id] = (signal, f_name, time_of_signal_file)
            else:
                if temp[trade_pair_id][2] < time_of_signal_file:
                    temp[trade_pair_id] = (signal, f_name, time_of_signal_file)
                    bt.logging.info(f"Found duplicate signals for trade pair {trade_pair_id}."
                                    f" Will save this signal for next round: {temp[trade_pair_id][1]}")
                    n_files_being_suppressed_this_round += 1
                elif temp[trade_pair_id][2] == time_of_signal_file:
                    raise Exception(f"MANUAL INTERVENTION REQUIRED. Multiple signals found for the same trade pair with "
                                    f"the same timestamp. "
                                    f"Preventing sending as orders may not be processed in the intended order."
                                    f"Relevant signal file: {f_name}")
        # Return all signals as a list
        return [x[0] for x in temp.values()], [x[1] for x in temp.values()], n_files_being_suppressed_this_round

    def send_signals(self, recently_acked_validators: list[str]):
        """
        Initiates the process of sending signals to all validators in parallel.
        This method improves efficiency by leveraging concurrent processing,
        which is especially effective during the initial phase where most signal
        sending attempts are expected to succeed.
        """
        self.recently_acked_validators = recently_acked_validators
        signals, signal_file_names, n_files_being_suppressed_this_round = self.get_all_files_in_dir_no_duplicate_trade_pairs()
        #self.order_cooldown_check(signals)
        if len(signals) == 0 and n_files_being_suppressed_this_round == 0 and int(time.time()) % 180 == 0:
            bt.logging.info(f"No signals found... will continue trying every second.")
        elif len(signals) > 0 or n_files_being_suppressed_this_round > 0:
            bt.logging.info(f"Total new signals to send this round: {len(signals)}. n signals waiting for next round: "
                            f"{n_files_being_suppressed_this_round}")

            # Use ThreadPoolExecutor to process signals in parallel
            with ThreadPoolExecutor() as executor:
                # Schedule the processing of each signal file
                futures = [executor.submit(self.process_each_signal, signal_file_path) for signal_file_path in
                           signal_file_names]
                # as_completed() is used to handle the results as they become available
                for future in as_completed(futures):
                    # Logging the completion of signal processing
                    bt.logging.info(f"Signal processing completed for: {future.result()}")

        # Wait before the next batch if no signals are found, to avoid busy looping
        if len(signals) == 0:
            time.sleep(1)

    def process_each_signal(self, signal_file_path: str):
        """
        Processes each signal file by attempting to send it to the validators.
        Manages retry attempts and employs exponential backoff for failed attempts.
        """
        signal_data = self.load_signal_data(signal_file_path)
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph.neurons}
        axons_to_try = self.metagraph.axons
        axons_to_try.sort(key=lambda validator: hotkey_to_v_trust[validator.hotkey], reverse=True)

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

        all_failure = len(info['validators_needing_retry']) == len(self.metagraph.axons)
        # If there is a validator that hasn't received our order after the max number of retries.
        if all_failure or (info['validators_needing_retry'] and self.config.write_failed_signal_logs):
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
        # Total number of axons being pinged this round. Used for percentage calculation
        total_n_validators_this_round = len(retry_status[signal_file_path]['validators_needing_retry'])
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph.neurons}

        if retry_status[signal_file_path]['retry_attempts'] != 0:  # Apply exponential backoff after the first attempt
            time.sleep(retry_status[signal_file_path]['retry_delay_seconds'])
            retry_status[signal_file_path]['retry_delay_seconds'] *= 2  # Double the delay for the next attempt

        send_signal_request = SendSignal(signal=signal_data)
        dendrite_copy = deepcopy(self.dendrite)  # So that we can query in parallel across multiple threads
        validator_responses = dendrite_copy.query(retry_status[signal_file_path]['validators_needing_retry'],
                                                  send_signal_request, deserialize=True)

        # Filtering validators for the next retry based on the current response.
        new_validators_to_retry = []
        for validator, response in zip(retry_status[signal_file_path]['validators_needing_retry'], validator_responses):
            eliminated = "has been eliminated" in response.error_message
            if not response.successfully_processed:
                if response.error_message:
                    bt.logging.error(f"Error sending order to {validator}. Error message: {response.error_message}")
                if eliminated:
                    continue
                if hotkey_to_v_trust[validator.hotkey] > 0:
                    new_validators_to_retry.append(validator)
                elif validator.hotkey in self.recently_acked_validators:
                    new_validators_to_retry.append(validator)
                else:
                    # Do not retry if the validator has 0 trust and is not in the recently acked list. Maybe another miner or inactive hotkey.
                    pass

        # Sort the new list of axons needing retry by trust, highest to lowest to reduce possible lag
        new_validators_to_retry.sort(key=lambda validator: hotkey_to_v_trust[validator.hotkey], reverse=True)

        retry_status[signal_file_path]['validators_needing_retry'] = new_validators_to_retry

        # Calculating the number of successful responses
        n_fails = len([response for response in validator_responses if not response.successfully_processed])
        # Logging detailed status including the retry attempt, successful responses, and total validators count
        bt.logging.info(
            f"Attempt {retry_status[signal_file_path]['retry_attempts'] + 1}: Signal file {signal_file_path} was successfully processed by"
            f" {total_n_validators_this_round - n_fails}/{total_n_validators_this_round} possible validators.")

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
        bt.logging.info(f"Signal file modified to include failure information: {new_file_path}")

    def move_signal_to_directory(self, directory: str, signal_file_path):
        ValiBkpUtils.make_dir(directory)
        shutil.move(signal_file_path, os.path.join(directory, os.path.basename(signal_file_path)))
        bt.logging.info(f"Signal file {signal_file_path} has been moved to {directory} ")
