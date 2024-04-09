# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
import os
import threading

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

executor = ThreadPoolExecutor()
trade_pair_id_to_last_order_send = {tp.trade_pair_id: 0 for tp in TradePair}


def signal_to_trade_pair_id(signal):
    return signal['trade_pair']['trade_pair_id']

def order_cooldown_filter(signal):
    # Find out if any signals are for the same trade pair so an exception can be thrown
    trade_pair_id = signal_to_trade_pair_id(signal)
    last_send_time_ms = trade_pair_id_to_last_order_send[trade_pair_id]
    if TimeUtil.now_in_millis() - last_send_time_ms < ValiConfig.ORDER_COOLDOWN_MS:
        time_to_wait_s = (ValiConfig.ORDER_COOLDOWN_MS - (TimeUtil.now_in_millis() - last_send_time_ms)) / 1000
        msg = f"Cannot send signal for trade pair {trade_pair_id} yet. Last order sent at " \
              f"{TimeUtil.millis_to_formatted_date_str(last_send_time_ms)}" \
              f" (cooldown period: {ValiConfig.ORDER_COOLDOWN_MS} ms). " \
              f"Order can be sent in {time_to_wait_s} seconds. Retrying later..."
        bt.logging.error(msg)
        return True
    else:
        return False

def load_signal_data(signal_file_path: str):
    """Loads the signal data from a file."""
    signal_data = json.loads(ValiBkpUtils.get_file(signal_file_path), cls=GeneralizedJSONDecoder)
    return signal_data

def get_all_files_in_dir_no_duplicate_trade_pairs():
    # If there are duplicate trade pairs, only the most recent signal for that trade pair will be sent this round.
    all_files = ValiBkpUtils.get_all_files_in_dir(MinerConfig.get_miner_received_signals_dir())
    bt.logging.warning("Found # of signals to send this round: " + str(len(all_files)))
    temp = {}
    n_files_being_suppressed_this_round = 0
    for f_name in all_files:
        time_of_signal_file = os.path.getmtime(f_name)
        signal = load_signal_data(f_name)
        trade_pair_id = signal['trade_pair']['trade_pair_id']
        if order_cooldown_filter(signal):
            n_files_being_suppressed_this_round += 1
        elif trade_pair_id not in temp:
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

    # Instead of waiting for tasks to complete, schedule a separate task to log their completion.
    def log_completion(self, futures):
        for future in futures:
            # Logging the completion of signal processing
            try:
                result = future.result()  # This will block until the future is completed
                bt.logging.info(f"Signal processing completed for: {result}")
            except Exception as e:
                bt.logging.error(f"Error processing signal: {e}")

    def send_signals(self, recently_acked_validators: list[str]):
        """
        Initiates the process of sending signals to all validators in parallel.
        This method improves efficiency by leveraging concurrent processing,
        which is especially effective during the initial phase where most signal
        sending attempts are expected to succeed.
        """
        self.recently_acked_validators = recently_acked_validators
        signals, signal_file_names, n_files_being_suppressed_this_round = get_all_files_in_dir_no_duplicate_trade_pairs()
        if len(signals) == 0:
            time.sleep(3)  # Prevent busy loop
            return

        bt.logging.info(f"Total new signals to send this round: {len(signals)}. n signals waiting for next round: "
                        f"{n_files_being_suppressed_this_round}")

        # Creating a list to hold references to the futures
        futures = []

        # Submitting tasks to the executor
        for signal_file_path in signal_file_names:
            future = executor.submit(self.process_each_signal, signal_file_path)
            futures.append(future)

        # Run the log_completion function in a separate thread to avoid blocking
        logging_thread = threading.Thread(target=self.log_completion, args=(futures,))
        logging_thread.start()


    def process_each_signal(self, signal_file_path: str):
        """
        Processes each signal file by attempting to send it to the validators.
        Manages retry attempts and employs exponential backoff for failed attempts.
        """
        signal_data = load_signal_data(signal_file_path)
        trade_pair_id = signal_to_trade_pair_id(signal_data)
        trade_pair_id_to_last_order_send[trade_pair_id] = TimeUtil.now_in_millis()
        os.remove(signal_file_path)  # delete from disk to prevent duplicate processing
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
        f_name = signal_file_path.split('/')[-1]
        if all_failure or (info['validators_needing_retry'] and self.config.write_failed_signal_logs):
            self.write_signal_to_failure_directory(signal_data, f_name, info['validators_needing_retry'])
        else:
            self.write_signal_to_processed_directory(signal_data, f_name)

        return signal_file_path

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

        retry_status[signal_file_path]['validators_needing_retry'] = new_validators_to_retry

        # Calculating the number of successful responses
        n_fails = len([response for response in validator_responses if not response.successfully_processed])
        # Logging detailed status including the retry attempt, successful responses, and total validators count
        bt.logging.info(
            f"Attempt {retry_status[signal_file_path]['retry_attempts'] + 1}: Signal file {signal_file_path} was successfully processed by"
            f" {total_n_validators_this_round - n_fails}/{total_n_validators_this_round} possible validators.")

        retry_status[signal_file_path]['retry_attempts'] += 1  # Update the retry attempt count for this signal file

    def write_signal_to_processed_directory(self, signal_data: dict, f_name: str):
        """Moves a processed signal file to the processed directory."""
        new_file_path = os.path.join(MinerConfig.get_miner_processed_signals_dir(), f_name)
        ValiBkpUtils.write_file(new_file_path, json.dumps(signal_data))
        bt.logging.info(f"Signal file modified to include failure information: {new_file_path}")


    def write_signal_to_failure_directory(self, signal_data:dict, f_name: str, validators_needing_retry: list):
        # Append the failure information to the signal data.
        json_validator_data = [{'ip': validator.ip, 'port': validator.port, 'ip_type': validator.ip_type,
                                'hotkey': validator.hotkey, 'coldkey': validator.coldkey, 'protocol': validator.protocol}
                               for validator in validators_needing_retry]
        new_data = {'original_signal': signal_data,
                    'validators_needing_retry': json_validator_data}

        new_file_path = os.path.join(MinerConfig.get_miner_failed_signals_dir(), f_name)
        ValiBkpUtils.write_file(new_file_path, json.dumps(new_data))
        bt.logging.info(f"Signal file modified to include failure information: {new_file_path}")

