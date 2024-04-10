# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import os
import threading
import uuid
from typing import Tuple

import template
import argparse
import traceback
import time
import bittensor as bt

from shared_objects.rate_limiter import RateLimiter
from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.metagraph_updater import MetagraphUpdater
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.position import Position
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig


class Validator:
    def __init__(self):
        self.config = self.get_config()
        self.is_mainnet = self.config.netuid == 8
        # Ensure the directory for logging exists, else create one.
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

        self.secrets = ValiUtils.get_secrets()
        if self.secrets is None:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )

        self.live_price_fetcher = LivePriceFetcher(secrets=self.secrets)
        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} "
            f"on network: {self.config.subtensor.chain_endpoint} with config:"
        )

        # This logs the active configuration to the specified logging directory for review.
        bt.logging.info(self.config)

        # Initialize Bittensor miner objects
        # These classes are vital to interact and function within the Bittensor network.
        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {wallet}")

        # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
        subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {subtensor}")


        # metagraph provides the network's current state, holding state about other participants in a subnet.
        # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
        self.metagraph = subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")
        self.position_manager = PositionManager(metagraph=self.metagraph, config=self.config,
                                                perform_price_adjustment=True,
                                                live_price_fetcher=self.live_price_fetcher,
                                                perform_fee_structure_update=False,
                                                perform_order_corrections=False)

        self.metagraph_updater = MetagraphUpdater(self.config, self.metagraph, wallet.hotkey.ss58_address,
                                                  False, position_manager=self.position_manager)

        # Start the metagraph updater loop in its own thread
        self.updater_thread = threading.Thread(target=self.metagraph_updater.run_update_loop, daemon=True)
        self.updater_thread.start()


        if wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {wallet} is not registered to chain "
                f"connection: {subtensor} \nRun btcli register and try again. "
            )
            exit()

        # Build and link vali functions to the axon.
        # The axon handles request processing, allowing validators to send this process requests.
        bt.logging.info(f"setting port [{self.config.axon.port}]")
        bt.logging.info(f"setting external port [{self.config.axon.external_port}]")
        self.axon = bt.axon(
            wallet=wallet, port=self.config.axon.port, external_port=self.config.axon.external_port
        )
        bt.logging.info(f"Axon {self.axon}")

        # Attach determines which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to axon.")

        self.rate_limiter = RateLimiter()

        def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def rs_priority_fn(synapse: template.protocol.SendSignal) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def gp_blacklist_fn(synapse: template.protocol.GetPositions) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def gp_priority_fn(synapse: template.protocol.GetPositions) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        self.axon.attach(
            forward_fn=self.receive_signal,
            blacklist_fn=rs_blacklist_fn,
            priority_fn=rs_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.get_positions,
            blacklist_fn=gp_blacklist_fn,
            priority_fn=gp_priority_fn,
        )

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving attached axons on network:"
            f" {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=subtensor)

        # Starts the miner's axon, making it active on the network.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

        # see if cache files exist and if not set them to empty
        self.position_manager.init_cache_files()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

        # Eliminations are read in validator, elimination_manager, mdd_checker, weight setter.
        # Eliminations are written in elimination_manager, mdd_checker
        # Since the mainloop is run synchronously, we just need to lock eliminations when writing to them and when
        # reading outside of the mainloop (validator).
        self.eliminations_lock = threading.Lock()
        # self.plagiarism_detector = PlagiarismDetector(self.config, self.metagraph)
        self.mdd_checker = MDDChecker(self.config, self.metagraph, self.position_manager, self.eliminations_lock)
        self.weight_setter = SubtensorWeightSetter(self.config, wallet, self.metagraph)
        self.elimination_manager = EliminationManager(self.metagraph, self.position_manager, self.eliminations_lock)

        # Validators on mainnet net to be syned for the first time or after interruption need to resync their
        # positions. Assert there are existing orders that occurred > 24hrs in the past. Assert that the newest order
        # was placed within 24 hours.
        if self.is_mainnet:
            n_positions_on_disk = self.position_manager.get_number_of_miners_with_any_positions()
            smallest_disk_ms, largest_disk_ms = (
                self.position_manager.get_extreme_position_order_processed_on_disk_ms())
            if (n_positions_on_disk > 0):
                bt.logging.info(f"Found {n_positions_on_disk} positions on disk."
                                f" Found youngest_disk_ms: {TimeUtil.millis_to_datetime(smallest_disk_ms)},"
                                f" oldest_disk_ms: {TimeUtil.millis_to_datetime(largest_disk_ms)}")
            if (n_positions_on_disk == 0 or
                    smallest_disk_ms > TimeUtil.timestamp_to_millis(TimeUtil.generate_start_timestamp(days=1)) or
                    largest_disk_ms < TimeUtil.timestamp_to_millis(TimeUtil.generate_start_timestamp(days=1))):
                msg = (f"Validator data needs to be synced with mainnet. Please restore data from checkpoint "
                       f"before running the validator. More info here: "
                       f"https://github.com/taoshidev/proprietary-trading-network/"
                       f"blob/main/docs/regenerating_validator_state.md")
                bt.logging.error(msg)
                raise Exception(msg)



    @staticmethod
    def blacklist_fn(synapse, metagraph) -> Tuple[bool, str]:
        miner_hotkey = synapse.dendrite.hotkey
        # Ignore requests from unrecognized entities.
        if miner_hotkey not in metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    @staticmethod
    def priority_fn(synapse, metagraph) -> float:
        # simply just prioritize based on uid as it's not significant
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.uids[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def get_config(self):
        # Step 2: Set up the configuration parser
        # This function initializes the necessary command-line arguments.
        # Using command-line arguments allows users to customize various miner settings.
        parser = argparse.ArgumentParser()
        # TODO(developer): Adds your custom miner arguments to the parser.
        # Adds override arguments for network and netuid.
        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
        # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.wallet.add_args(parser)
        # Adds axon specific arguments i.e. --axon.port ...
        bt.axon.add_args(parser)
        # Activating the parser to read any command-line inputs.
        # To print help message, run python3 template/miner.py --help
        config = bt.config(parser)

        # Step 3: Set up logging directory
        # Logging captures events for diagnosis or understanding miner's behavior.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "miner",
            )
        )
        return config

    # Main takes the config and starts the miner.
    def main(self):
        # Keep the vali alive. This loop maintains the vali's operations until intentionally stopped.
        bt.logging.info(f"Starting main loop")
        while True:
            try:
                self.mdd_checker.mdd_check()
                self.weight_setter.set_weights()
                self.elimination_manager.process_eliminations()
                self.position_manager.position_locks.cleanup_locks(self.metagraph.hotkeys)

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Validator killed by keyboard interrupt.")
                self.metagraph_updater.stop_update_loop()
                self.updater_thread.join()
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception:
                bt.logging.error(traceback.format_exc())
                time.sleep(10)


    def parse_trade_pair_from_signal(self, signal) -> TradePair | None:
        if not signal or not isinstance(signal, dict):
            return None
        if 'trade_pair' not in signal:
            return None
        temp = signal["trade_pair"]
        if 'trade_pair_id' not in temp:
            return None
        string_trade_pair = signal["trade_pair"]["trade_pair_id"]
        trade_pair = TradePair.from_trade_pair_id(string_trade_pair)
        return trade_pair

    def convert_signal_to_order(self, signal, hotkey) -> Order:
        """
        Example input signal
          {'trade_pair': {'trade_pair_id': 'BTCUSD', 'trade_pair': 'BTC/USD', 'fees': 0.003, 'min_leverage': 0.0001, 'max_leverage': 20},
          'order_type': 'LONG',
          'leverage': 0.5}
        """
        trade_pair = self.parse_trade_pair_from_signal(signal)
        if trade_pair is None:
            bt.logging.error(f"[{trade_pair}] not in TradePair enum.")
            raise SignalException(
                f"miner [{hotkey}] incorrectly sent trade pair. Raw signal: {signal}"
            )

        bt.logging.info(f"Parsed trade pair from signal: {trade_pair}")
        signal_order_type = OrderType.from_string(signal["order_type"])
        bt.logging.info(f"Parsed order type from signal: {signal_order_type}")
        signal_leverage = signal["leverage"]
        bt.logging.info(f"Parsed leverage from signal: {signal_leverage}")

        bt.logging.info("Attempting to get closing price for trade pair: " + trade_pair.trade_pair_id)
        live_closing_price = self.live_price_fetcher.get_close(trade_pair=trade_pair)

        order = Order(
            trade_pair=trade_pair,
            order_type=signal_order_type,
            leverage=signal_leverage,
            price=live_closing_price,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=str(uuid.uuid4()),
        )
        bt.logging.success(f"Converted signal to order: {order}")
        return order

    def _enforce_num_open_order_limit(self, trade_pair_to_open_position: dict, signal_to_order):
        # Check if there are too many orders across all open positions.
        # If so, check if the current order is a FLAT order (reduces number of open orders). If not, raise an exception
        n_open_positions = sum([len(position.orders) for position in trade_pair_to_open_position.values()])
        if n_open_positions >= ValiConfig.MAX_OPEN_ORDERS_PER_HOTKEY:
            if signal_to_order.order_type != OrderType.FLAT:
                raise SignalException(
                    f"miner [{signal_to_order}] sent too many open orders [{len(trade_pair_to_open_position)}] and "
                    f"order [{signal_to_order}] is not a FLAT order."
                )

    def _get_or_create_open_position(self, signal_to_order: Order, miner_hotkey: str, trade_pair_to_open_position: dict):
        trade_pair = signal_to_order.trade_pair

        # if a position already exists, add the order to it
        if trade_pair in trade_pair_to_open_position:
            # If the position is closed, raise an exception. This can happen if the miner is eliminated in the main
            # loop thread.
            if trade_pair_to_open_position[trade_pair].is_closed_position:
                raise SignalException(
                    f"miner [{miner_hotkey}] sent signal for "
                    f"closed position [{trade_pair}]")
            bt.logging.debug("adding to existing position")
            open_position = trade_pair_to_open_position[trade_pair]
        else:
            bt.logging.debug("processing new position")
            # if the order is FLAT ignore and log
            if signal_to_order.order_type == OrderType.FLAT:
                raise SignalException(
                    f"miner [{miner_hotkey}] sent a "
                    f"FLAT order with no existing position."
                )
            else:
                # if a position doesn't exist, then make a new one
                open_position = Position(
                    miner_hotkey=miner_hotkey,
                    position_uuid=str(uuid.uuid4()),
                    open_ms=TimeUtil.now_in_millis(),
                    trade_pair=trade_pair
                )
        return open_position

    def should_fail_early(self, synapse: template.protocol.SendSignal | template.protocol.GetPositions, signal=None) -> bool:
        miner_hotkey = synapse.dendrite.hotkey
        # Don't allow miners to send too many signals in a short period of time
        allowed, wait_time = self.rate_limiter.is_allowed(miner_hotkey)
        if not allowed:
            msg = f"Rate limited. Please wait {wait_time} seconds before sending another signal."
            bt.logging.trace(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        # don't process eliminated miners
        with self.eliminations_lock:
            eliminations = self.mdd_checker.get_eliminations_from_disk()
        eliminated_hotkey_to_info ={x['hotkey']: x for x in eliminations} if eliminations else dict()
        if synapse.dendrite.hotkey in eliminated_hotkey_to_info:
            info = eliminated_hotkey_to_info[synapse.dendrite.hotkey]
            msg = f"This miner hotkey {synapse.dendrite.hotkey} has been eliminated for reason {info} and cannot participate in this subnet. Try again after re-registering."
            bt.logging.debug(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if signal:
            tp = self.parse_trade_pair_from_signal(signal)
            if tp and self.live_price_fetcher.is_market_closed_for_trade_pair(tp):
                msg = (f"Market for trade pair [{tp.trade_pair_id}] is likely closed or this validator is"
                       f" having issues fetching live price. Please try again later.")
                bt.logging.error(msg)
                synapse.successfully_processed = False
                synapse.error_message = msg
                return True

            if tp and (tp.is_indices or tp == TradePair.CADCHF):
                msg = (f"Trade pair [{tp.trade_pair_id}] has been temporarily halted. "
                       f"Please try again with a different trade pair.")
                bt.logging.error(msg)
                synapse.successfully_processed = False
                synapse.error_message = msg
                return True

        return False

    def enforce_order_cooldown(self, order, open_position):
        # Don't allow orders to be placed within ORDER_COOLDOWN_MS of the previous order
        if len(open_position.orders) == 0:
            return
        last_order = open_position.orders[-1]
        if order.processed_ms - last_order.processed_ms < ValiConfig.ORDER_COOLDOWN_MS:
            previous_order_time = TimeUtil.millis_to_formatted_date_str(last_order.processed_ms)
            current_time = TimeUtil.millis_to_formatted_date_str(order.processed_ms)
            time_to_wait_in_s = (ValiConfig.ORDER_COOLDOWN_MS - (order.processed_ms - last_order.processed_ms)) / 1000
            raise SignalException(
                f"Order for trade pair [{order.trade_pair.trade_pair_id}] was placed too soon after the previous order. "
                f"Last order was placed at [{previous_order_time}] and current order was placed at [{current_time}]."
                f"Please wait {time_to_wait_in_s} seconds before placing another order."
            )

    # This is the core validator function to receive a signal
    def receive_signal(self, synapse: template.protocol.SendSignal,
                       ) -> template.protocol.SendSignal:
        # pull miner hotkey to reference in various activities
        miner_hotkey = synapse.dendrite.hotkey
        signal = synapse.signal
        bt.logging.info(f"received signal [{signal}] from miner_hotkey [{miner_hotkey}].")
        if self.should_fail_early(synapse, signal=signal):
            return synapse

        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""
        try:
            signal_to_order = self.convert_signal_to_order(signal, miner_hotkey)
            # Multiple threads can run receive_signal at once. Don't allow two threads to trample each other.
            with self.position_manager.position_locks.get_lock(miner_hotkey, signal_to_order.trade_pair.trade_pair_id):
                # gather open positions and see which trade pairs have an open position
                trade_pair_to_open_position = {position.trade_pair: position for position in
                                               self.position_manager.get_all_miner_positions(miner_hotkey,
                                                                                             only_open_positions=True)}
                self._enforce_num_open_order_limit(trade_pair_to_open_position, signal_to_order)
                open_position = self._get_or_create_open_position(signal_to_order, miner_hotkey, trade_pair_to_open_position)
                self.enforce_order_cooldown(signal_to_order, open_position)
                open_position.add_order(signal_to_order)
                self.position_manager.save_miner_position_to_disk(open_position)
                # Log the open position for the miner
                bt.logging.info(f"Position for miner [{miner_hotkey}] updated: {open_position}")
                open_position.log_position_status()
            # self.plagiarism_detector.check_plagiarism(open_position, signal_to_order)

        except SignalException as e:
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())
        except Exception as e:
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False

        synapse.error_message = error_message
        bt.logging.success(f"Sending ack back to miner [{miner_hotkey}]")
        return synapse

    def get_positions(self, synapse: template.protocol.GetPositions,
                      ) -> template.protocol.GetPositions:
        if self.should_fail_early(synapse):
            return synapse

        miner_hotkey = synapse.dendrite.hotkey
        error_message = ""
        try:
            hotkey = synapse.dendrite.hotkey
            positions = self.position_manager.get_all_miner_positions(hotkey, sort_positions=True)
            synapse.positions = [position.to_dict() for position in positions]
            bt.logging.info(f"Sending {len(positions)} positions back to miner: " + hotkey)
        except Exception as e:
            error_message = f"Error in GetPositions for [{miner_hotkey}] with error [{e}]. Perhaps the position was being written to disk at the same time."
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False
        synapse.error_message = error_message
        return synapse


# This is the main function, which runs the miner.
if __name__ == "__main__":
    validator = Validator()
    validator.main()
