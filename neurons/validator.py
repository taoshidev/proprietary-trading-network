# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import os
import uuid
from typing import Tuple

import template
import argparse
import traceback
import bittensor as bt

from data_generator.twelvedata_service import TwelveDataService
from shared_objects.RateLimiter import RateLimiter
from shared_objects.challenge_utils import ChallengeBase
from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.MetagraphUpdater import MetagraphUpdater
from vali_objects.utils.EliminationManager import EliminationManager
from vali_objects.utils.SubtensorWeightSetter import SubtensorWeightSetter
from vali_objects.utils.MDDChecker import MDDChecker
from vali_objects.utils.PlagiarismDetector import PlagiarismDetector
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.position import Position
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_utils import ValiUtils




class Validator:
    def __init__(self):
        self.config = self.get_config()
        # Ensure the directory for logging exists, else create one.
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

        self.secrets = ValiUtils.get_secrets()
        if self.secrets is None:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )

        self.tds = TwelveDataService(api_key=self.secrets["twelvedata_apikey"])
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
        self.metagraphUpdater = MetagraphUpdater(self.config, self.metagraph)
        self.metagraphUpdater.update_metagraph()
        bt.logging.info(f"Metagraph: {self.metagraph}")

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

        self.rateLimiter = RateLimiter()

        def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph, self.rateLimiter)

        def rs_priority_fn(synapse: template.protocol.SendSignal) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def gp_blacklist_fn(synapse: template.protocol.GetPositions) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph, self.rateLimiter)

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
        ValiUtils.init_cache_files(self.metagraph)

        if wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {wallet} is not registered to chain "
                f"connection: {subtensor} \nRun btcli register and try again. "
            )
            exit()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

        self.plagiarismDetector = PlagiarismDetector(self.config, self.metagraph)
        self.mddChecker = MDDChecker(self.config, self.metagraph)
        self.weightSetter = SubtensorWeightSetter(self.config, wallet, self.metagraph)
        self.eliminationManager = EliminationManager(self.metagraph)
        self.positionLocks = PositionLocks()

    @staticmethod
    def blacklist_fn(synapse, metagraph, rateLimiter) -> Tuple[bool, str]:
        bt.logging.debug("got to blacklisting rs")
        miner_hotkey = synapse.dendrite.hotkey
        allowed, wait_time = rateLimiter.is_allowed(miner_hotkey)
        if not allowed:
            bt.logging.trace(f"Blacklisting {miner_hotkey} for {wait_time} seconds.")
            return True, synapse.dendrite.hotkey

        # Ignore requests from unrecognized entities.
        if miner_hotkey not in metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        eliminations = ChallengeBase.get_filtered_eliminations_from_disk(metagraph.hotkeys)
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()

        # don't process eliminated miners
        if synapse.dendrite.hotkey in eliminated_hotkeys:
            # Ignore requests from eliminated hotkeys
            msg = f"This miner hotkey {synapse.dendrite.hotkey} has been deregistered and cannot participate in this subnet. Try again after re-registering."
            bt.logging.trace(msg)
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    @staticmethod
    def priority_fn(synapse, metagraph) -> float:
        bt.logging.debug("got to prioritization rs")
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
                self.metagraphUpdater.update_metagraph()
                self.mddChecker.mdd_check()
                self.weightSetter.set_weights()
                self.eliminationManager.process_eliminations()
                self.positionLocks.cleanup_locks(self.metagraph.hotkeys)

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Validator killed by keyboard interrupt.")
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception:
                bt.logging.error(traceback.format_exc())

    def convert_signal_to_order(self, signal, hotkey) -> Order:
        """
        Example input signal
          {'trade_pair': {'trade_pair_id': 'BTCUSD', 'trade_pair': 'BTC/USD', 'fees': 0.003, 'min_leverage': 0.0001, 'max_leverage': 20},
          'order_type': 'LONG',
          'leverage': 0.5}
        """
        string_trade_pair = signal["trade_pair"]["trade_pair_id"]
        trade_pair = TradePair.get_trade_pair(string_trade_pair)
        if trade_pair is None:
            bt.logging.error(f"[{trade_pair}] not in TradePair enum.")
            raise SignalException(
                f"miner [{hotkey}] incorrectly "
                f"sent trade pair [{trade_pair}]"
            )

        bt.logging.info(f"Parsed trade pair from signal: {trade_pair}")
        signal_order_type = OrderType.get_order_type(signal["order_type"])
        bt.logging.info(f"Parsed order type from signal: {signal_order_type}")
        signal_leverage = signal["leverage"]
        bt.logging.info(f"Parsed leverage from signal: {signal_leverage}")

        bt.logging.info("Attempting to get closing price for trade pair: " + trade_pair.trade_pair_id)
        live_closing_price = self.tds.get_close(trade_pair=trade_pair)[trade_pair]

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

    def get_relevant_position(self, signal_to_order, open_position_trade_pairs, miner_hotkey):
        trade_pair = signal_to_order.trade_pair
        # if a position already exists, add the order to it
        if trade_pair in open_position_trade_pairs:
            # If the position is closed, raise an exception
            if open_position_trade_pairs[trade_pair].is_closed_position:
                raise SignalException(
                    f"miner [{miner_hotkey}] sent signal for "
                    f"closed position [{trade_pair}]")
            bt.logging.debug("adding to existing position")
            open_position = open_position_trade_pairs[trade_pair]
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

    # This is the core validator function to receive a signal
    def receive_signal(self, synapse: template.protocol.SendSignal,
    ) -> template.protocol.SendSignal:
        # pull miner hotkey to reference in various activities
        miner_hotkey = synapse.dendrite.hotkey
        signal = synapse.signal
        bt.logging.info(f"received signal [{signal}] from miner_hotkey [{miner_hotkey}].")
        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""
        try:
            signal_to_order = self.convert_signal_to_order(signal, miner_hotkey)
            with self.positionLocks.get_lock(miner_hotkey, signal_to_order.trade_pair.trade_pair_id):
                # gather open positions and see which trade pairs have an open position
                open_position_trade_pairs = {position.trade_pair: position for position in PositionUtils.get_all_miner_positions(miner_hotkey, only_open_positions=True)}
                open_position = self.get_relevant_position(signal_to_order, open_position_trade_pairs, miner_hotkey)
                open_position.add_order(signal_to_order)
                ValiUtils.save_miner_position_to_disk(open_position)
                # Log the open position for the miner
                bt.logging.info(f"Position for miner [{miner_hotkey}] updated: {open_position}")
                open_position.log_position_status()
            self.plagiarismDetector.check_plagiarism(open_position, signal_to_order)

        except SignalException as e:
            error_message = f"error processing signal [{e}]"
            bt.logging.error(error_message)
        except Exception as e:
            error_message = e
            bt.logging.error(f"Error processing signal for [{miner_hotkey}] with error [{e}]")
            bt.logging.error(traceback.format_exc())

        synapse.successfully_processed = bool(error_message == "")
        synapse.error_message = error_message
        bt.logging.success(f"Sending back signal to miner [{miner_hotkey}] signal{synapse}")
        return synapse

    def get_positions(self, synapse: template.protocol.GetPositions,
    ) -> template.protocol.GetPositions:
        # use position_util.get_all_miner_positions and return the synapse over the network
        # synapse.positions = position_util.get_all_miner_positions()
        # return synapse
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    validator = Validator()
    validator.main()
