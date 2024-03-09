# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import multiprocessing

# Step 1: Import necessary libraries and modules
import os
import threading
import time
import uuid
from typing import Tuple

import template
import argparse
import traceback
import bittensor as bt

from data_generator.twelvedata_service import TwelveDataService
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.MetagraphUpdater import MetagraphUpdater
from vali_objects.utils.EliminationManager import EliminationManager
from vali_objects.utils.SubtensorWeightSetter import SubtensorWeightSetter
from vali_objects.utils.MDDChecker import MDDChecker
from vali_objects.utils.PlagiarismDetector import PlagiarismDetector
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.position import Position
from vali_objects.vali_dataclasses.signal import Signal
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_utils import ValiUtils


def get_config():
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
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


# Main takes the config and starts the miner.
def main(config):
    secrets = ValiUtils.get_secrets()

    if secrets is None:
        raise Exception(
            "unable to get secrets data from "
            "validation/miner_secrets.json. Please ensure it exists"
        )

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} "
        f"on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
    metagraph = subtensor.metagraph(config.netuid)
    metagraphUpdater = MetagraphUpdater(config, metagraph)
    metagraphUpdater.update_metagraph()
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} is not registered to chain "
            f"connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
        bt.logging.debug("got to blacklisting rs")

        # Ignore requests from unrecognized entities.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )

        # don't process eliminated miners
        if synapse.dendrite.hotkey in eliminations:
            # Ignore requests from eliminated hotkeys
            bt.logging.trace(f"Eliminated hotkey {synapse.dendrite.hotkey}")
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    def rs_priority_fn(synapse: template.protocol.SendSignal) -> float:
        bt.logging.debug("got to prioritization rs")
        # simply just prioritize based on uid as it's not significant
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.uids[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def convert_signal_to_order(signal, hotkey) -> Order:
        """
        Example input signal
          {'trade_pair': {'trade_pair_id': 'BTCUSD', 'trade_pair': 'BTC/USD', 'fees': 0.003, 'min_leverage': 0.0001, 'max_leverage': 20},
          'order_type': 'LONG',
          'leverage': 0.5}
        """
        signal_trade_pair = TradePair.get_trade_pair(signal["trade_pair"]["trade_pair_id"])
        if signal_trade_pair not in ValiConfig.TRADE_PAIR_FEES:
            raise SignalException(
                f"miner [{hotkey}] incorrectly "
                f"sent trade pair [{signal_trade_pair}]"
            )
    
        bt.logging.success(f"Parsed trade pair from signal: {signal_trade_pair}")
        signal_order_type = OrderType.get_order_type(signal["order_type"])
        bt.logging.success(f"Parsed order type from signal: {signal_order_type}")
        signal_leverage = signal["leverage"]
        bt.logging.success(f"Parsed leverage from signal: {signal_leverage}")

        tds = TwelveDataService(api_key=secrets["twelvedata_apikey"])
        bt.logging.info("Attempting to get closing price for trade pair: " + signal_trade_pair.trade_pair_id)
        signal_closing_price = tds.get_close(trade_pair=signal_trade_pair)[signal_trade_pair]
        
        order = Order(
            trade_pair=signal_trade_pair,
            order_type=signal_order_type,
            leverage=signal_leverage,
            price=signal_closing_price,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=str(uuid.uuid4()),
        )
        bt.logging.success(f"Converted signal to order: {order}")
        return order
    
    plagiarismDetector = PlagiarismDetector(config, metagraph)
    # This is the core validator function to receive a signal
    def receive_signal(
        synapse: template.protocol.SendSignal,
    ) -> template.protocol.SendSignal:
        # pull miner hotkey to reference in various activities
        miner_hotkey = synapse.dendrite.hotkey
        signal = synapse.signal
        bt.logging.info(f"received signal [{signal}] from miner_hotkey [{miner_hotkey}]. Signal is of type: {type(signal)}")

        eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )

        bt.logging.info(f"Current eliminations: {eliminations}")

        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""

        # gather open positions and see which trade pairs have an open position
        open_position_trade_pairs = {
            position.trade_pair: position
            for position in PositionUtils.get_all_miner_positions(
                miner_hotkey, only_open_positions=True
            )
        }

        # ensure all signals are for an existing trade pair
        # can determine using the fees object
        try:
            signal_to_order = convert_signal_to_order(signal, synapse.dendrite.hotkey)
            trade_pair = signal_to_order.trade_pair
            # if a position already exists, add the order to it 
            if trade_pair in open_position_trade_pairs:
                # If the position is closed, raise an exception
                if open_position_trade_pairs[trade_pair].is_closed_position():
                    raise SignalException(
                        f"miner [{synapse.dendrite.hotkey}] sent signal for "
                        f"closed position [{trade_pair}]")
                bt.logging.debug("adding to existing position")
                open_position = open_position_trade_pairs[trade_pair]
                open_position.add_order(signal_to_order)
                ValiUtils.save_miner_position(
                    miner_hotkey, open_position.position_uuid, open_position
                )
            else:
                bt.logging.debug("processing new position")
                # if the order is FLAT ignore and log
                if signal_to_order.order_type == OrderType.FLAT:
                    raise SignalException(
                        f"miner [{synapse.dendrite.hotkey}] sent a "
                        f"FLAT order with no existing position."
                    )
                else:
                    # if an position doesn't exist, then make a new one
                    open_position = Position(
                        miner_hotkey=miner_hotkey,
                        position_uuid=str(uuid.uuid4()),
                        open_ms=TimeUtil.now_in_millis(),
                        open_price=signal_to_order.price,
                        trade_pair=trade_pair,
                        orders=[signal_to_order],
                    )
                    ValiUtils.save_miner_position(
                        miner_hotkey, open_position.position_uuid, open_position
                    )

            open_position.log_position_status()
            plagiarismDetector.check_plagiarism(open_position, signal_to_order, miner_hotkey)

        except SignalException as e:
            error_message = f"error processing signal [{e}]"
            bt.logging.warning(error_message)
        except Exception as e:
            error_message = e
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = 1
        synapse.error_message = error_message

        return synapse

    def gp_blacklist_fn(synapse: template.protocol.GetPositions) -> Tuple[bool, str]:
        bt.logging.debug("got to blacklisting gp")

        # Ignore requests from unrecognized entities.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )

        # don't process eliminated miners
        if synapse.dendrite.hotkey in eliminations:
            # Ignore requests from eliminated hotkeys
            bt.logging.trace(f"Eliminated hotkey {synapse.dendrite.hotkey}")
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    def gp_priority_fn(synapse: template.protocol.GetPositions) -> float:
        bt.logging.debug("got to prioritization gp")
        # simply just prioritize based on uid as it's not significant
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.uids[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def get_positions(
        synapse: template.protocol.GetPositions,
    ) -> template.protocol.GetPositions:
        # use position_util.get_all_miner_positions and return the synapse over the network
        # synapse.positions = position_util.get_all_miner_positions()
        # return synapse
        pass

    # Step 5: Build and link vali functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    bt.logging.info(f"setting port [{config.axon.port}]")
    bt.logging.info(f"setting external port [{config.axon.external_port}]")
    axon = bt.axon(
        wallet=wallet, port=config.axon.port, external_port=config.axon.external_port
    )
    bt.logging.info(f"Axon {axon}")

    # Attach determines which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn=receive_signal,
        blacklist_fn=rs_blacklist_fn,
        priority_fn=rs_priority_fn,
    )
    axon.attach(
        forward_fn=get_positions,
        blacklist_fn=gp_blacklist_fn,
        priority_fn=gp_priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving attached axons on network:"
        f" {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # see if files exist and if not set them to empty
    ValiUtils.init_cache_files(metagraph)

    # Starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    mddChecker = MDDChecker(config, metagraph)
    weightSetter = SubtensorWeightSetter(config, wallet, metagraph)
    eliminationManager = EliminationManager()
    

    # Step 6: Keep the vali alive
    # This loop maintains the vali's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    while True:
        try:
            metagraphUpdater.update_metagraph()
            mddChecker.mdd_check()
            weightSetter.set_weights()
            eliminationManager.process_eliminations()

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception:
            bt.logging.error(traceback.format_exc())


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config())
