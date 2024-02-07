# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc


# Step 1: Import necessary libraries and modules
import os
import time
import uuid
from typing import Tuple

import template
import argparse
import traceback
import bittensor as bt

from data_generator.twelvedata import TwelveData
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.dataclasses.order import Order
from vali_objects.dataclasses.position import Position
from vali_objects.dataclasses.signal import Signal
from vali_objects.enums.order_type_enum import OrderTypeEnum
from vali_objects.utils.exchange_utils import ExchangeUtils
from vali_objects.utils.vali_utils import ValiUtils

base_mining_model = None
base_model_id = None


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
    global secrets

    secrets = ValiUtils.get_secrets()
    if secrets is None:
        raise Exception("unable to get secrets data from "
                        "validation/secrets.json. Please ensure it exists")

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
    metagraph = subtensor.metagraph(config.netuid)
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
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, synapse.dendrite.hotkey
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, synapse.dendrite.hotkey

    def rs_priority_fn(synapse: template.protocol.SendSignal) -> float:
        bt.logging.debug("got to prioritization rs")
        # simply just prioritize based on uid as it's not significant
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.uids[caller_uid])
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', priority)
        return priority

    # This is the core validator function, which decides the miner's response to a valid, high-priority request.
    def receive_signal(synapse: template.protocol.SendSignal) -> template.protocol.SendSignal:

        bt.logging.info(f"received signal request")

        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""

        # convert every signal to orders
        # set the closing price for every order
        signals = [Signal(**signal) for signal in synapse.signals]

        bt.logging.info(f"received [{len(signals)}] signal request(s)")

        # pull miner hotkey to reference in various activities
        miner_hotkey = synapse.dendrite.hotkey

        # gather open positions and see which trade pairs have an open position
        open_positions = ValiUtils.get_all_miner_positions_and_orders(miner_hotkey,
                                                                      only_open_positions=True)
        open_position_trade_pairs = {position.trade_pair: position
                                     for position in open_positions}

        # ensure all signals are for an existing trade pair
        # can determine using the fees object
        try:
            for signal in signals:
                if signal.trade_pair not in ValiConfig.TRADE_PAIR_FEES:
                    error_message = f"miner [{synapse.dendrite.hotkey}] incorrectly " \
                                    f"sent trade pair [{signal.trade_pair}]"
                    bt.logging.error(error_message)
                else:
                    trade_pair = signal.trade_pair
                    # trade pair exists, convert signal to order
                    signal_closing_price = TwelveData(api_key=secrets["twelvedata_apikey"]) \
                        .get_data(symbol=trade_pair)
                    signal_to_order = Order(
                        trade_pair=trade_pair,
                        order_type=signal.order_type,
                        leverage=signal.leverage,
                        price=signal_closing_price,
                        processed_ms=TimeUtil.now_in_millis(),
                        order_uuid=str(uuid.uuid4()))
                    # if a position already exists, add the order to it and
                    # close if the order generates a FLAT
                    if trade_pair in open_position_trade_pairs:
                        bt.logging.debug("adding to existing position")
                        open_position = open_position_trade_pairs[trade_pair]
                        open_position.orders.append(signal_to_order)
                        if ExchangeUtils.is_closed_position(open_position):
                            bt.logging.debug("closing existing position")
                            open_position.close_ms = TimeUtil.now_in_millis()
                            open_position.close_price = signal_closing_price
                            open_position.return_at_close = (ExchangeUtils
                                                             .calculate_position_return(open_position,
                                                                                        signal_closing_price))
                            bt.logging.debug(f"closing existing position details: "
                                             f"close_ms [{open_position.close_ms }] "
                                             f"close_price [{open_position.close_price }] "
                                             f"return_at_close [{open_position.return_at_close}]")
                        ValiUtils.save_miner_position(miner_hotkey,
                                                      open_position.position_uuid,
                                                      open_position)
                    else:
                        bt.logging.debug("processing new position")
                        # if the order is FLAT ignore and log
                        if signal_to_order.order_type != OrderTypeEnum.FLAT:
                            # if a position doesn't exist, then make a new one
                            open_position = Position(miner_hotkey=miner_hotkey,
                                                     position_uuid=str(uuid.uuid4()),
                                                     open_ms=TimeUtil.now_in_millis(),
                                                     close_price=signal_closing_price,
                                                     trade_pair=trade_pair,
                                                     orders=[signal_to_order])
                            ValiUtils.save_miner_position(miner_hotkey,
                                                          open_position.position_uuid,
                                                          open_position)
                        else:
                            bt.logging.info(f"miner [{synapse.dendrite.hotkey}] sent a "
                                            f"FLAT order with no existing position")
        except Exception as e:
            error_message = e
            bt.logging.error(traceback.format_exc())

        synapse.received = 1
        synapse.error_message = error_message

        print(f"printing synapse [{synapse}]")

        return synapse

    # Step 5: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    bt.logging.info(f"setting port [{config.axon.port}]")
    bt.logging.info(f"setting external port [{config.axon.external_port}]")
    axon = bt.axon(wallet=wallet, port=config.axon.port, external_port=config.axon.external_port)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn=receive_signal,
        blacklist_fn=rs_blacklist_fn,
        priority_fn=rs_priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving attached axons on network:"
                    f" {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 6: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # TODO(developer): Define any additional operations to be performed by the miner.
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success('Miner killed by keyboard interrupt.')
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config())
