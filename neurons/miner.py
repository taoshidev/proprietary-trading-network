# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import os
import argparse
import time
import traceback
from typing import List

import bittensor as bt
from miner_objects.prop_net_order_placer import PropNetOrderPlacer
from miner_objects.position_inspector import PositionInspector
from shared_objects.metagraph_updater import MetagraphUpdater
from template.protocol import GetPositions, Dummy


class Miner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging_directory()
        self.initialize_bittensor_objects()
        self.check_miner_registration()
        self.my_subnet_uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def setup_logging_directory(self):
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

    def initialize_bittensor_objects(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.metagraph_updater = MetagraphUpdater(self.config, self.metagraph)
        self.prop_net_order_placer = PropNetOrderPlacer(
            self.dendrite, self.metagraph, self.config
        )
        self.position_inspector = PositionInspector(
            self.dendrite, self.metagraph, self.config
        )

    def check_miner_registration(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                "Your miner is not registered. Please register and try again."
            )
            exit()

    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        # Adds override arguments for network and netuid.
        parser.add_argument(
            "--netuid", type=int, default=1, help="The chain subnet uid."
        )
        # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.wallet.add_args(parser)
        # Adds an argument to allow setting write_failed_signal_logs from the command line
        # We use a placeholder default value here (None) to check if the user has provided a value later
        parser.add_argument(
            "--write_failed_signal_logs",
            type=bool,
            default=None,
            help="Whether to write logs for failed signals. Default is True unless --subtensor.network is 'test'.",
        )

        # Parse the config (will take command-line arguments if provided)
        # To print help message, run python3 template/miner.py --help
        config = bt.config(parser)

        # Determine the default value for write_failed_signal_logs based on the subtensor.network,
        # but only if the user hasn't explicitly set it via command line.
        if config.write_failed_signal_logs is None:
            config.write_failed_signal_logs = (
                False if config.subtensor.network == "test" else True
            )

        # Step 3: Set up logging directory
        # Logging is crucial for monitoring and debugging purposes.
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

    def run(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info("Starting miner loop.")

        while True:
            try:
                self.metagraph_updater.update_metagraph()
                self.prop_net_order_placer.send_signals()
                self.position_inspector.send_signals_with_cooldown()
            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception:
                bt.logging.error(traceback.format_exc())


def basic_get_config():
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument(
        "--test_only_historical",
        default=0,
        help="if you only want to pull in " "historical data for testing.",
    )
    parser.add_argument(
        "--continuous_data_feed",
        default=0,
        help="this will have the validator ping every 5 mins "
        "for updated predictions",
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config


def basic_main():
    config = basic_get_config()

    # base setup for valis

    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other validators and miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again."
        )
        exit()

    # Each validator gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    while True:
        dummy_proto = Dummy()
        dummy_responses = dendrite.query(metagraph.axons, dummy_proto, deserialize=True)
        print(dummy_responses)
        time.sleep(5)


if __name__ == "__main__":
    # miner = Miner()
    # miner.run()
    basic_main()
