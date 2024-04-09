# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import os
import argparse
import threading
import traceback
import time
import bittensor as bt
from miner_objects.prop_net_order_placer import PropNetOrderPlacer
from miner_objects.position_inspector import PositionInspector
from shared_objects.metagraph_updater import MetagraphUpdater


class Miner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging_directory()
        self.initialize_bittensor_objects()
        self.check_miner_registration()
        self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        # Start the metagraph updater loop in its own thread
        self.updater_thread = threading.Thread(target=self.metagraph_updater.run_update_loop, daemon=True)
        self.updater_thread.start()

    def setup_logging_directory(self):
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

    def initialize_bittensor_objects(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.prop_net_order_placer = PropNetOrderPlacer(self.dendrite, self.metagraph, self.config)
        self.position_inspector = PositionInspector(self.dendrite, self.metagraph, self.config)
        self.metagraph_updater = MetagraphUpdater(self.config, self.metagraph, self.wallet.hotkey.ss58_address,
                                                  True, position_inspector=self.position_inspector)


    def check_miner_registration(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error("Your miner is not registered. Please register and try again.")
            exit()

    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        # Adds override arguments for network and netuid.
        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
        # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.wallet.add_args(parser)
        # Adds an argument to allow setting write_failed_signal_logs from the command line
        # We use a placeholder default value here (None) to check if the user has provided a value later
        parser.add_argument("--write_failed_signal_logs", type=bool, default=None,
                            help="Whether to write logs for failed signals. Default is True unless --subtensor.network is 'test'.")

        # Parse the config (will take command-line arguments if provided)
        # To print help message, run python3 template/miner.py --help
        config = bt.config(parser)

        # Determine the default value for write_failed_signal_logs based on the subtensor.network,
        # but only if the user hasn't explicitly set it via command line.
        if config.write_failed_signal_logs is None:
            config.write_failed_signal_logs = False if config.subtensor.network == "test" else True

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
                self.prop_net_order_placer.send_signals(recently_acked_validators=
                                                        self.position_inspector.get_recently_acked_validators())
                self.position_inspector.log_validator_positions()
            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                bt.logging.success("Miner killed by keyboard interrupt.")
                self.metagraph_updater.stop_update_loop()
                self.updater_thread.join()
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(f"Error in miner loop: {e}")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)


if __name__ == "__main__":
    miner = Miner()
    miner.run()

