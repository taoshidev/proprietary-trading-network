# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import json
import os
import argparse
import threading
import traceback
import time
import bittensor as bt
import subprocess

from miner_config import MinerConfig
from miner_objects.dashboard import Dashboard
from miner_objects.prop_net_order_placer import PropNetOrderPlacer
from miner_objects.position_inspector import PositionInspector
from shared_objects.metagraph_updater import MetagraphUpdater
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


class Miner:
    def __init__(self):
        self.config = self.get_config()
        self.is_testnet = self.config.subtensor.network == "test"
        self.setup_logging_directory()
        self.initialize_bittensor_objects()
        self.check_miner_registration()
        self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        # Start the metagraph updater loop in its own thread
        self.metagraph_updater_thread = threading.Thread(target=self.metagraph_updater.run_update_loop, daemon=True)
        self.metagraph_updater_thread.start()
        # Start position inspector loop in its own thread
        self.position_inspector_thread = threading.Thread(target=self.position_inspector.run_update_loop, daemon=True)
        self.position_inspector_thread.start()

        # Dashboard
        # Start the miner data api in its own thread
        try:
            self.dashboard = Dashboard(self.wallet, self.metagraph, self.config, self.is_testnet)
            self.dashboard_api_thread = threading.Thread(target=self.dashboard.run, daemon=True)
            self.dashboard_api_thread.start()
        except OSError as e:
            bt.logging.info(f"Unable to start miner dashboard with error {e}. Restart miner and specify a new port if desired.")
        # Initialize the dashboard process variable for the frontend
        self.dashboard_frontend_process = None

    def setup_logging_directory(self):
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

    def initialize_bittensor_objects(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.position_inspector = PositionInspector(self.wallet, self.metagraph, self.config)
        self.prop_net_order_placer = PropNetOrderPlacer(self.wallet, self.metagraph, self.config, self.is_testnet, position_inspector=self.position_inspector)
        self.metagraph_updater = MetagraphUpdater(self.config, self.metagraph, self.wallet.hotkey.ss58_address,
                                                  True, position_inspector=self.position_inspector)


    def check_miner_registration(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error("Your miner is not registered. Please register and try again.")
            exit()

    def load_signal_data(self, signal_file_path: str):
        """Loads the signal data from a file."""
        try:
            data = ValiBkpUtils.get_file(signal_file_path)
            return json.loads(data, cls=GeneralizedJSONDecoder)
        except json.JSONDecodeError as e:
            bt.logging.error(f"Failed to decode JSON from {signal_file_path}: {e}")
            return None  # Or handle the error as needed

    def get_all_files_in_dir_no_duplicate_trade_pairs(self):
        # If there are duplicate trade pairs, only the most recent signal for that trade pair will be sent this round.
        all_files = ValiBkpUtils.get_all_files_in_dir(MinerConfig.get_miner_received_signals_dir())
        signals_dict = {}
        files_to_delete = []
        for f_name in all_files:
            try:
                bt.logging.info(f"Reading signal file {f_name}")
                signal = self.load_signal_data(f_name)
                trade_pair_id = signal['trade_pair']['trade_pair_id']
                time_of_signal_file = os.path.getmtime(f_name)
                if trade_pair_id not in signals_dict or signals_dict[trade_pair_id][2] < time_of_signal_file:
                    signals_dict[trade_pair_id] = (signal, f_name, time_of_signal_file)
                    files_to_delete.append(f_name)
            except json.JSONDecodeError as e:
                bt.logging.error(f"Error decoding JSON from file {f_name}: {e}")

        # Delete files to prevent duplicate reading and conflicts
        for f_name in files_to_delete:
            bt.logging.info(f"Deleting signal file {f_name}")
            os.remove(f_name)

        # Return all signals as a list
        return [x[0] for x in signals_dict.values()], [x[1] for x in signals_dict.values()]

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
        parser.add_argument(
            '--start-dashboard',
            action='store_true',
            help='Start the miner-dashboard along with the miner.'
        )

        # Parse the config (will take command-line arguments if provided)
        # To print help message, run python3 template/miner.py --help
        config = bt.config(parser)
        bt.logging.enable_default()
        if config.logging.debug:
            bt.logging.enable_debug()
        if config.logging.trace:
            bt.logging.enable_trace()

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

    def start_dashboard_frontend(self):
        """
        starts the miner dashboard. Allows the use of npm, yarn, or pnpm
        """
        try:
            dashboard_dir = "miner_objects/miner_dashboard"
            # Determine which package manager is available
            package_manager = None
            for pm in ['pnpm', 'yarn', 'npm']:
                if subprocess.run(['which', pm], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                    package_manager = pm
                    break

            if not package_manager:
                bt.logging.error("No package manager found. Please install npm, yarn, or pnpm.")
                return

            # Run 'install' command for the identified package manager
            subprocess.run([package_manager, "install"], cwd=dashboard_dir, check=True)
            bt.logging.info(f"Install completed using {package_manager}.")

            # Start the dashboard process
            if package_manager == 'npm':
                self.dashboard_frontend_process = subprocess.Popen(['npm', 'run', 'dev'], cwd=dashboard_dir)  # Popen runs in the background
            else:
                self.dashboard_frontend_process = subprocess.Popen([package_manager, 'dev'], cwd=dashboard_dir)
            bt.logging.info("Dashboard started.")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Command '{e.cmd}' failed with return code {e.returncode}.")
        except Exception as e:
            bt.logging.error(f"Failed to start dashboard: {e}")

    def run(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info("Starting miner loop.")

        # Start the dashboard if the flag is set
        if self.config.start_dashboard:
            bt.logging.info("Starting miner dashboard.")
            self.start_dashboard_frontend()

        bt.logging.info("Waiting for signals...")
        while True:
            try:
                signals, signal_file_names = self.get_all_files_in_dir_no_duplicate_trade_pairs()
                self.prop_net_order_placer.send_signals(signals, signal_file_names, recently_acked_validators=
                                                    self.position_inspector.get_recently_acked_validators())
                time.sleep(1)
            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                bt.logging.success("Miner killed by keyboard interrupt.")
                if self.dashboard_frontend_process:
                    self.dashboard_frontend_process.terminate()  # Terminate the dashboard if it was started
                    self.dashboard_frontend_process.wait()
                    bt.logging.info("Dashboard terminated.")
                self.metagraph_updater_thread.join()
                self.position_inspector.stop_update_loop()
                self.position_inspector_thread.join()
                # dashboard api server
                if self.dashboard_api_thread is not None and self.dashboard_api_thread.is_alive():
                    self.dashboard_api_thread.join()
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception:
                bt.logging.error(traceback.format_exc())
                time.sleep(10)


if __name__ == "__main__":
    miner = Miner()
    miner.run()

