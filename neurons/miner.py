# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import os
import argparse
from typing import List

import bittensor as bt
from miner_objects.PropNetOrderPlacer import PropNetOrderPlacer
from shared_objects.MetagraphUpdater import MetagraphUpdater
from template.protocol import GetPositions


class Miner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging_directory()
        self.initialize_bittensor_objects()
        self.check_miner_registration()
        self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

    def setup_logging_directory(self):
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

    def initialize_bittensor_objects(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.metagraphUpdater = MetagraphUpdater(self.config, self.metagraph)
        self.propNetOrderPlacer = PropNetOrderPlacer(self.dendrite, self.metagraph, self.config)

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
                "miner",
            )
        )
        return config

    def get_positions(self, _dendrite, _config, _metagraph, validators):
        get_position_proto = GetPositions()

        # order validators by largest stake holding, request from 1 of the top 3
        # get all uids, order by largest stake
        # stake = metagraph.axons[uid].stake.tao

        # if they're not in the same state, then choose from the vali that has the most
        # orders filled for the miner

        vali_responses = _dendrite.query(
            _metagraph.axons[0], get_position_proto, deserialize=True
        )

    def order_valis_by_stake(self, metagraph) -> List[int]:
        # stake = metagraph.axons[uid].stake.tao
        pass

    def run(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info("Starting miner loop.")

        while True:
            self.metagraphUpdater.update_metagraph()
            self.propNetOrderPlacer.send_signals()


if __name__ == "__main__":
    miner = Miner()
    miner.run()

