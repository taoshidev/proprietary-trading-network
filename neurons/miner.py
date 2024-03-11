# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import os
import argparse

import bittensor as bt
from typing import List
from miner_objects.PropNetOrderPlacer import PropNetOrderPlacer
from shared_objects.MetagraphUpdater import MetagraphUpdater

# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
from template.protocol import SendSignal, GetPositions


def get_config():
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
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config



def get_positions(_dendrite, _config, _metagraph, validators):
    get_position_proto = GetPositions()

    # order validators by largest stake holding, request from 1 of the top 3
    # get all uids, order by largest stake
    # stake = metagraph.axons[uid].stake.tao

    # if they're not in the same state, then choose from the vali that has the most
    # orders filled for the miner

    vali_responses = _dendrite.query(
        _metagraph.axons[0], get_position_proto, deserialize=True
    )


def order_valis_by_stake(metagraph) -> List[int]:
    # stake = metagraph.axons[uid].stake.tao
    pass



# The main function parses the configuration and runs the miner.
if __name__ == "__main__":
    config = get_config()

    # base setup for valis

    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor miner objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the miner.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt.logging.info(f"Dendrite: {dendrite}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
    metagraph = subtensor.metagraph(config.netuid)
    metagraphUpdater = MetagraphUpdater(config, metagraph)
    metagraphUpdater.update_metagraph()
    bt.logging.info(f"Metagraph: {metagraph}")

    propNetOrderPlacer = PropNetOrderPlacer(dendrite, metagraph, config)

    # Step 5: Connect the miner to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again."
        )
        exit()

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting miner loop.")

    while True:
        metagraphUpdater.update_metagraph()
        propNetOrderPlacer.send_signals()
