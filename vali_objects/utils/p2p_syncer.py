import asyncio
import base64
import gzip
import json
# import random
import traceback
from collections import defaultdict

import bittensor as bt
# from tabulate import tabulate

import template
from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order


class P2PSyncer:
    def __init__(self, wallet=None, metagraph=None, is_testnet=None):
        self.wallet = wallet
        self.metagraph = metagraph
        self.last_signal_sync_time_ms = 0
        self.num_checkpoints_received = 0
        self.golden = {}
        if self.wallet is not None:
            self.hotkey = self.wallet.hotkey.ss58_address

        # flag for testnet
        self.is_testnet = is_testnet

    async def send_checkpoint_poke(self):
        """
        serializes checkpoint json and transmits to all validators via synapse
        """
        # only validators should request a checkpoint with a poke
        # TODO: remove testnet flag
        if not self.is_testnet and self.hotkey not in [axon.hotkey for axon in self.get_validators()]:
            bt.logging.info("Aborting send_checkpoint_poke; not a qualified validator")
            return

        # get axons to send checkpoints to
        dendrite = bt.dendrite(wallet=self.wallet)
        validator_axons = self.get_trusted_validators(ValiConfig.TOP_N)

        try:
            # create dendrite and transmit synapse
            checkpoint_synapse = template.protocol.ValidatorCheckpoint()
            validator_responses = await dendrite.forward(axons=validator_axons, synapse=checkpoint_synapse, timeout=60) # TODO: make sure this is blocking call, give it a timeout

            bt.logging.info(f"Sending checkpoint from validator {self.wallet.hotkey.ss58_address}")

            failures = 0
            successful_checkpoints = 0
            received_hotkeys_checkpoints = {}

            for response in validator_responses:
                if response.successfully_processed:
                    # Decode from base64 and decompress back into json
                    decoded = base64.b64decode(response.checkpoint)
                    decompressed = gzip.decompress(decoded).decode('utf-8')
                    recv_checkpoint = json.loads(decompressed)

                    received_hotkeys_checkpoints[response.validator_receive_hotkey] = recv_checkpoint

                    bt.logging.info(f"Successfully processed checkpoint from {response.validator_receive_hotkey}")
                    successful_checkpoints += 1
                else:
                    failures += 1
                    bt.logging.info(f"Checkpoint poke to {response.axon.hotkey} failed")

            bt.logging.info(f"{successful_checkpoints} responses succeeded")
            bt.logging.info(f"{failures} responses failed")

            if successful_checkpoints >= ValiConfig.MIN_CHECKPOINTS_RECEIVED:
                bt.logging.info("Received enough checkpoints, now creating golden.")
                self.create_golden(received_hotkeys_checkpoints)
            else:
                bt.logging.info("Not enough checkpoints received to create a golden.")

        except Exception as e:
            bt.logging.info(f"Error sending checkpoint with error [{e}]")

    def get_validators(self, neurons=None):
        """
        get a list of all validators. defined as:
        stake > 1000 and validator_trust > 0.5
        """
        # TODO: remove testnet flag
        if self.is_testnet:
            return self.metagraph.axons
            # return [a for a in self.metagraph.axons if a.ip != ValiConfig.AXON_NO_IP]
        if neurons is None:
            neurons = self.metagraph.neurons
        validator_axons = [n.axon_info for n in neurons
                           if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                           and n.axon_info.ip != ValiConfig.AXON_NO_IP]
        return validator_axons

    def get_trusted_validators(self, top_n_validators, neurons=None):
        """
        get a list of the trusted validators for checkpoint sending
        return top 10 neurons sorted by stake
        """
        if self.is_testnet:
            return self.get_validators()
        if neurons is None:
            neurons = self.metagraph.neurons
        sorted_stake_neurons = sorted(neurons, key=lambda n: n.stake, reverse=True)

        return self.get_validators(sorted_stake_neurons)[:top_n_validators]

    # # TODO: remove temp test method to print out the metagraph state
    # def print_metagraph_attributes(self):
    #     # for n in self.metagraph.neurons:
    #     #     n.axon_info.ip = "1.1.1.1"
    #     #     n.validator_trust = round(random.random(), 3)
    #     #     n.stake = random.randrange(1500)
    #
    #     table = [[n.axon_info.hotkey[:4] for n in self.metagraph.neurons],
    #              [n.axon_info.ip for n in self.metagraph.neurons],
    #              [str(n.stake)[:7] for n in self.metagraph.neurons],
    #              [n.validator_trust for n in self.metagraph.neurons]]
    #     # my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
    #     # bt.logging.info(f"my uid {my_uid}")
    #     smalltable = [r[:20] for r in table]
    #     print(tabulate(smalltable, tablefmt="simple_grid"))
    #     smalltable = [r[20:40] for r in table]
    #     print(tabulate(smalltable, tablefmt="simple_grid"))
    #     smalltable = [r[40:60] for r in table]
    #     print(tabulate(smalltable, tablefmt="simple_grid"))
    #     smalltable = [r[60:80] for r in table]
    #     print(tabulate(smalltable, tablefmt="simple_grid"))
    #     smalltable = [r[80:100] for r in table]
    #     print(tabulate(smalltable, tablefmt="simple_grid"))
    #     # bt.logging.info(f"stake {[n.stake for n in self.metagraph.neurons]}")
    #
    #     # print(self.num_trusted_validators, " trusted validators___________")
    #     for a in self.get_trusted_validators():
    #         print(a.hotkey)

    def sync_positions_with_cooldown(self, auto_sync_enabled:bool):
        # Check if the time is right to sync signals
        if not self.is_testnet:
            if not auto_sync_enabled:
                return
            now_ms = TimeUtil.now_in_millis()
            # Already performed a sync recently
            if now_ms - self.last_signal_sync_time_ms < 1000 * 60 * 30:
                return

            # Check if we are between 7:09 AM and 7:19 AM UTC
            datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
            if not (datetime_now.hour == 7 and (8 < datetime_now.minute < 20)):
                return

        try:
            bt.logging.info("calling send_checkpoint_poke")
            asyncio.run(self.send_checkpoint_poke())
        except Exception as e:
            bt.logging.error(f"Error sending checkpoint: {e}")
            bt.logging.error(traceback.format_exc())

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()

    def create_golden(self, trusted_checkpoints: dict) -> dict:
        """
        Simple majority approach (preferred to start)
            If a position’s uuid exists on the majority of validators, that position is kept.
            If an order uuid exists in the majority of positions, that order is kept.
                Choose the order with the median price.

        return a single checkpoint dict
        """
        position_counts = defaultdict(int)                      # {position_uuid: count}
        position_data = defaultdict(list)                       # {position_uuid: [{position}]}
        position_orders = defaultdict(set)                      # {position_uuid: {order_uuid}}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}

        # simple majority of positions/number of checkpoints
        positions_threshold = len(trusted_checkpoints) / 2

        # parse each checkpoint to count occurrences of each position and order
        for checkpoint in trusted_checkpoints.values():
            # get positions for each miner
            for miner_positions in checkpoint["positions"].values():
                for position in miner_positions["positions"]:
                    position_uuid = position["position_uuid"]
                    position_counts[position_uuid] += 1
                    position_data[position_uuid].append(dict(position, orders=[]))

                    # count and save orders
                    for order in position["orders"]:
                        order_uuid = order["order_uuid"]
                        order_counts[position_uuid][order_uuid] += 1
                        order_data[order_uuid].append(dict(order))

                        position_orders[position_uuid].add(order_uuid)

        # get the set of position_uuids that appear in the majority of checkpoints
        majority_positions = {position_uuid for position_uuid, count in position_counts.items()
                              if count > positions_threshold}

        golden = defaultdict(lambda: defaultdict(list))

        for checkpoint in trusted_checkpoints.values():
            for miner_hotkey, miner_positions in checkpoint["positions"].items():
                for position in miner_positions["positions"]:
                    position_uuid = position["position_uuid"]
                    # position exists on majority of validators
                    if position_uuid in majority_positions:
                        # create a single combined position, and delete uuid to avoid duplicates
                        new_position = Position(miner_hotkey=miner_hotkey,
                                                position_uuid=position_uuid,
                                                open_ms=position["open_ms"],
                                                trade_pair=position["trade_pair"],
                                                orders=[])

                        majority_positions.remove(position_uuid)

                        # simple majority of orders out of the positions they could appear in
                        orders_threshold = position_counts[position_uuid] / 2

                        # get the set of order_uuids that appear in the majority of positions for a position_uuid
                        majority_orders = {order_uuid for order_uuid, count in order_counts[position_uuid].items()
                                           if count > orders_threshold}

                        # order exists in the majority of positions
                        for order_uuid in position_orders[position_uuid]:
                            if order_uuid in majority_orders:
                                trade_pair = TradePair.to_enum(position["trade_pair"][0])
                                combined_order = self.get_median_order(order_data[order_uuid], trade_pair)
                                majority_orders.remove(order_uuid)

                                # TODO: sort orders by "processed_ms" time
                                new_position.add_order(combined_order)
                        new_position.rebuild_position_with_updated_orders()
                        position_dict = json.loads(new_position.to_json_string())
                        golden[miner_hotkey]["positions"].append(position_dict)

        # Convert defaultdict to regular dict
        self.golden = {miner: dict(golden[miner]) for miner in golden}
        return self.golden

    def get_median_order(self, orders, trade_pair) -> Order:
        """
        select the order with the median price from a list of orders with the same order_uuid
        """
        sorted_orders = sorted(orders, key=lambda o: o["price"])
        median_order = sorted_orders[len(orders)//2]
        order = Order(trade_pair=trade_pair,
                      order_type=median_order["order_type"],
                      leverage=median_order["leverage"],
                      price=median_order["price"],
                      processed_ms=median_order["processed_ms"],
                      order_uuid=median_order["order_uuid"])
        return order

if __name__ == "__main__":
    bt.logging.enable_default()
    position_syncer = P2PSyncer()
    asyncio.run(position_syncer.send_checkpoint_poke())