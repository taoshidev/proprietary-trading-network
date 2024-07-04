import asyncio
import base64
import gzip
import json
import math
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
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.validator_sync_base import ValidatorSyncBase

class P2PSyncer(ValidatorSyncBase):
    def __init__(self, wallet=None, metagraph=None, is_testnet=None, shutdown_dict=None, signal_sync_lock=None, signal_sync_condition=None, n_orders_being_processed=None):
        super().__init__(shutdown_dict, signal_sync_lock, signal_sync_condition, n_orders_being_processed)
        self.wallet = wallet
        self.metagraph = metagraph
        self.golden = None
        if self.wallet is not None:
            self.hotkey = self.wallet.hotkey.ss58_address
        # flag for testnet
        self.is_testnet = is_testnet
        self.created_golden = False

    async def send_checkpoint_requests(self):
        """
        serializes checkpoint json and transmits to all validators via synapse
        """
        # only validators should request a checkpoint with a poke
        if not self.is_testnet and self.hotkey not in [axon.hotkey for axon in self.get_validators()]:
            bt.logging.info("Aborting send_checkpoint_poke; not a qualified validator")
            return

        # get axons to send checkpoints to
        dendrite = bt.dendrite(wallet=self.wallet)
        validator_axons = self.get_largest_staked_validators(ValiConfig.TOP_N_STAKE)

        try:
            # create dendrite and transmit synapse
            checkpoint_synapse = template.protocol.ValidatorCheckpoint()
            validator_responses = await dendrite.forward(axons=validator_axons, synapse=checkpoint_synapse, timeout=60)

            bt.logging.info(f"Requesting checkpoints from validator {self.wallet.hotkey.ss58_address}")

            n_failures = 0
            n_successful_checkpoints = 0
            hotkey_to_received_checkpoint = {}

            hotkey_to_v_trust = {}
            for neuron in self.metagraph.neurons:
                if neuron.validator_trust >= 0:
                    hotkey_to_v_trust[neuron.hotkey] = neuron.validator_trust

            for i, response in enumerate(validator_responses):
                if response.successfully_processed:
                    # Decode from base64 and decompress back into json
                    decoded = base64.b64decode(response.checkpoint)
                    decompressed = gzip.decompress(decoded).decode('utf-8')
                    recv_checkpoint = json.loads(decompressed)

                    hotkey = response.validator_receive_hotkey
                    hotkey_to_received_checkpoint[hotkey] = [hotkey_to_v_trust[hotkey], recv_checkpoint]

                    bt.logging.info(f"Successfully processed checkpoint from axon [{i}/{len(validator_responses)}]: {response.validator_receive_hotkey}")
                    n_successful_checkpoints += 1
                else:
                    n_failures += 1
                    bt.logging.info(f"Checkpoint poke to axon [{i}/{len(validator_responses)}] {response.axon.hotkey} failed")

            bt.logging.info(f"{n_successful_checkpoints} responses succeeded. {n_failures} responses failed")

            if (n_successful_checkpoints > 0 and self.is_testnet) or n_successful_checkpoints >= ValiConfig.MIN_CHECKPOINTS_RECEIVED:
                # sort all our successful responses to get the 10 largest by validator_trust
                sorted_v_trust = sorted(hotkey_to_received_checkpoint.items(), key=lambda item: item[1][0], reverse=True)[:ValiConfig.TOP_N_VTRUST]
                hotkey_to_received_checkpoint = {checkpoint[0]: checkpoint[1] for checkpoint in sorted_v_trust}

                bt.logging.info("Received enough checkpoints, now creating golden.")
                self.create_golden(hotkey_to_received_checkpoint)
                self.created_golden = True
            else:
                bt.logging.info("Not enough checkpoints received to create a golden.")
                self.created_golden = False

        except Exception as e:
            bt.logging.info(f"Error sending checkpoint with error [{e}]")

    def get_validators(self, neurons=None):
        """
        get a list of all validators. defined as:
        stake > 1000 and validator_trust > 0.5
        """
        if self.is_testnet:
            return self.metagraph.axons
            # return [a for a in self.metagraph.axons if a.ip != ValiConfig.AXON_NO_IP]
        if neurons is None:
            neurons = self.metagraph.neurons
        validator_axons = [n.axon_info for n in neurons
                           if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                           and n.axon_info.ip != ValiConfig.AXON_NO_IP]
        return validator_axons

    def get_largest_staked_validators(self, top_n_validators, neurons=None):
        """
        get a list of the trusted validators for checkpoint sending
        return top 20 neurons sorted by stake
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

    def sync_positions_with_cooldown(self):
        # Check if the time is right to sync signals
        if self.is_testnet:
            datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
            # every hour in testnet
            if not (47 < datetime_now.minute < 57):
                return
        else:
            now_ms = TimeUtil.now_in_millis()
            # Already performed a sync recently
            if now_ms - self.last_signal_sync_time_ms < 1000 * 60 * 30:
                return

            # Check if we are between 7:09 AM and 7:19 AM UTC
            datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
            # Temp change time to 21:00 UTC so we can see the effects in shadow mode ASAP
            if not (datetime_now.hour == 21 and (8 < datetime_now.minute < 20)):
                return

        try:
            bt.logging.info("Calling send_checkpoint_requests")
            self.golden = None
            asyncio.run(self.send_checkpoint_requests())
            if self.created_golden:
                bt.logging.info("Calling apply_golden")
                # TODO guard sync_positions with the signal lock once we move on from shadow mode
                self.sync_positions(True, candidate_data=self.golden)
        except Exception as e:
            bt.logging.error(f"Error sending checkpoint: {e}")
            bt.logging.error(traceback.format_exc())

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()
    def create_golden(self, trusted_checkpoints: dict):
        """
        Simple majority approach (preferred to start)
            If a position’s uuid exists on the majority of validators, that position is kept.
            If an order uuid exists in the majority of positions, that order is kept.
                Choose the order with the median price.
        """
        time_now = TimeUtil.now_in_millis()

        position_manager = PositionManager(
            config=None,
            metagraph=None,
            running_unit_tests=False
        )
        eliminations = position_manager.get_eliminations_from_disk()

        position_counts = defaultdict(int)                      # {position_uuid: count}
        position_data = defaultdict(list)                       # {position_uuid: [{position}]}
        position_orders = defaultdict(set)                      # {position_uuid: {order_uuid}}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}

        # simple majority of positions/number of checkpoints
        positions_threshold = math.ceil(len(trusted_checkpoints) / 2)

        # parse each checkpoint to count occurrences of each position and order
        for checkpoint in trusted_checkpoints.values():
            # get positions for each miner
            for miner_positions in checkpoint[1]["positions"].values():
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
                              if count >= positions_threshold}

        golden_positions = defaultdict(lambda: defaultdict(list))

        for checkpoint in trusted_checkpoints.values():
            for miner_hotkey, miner_positions in checkpoint[1]["positions"].items():
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
                        orders_threshold = math.ceil(position_counts[position_uuid] / 2)

                        # get the set of order_uuids that appear in the majority of positions for a position_uuid
                        majority_orders = {order_uuid for order_uuid, count in order_counts[position_uuid].items()
                                           if count >= orders_threshold}

                        # order exists in the majority of positions
                        for order_uuid in position_orders[position_uuid]:
                            if order_uuid in majority_orders:
                                trade_pair = TradePair.to_enum(position["trade_pair"][0])
                                combined_order = self.get_median_order(order_data[order_uuid], trade_pair)
                                majority_orders.remove(order_uuid)
                                new_position.orders.append(combined_order)
                        new_position.orders.sort(key=lambda o: o.processed_ms)
                        new_position.rebuild_position_with_updated_orders()
                        position_dict = json.loads(new_position.to_json_string())
                        golden_positions[miner_hotkey]["positions"].append(position_dict)

        # Construct golden and convert defaultdict to dict
        self.golden = {"created_timestamp_ms": time_now,
                       "eliminations": eliminations,
                       "positions": {miner: dict(golden_positions[miner]) for miner in golden_positions}}
        temp = {k: len(self.golden['positions'][k]) for k in self.golden['positions']}
        bt.logging.info(f"Created golden checkpoint {temp}")

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
    position_syncer = P2PSyncer(is_testnet=True)
    asyncio.run(position_syncer.send_checkpoint_requests())