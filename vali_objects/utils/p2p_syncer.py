import asyncio
import base64
import gzip
import json
import traceback
from collections import defaultdict

import bittensor as bt

import template
from runnable.generate_request_core import generate_request_core
from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.position import Position
from vali_objects.utils.auto_sync import AUTO_SYNC_ORDER_LAG_MS
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.vali_dataclasses.order import Order


class P2PSyncer:
    def __init__(self, wallet=None, metagraph=None):
        self.wallet = wallet
        self.metagraph = metagraph
        self.last_signal_sync_time_ms = 0
        self.checkpoints = []
        self.num_checkpoints_received = 0
        self.golden = {}

    async def send_checkpoint(self):
        """
        serializes checkpoint json and transmits to all validators via synapse
        """
        # get our current checkpoint
        now_ms = TimeUtil.now_in_millis()
        checkpoint_dict = generate_request_core(time_now=now_ms)

        # serialize the position data
        positions = checkpoint_dict['positions']
        # 24 hours in milliseconds
        max_allowed_t_ms = TimeUtil.now_in_millis() - AUTO_SYNC_ORDER_LAG_MS
        for hotkey, positions in positions.items():
            new_positions = []
            positions_deserialized = [Position(**json_positions_dict) for json_positions_dict in positions['positions']]
            for position in positions_deserialized:
                new_orders = []
                for order in position.orders:
                    if order.processed_ms < max_allowed_t_ms:
                        new_orders.append(order)
                if len(new_orders):
                    position.orders = new_orders
                    position.rebuild_position_with_updated_orders()
                    new_positions.append(position)
                else:
                    # if no orders are left, remove the position
                    pass

            positions_serialized = [json.loads(str(p), cls=GeneralizedJSONDecoder) for p in new_positions]
            positions['positions'] = positions_serialized

        # compress json and encode as base64 to keep as a string
        checkpoint_str = json.dumps(checkpoint_dict, cls=CustomEncoder)
        compressed = gzip.compress(checkpoint_str.encode("utf-8"))
        encoded_checkpoint = base64.b64encode(compressed).decode("utf-8")

        # get axons to send checkpoints to
        dendrite = bt.dendrite(wallet=self.wallet)
        validator_axons = self.metagraph.axons

        if len(validator_axons) == 0:
            bt.logging.info(f"no valid validators found. skipping sending checkpoint")
        else:
            # create dendrite and transmit synapse
            checkpoint_synapse = template.protocol.ValidatorCheckpoint(checkpoint=encoded_checkpoint)
            validator_responses = await dendrite.forward(axons=validator_axons, synapse=checkpoint_synapse)

            bt.logging.info(f"sending checkpoint from validator {self.wallet.hotkey.ss58_address}")

            successes = 0
            failures = 0
            for response in validator_responses:
                if response.successfully_processed:
                    print(f"successfully processed ack from {response.validator_receive_hotkey}")
                    successes += 1
                else:
                    failures += 1

            bt.logging.info(f"{successes} responses succeeded")
            bt.logging.info(f"{failures} responses failed")

    def sync_positions_with_cooldown(self, auto_sync_enabled:bool, run_more:bool):
        # Check if the time is right to sync signals
        # TODO: run_more just to make it run constantly. remove when done.
        if not run_more:
            if not auto_sync_enabled:
                return
            now_ms = TimeUtil.now_in_millis()
            # Already performed a sync recently
            if now_ms - self.last_signal_sync_time_ms < 1000 * 60 * 30:
                return

            # Check if we are between 6:09 AM and 6:19 AM UTC
            datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
            if not (datetime_now.hour == 6 and (8 < datetime_now.minute < 20)):
                return

        try:
            bt.logging.info("calling send_checkpoint")
            asyncio.run(self.send_checkpoint())
        except Exception as e:
            bt.logging.error(f"Error syncing positions: {e}")
            bt.logging.error(traceback.format_exc())

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()

    def add_checkpoint(self, received_checkpoint: json, total_checkpoints):
        """
        receives a checkpoint from a trusted validator, and appends to a list
        when all checkpoints are received, build the golden.
        """
        self.checkpoints.append(received_checkpoint)
        self.num_checkpoints_received += 1

        if self.num_checkpoints_received == total_checkpoints:
            self.golden = self.create_golden(self.checkpoints)

            bt.logging.info("successfully created golden checkpoint")
            # print(json.dumps(self.golden, indent=4))

            # TODO: reset num checkpoints at end of cycle
            self.num_checkpoints_received = 0

    def create_golden(self, trusted_checkpoints: list[json]) -> dict:
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
        for checkpoint in trusted_checkpoints:
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

        for checkpoint in trusted_checkpoints:
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
                                # combined_position["orders"].append(combined_order)
                        new_position.rebuild_position_with_updated_orders()
                        position_dict = json.loads(new_position.to_json_string())
                        golden[miner_hotkey]["positions"].append(position_dict)

        # Convert defaultdict to regular dict
        golden = {miner: dict(golden[miner]) for miner in golden}
        return golden

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
    asyncio.run(position_syncer.send_checkpoint())