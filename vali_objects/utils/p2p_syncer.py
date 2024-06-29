import asyncio
import base64
import gzip
import json
# import random
import traceback

import bittensor as bt
# from tabulate import tabulate

import template
from runnable.generate_request_core import generate_request_core
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.position import Position
from vali_objects.utils.auto_sync import AUTO_SYNC_ORDER_LAG_MS
from vali_objects.utils.vali_bkp_utils import CustomEncoder


class P2PSyncer:
    def __init__(self, wallet=None, metagraph=None, is_testnet=None):
        self.wallet = wallet
        self.metagraph = metagraph
        self.last_signal_sync_time_ms = 0
        self.received_checkpoints = 0
        self.hotkey = self.wallet.hotkey.ss58_address

        # flag for testnet
        self.is_testnet = is_testnet

    async def send_checkpoint(self):
        """
        serializes checkpoint json and transmits to all validators via synapse
        """
        # only trusted validators should send checkpoints
        # TODO: remove testnet flag
        if not self.is_testnet and self.hotkey not in [axon.hotkey for axon in self.get_trusted_validators()]:
            bt.logging.info("Aborting send_checkpoint; not a top trusted validator")
            return

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
        validator_axons = self.get_validators()

        try:
            # create dendrite and transmit synapse
            checkpoint_synapse = template.protocol.ValidatorCheckpoint(checkpoint=encoded_checkpoint)
            validator_responses = await dendrite.forward(axons=validator_axons, synapse=checkpoint_synapse)

            bt.logging.info(f"Sending checkpoint from validator {self.wallet.hotkey.ss58_address}")

            successes = 0
            failures = 0
            for response in validator_responses:
                if response.successfully_processed:
                    bt.logging.info(f"Successfully processed ack from {response.validator_receive_hotkey}")
                    successes += 1
                else:
                    failures += 1

            bt.logging.info(f"{successes} responses succeeded")
            bt.logging.info(f"{failures} responses failed")
        except Exception as e:
            bt.logging.info(f"Error sending checkpoint with error [{e}]")

    def get_validators(self, neurons=None):
        """
        get a list of all validators. defined as:
        stake > 1000 and validator_trust > 0.5
        """
        # TODO: remove testnet flag
        if self.is_testnet:
            return [a for a in self.metagraph.axons if a.ip != ValiConfig.AXON_NO_IP]
        if neurons is None:
            neurons = self.metagraph.neurons
        validator_axons = [n.axon_info for n in neurons
                           if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                           and n.axon_info.ip != ValiConfig.AXON_NO_IP]
        return validator_axons

    def get_trusted_validators(self):
        """
        get a list of the trusted validators for checkpoint sending
        return top 10 neurons sorted by stake
        """
        if self.is_testnet:
            return self.get_validators()
        neurons = self.metagraph.neurons
        sorted_stake_neurons = sorted(neurons, key=lambda n: n.stake, reverse=True)

        return self.get_validators(sorted_stake_neurons)[:ValiConfig.TOP_N]

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
            bt.logging.error(f"Error sending checkpoint: {e}")
            bt.logging.error(traceback.format_exc())

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()

if __name__ == "__main__":
    bt.logging.enable_default()
    position_syncer = P2PSyncer()
    asyncio.run(position_syncer.send_checkpoint())