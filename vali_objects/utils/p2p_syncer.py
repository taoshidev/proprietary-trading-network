import asyncio
import base64
import gzip
import json
import traceback

import bittensor as bt

import template
from runnable.generate_request_core import generate_request_core
from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.position import Position
from vali_objects.utils.auto_sync import AUTO_SYNC_ORDER_LAG_MS
from vali_objects.utils.vali_bkp_utils import CustomEncoder


class P2PSyncer:
    def __init__(self, wallet=None, metagraph=None):
        self.wallet = wallet
        self.metagraph = metagraph
        self.last_signal_sync_time_ms = 0
        self.received_checkpoints = 0
        self.hotkey = self.wallet.hotkey.ss58_address

        # TODO: temp flag for testnet
        self.testnet = True

    async def send_checkpoint(self):
        """
        serializes checkpoint json and transmits to all validators via synapse
        """
        # only trusted validators should send checkpoints
        # TODO: remove testnet flag
        if not self.testnet and self.hotkey not in [axon.hotkey for axon in self.get_trusted_validators()]:
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

    def get_validators(self):
        """
        get a list of all validators. defined as:
        stake > 1000 and validator_trust > 0.5
        """
        # TODO: remove testnet flag
        if self.testnet:
            return self.metagraph.axons
        neurons = self.metagraph.neurons
        validator_neurons = [n for n in neurons if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                             and n.validator_trust > ValiConfig.V_TRUST_MIN]

        return [n.axon_info for n in validator_neurons if n.axon_info.ip != "0.0.0.0"]

    def get_trusted_validators(self):
        """
        get a list of the trusted validators for checkpoint sending. defined as:
        top 10 by stake OR validator_trust > 0.9
        """
        # TODO: filter min stake as well?
        neurons = self.metagraph.neurons
        # only validators with top 10 stake, or v_trust > 0.9 should send checkpoints
        top_stake_neurons = sorted(neurons, key=lambda n: n.stake, reverse=True)
        top_stake_axons = [n.axon_info for n in top_stake_neurons if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                           and n.axon_info.ip != "0.0.0.0"][:ValiConfig.TOP_N]
        high_v_trust_axons = [n.axon_info for n in neurons if n.validator_trust > ValiConfig.V_TRUST_THRESHOLD]

        # axons could contain duplicates, create set based on axon hotkeys and then filter axons
        axons = top_stake_axons + high_v_trust_axons
        trusted_hotkeys = set(a.hotkey for a in axons)
        trusted_axons = [a for a in axons if a.hotkey in trusted_hotkeys and a.ip != "0.0.0.0"]

        return trusted_axons

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