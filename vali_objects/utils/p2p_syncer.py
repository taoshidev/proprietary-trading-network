import asyncio
import base64
import gzip
import json
import math
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List

import bittensor as bt

import template
from runnable.generate_request_core import generate_request_core
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
        self.last_signal_sync_time_ms = 0

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
            bt.logging.info(f"Validator {self.wallet.hotkey.ss58_address} requesting checkpoints")
            # create dendrite and transmit synapse
            checkpoint_synapse = template.protocol.ValidatorCheckpoint()
            validator_responses = await dendrite.forward(axons=validator_axons, synapse=checkpoint_synapse, timeout=60 * 5)

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

                    bt.logging.info(f"Successfully processed checkpoint from axon [{i+1}/{len(validator_responses)}]: {response.validator_receive_hotkey}")
                    n_successful_checkpoints += 1
                else:
                    n_failures += 1
                    if response.error_message:
                        bt.logging.info(
                            f"Checkpoint poke to axon [{i + 1}/{len(validator_responses)}] {response.axon.hotkey} errored: {response.error_message}")
                    else:
                        bt.logging.info(
                            f"Checkpoint poke to axon [{i + 1}/{len(validator_responses)}] {response.axon.hotkey} failed with status code: {response.axon.status_code}")

            bt.logging.info(f"{n_successful_checkpoints} responses succeeded. {n_failures} responses failed")

            if (n_successful_checkpoints > 0 and self.is_testnet) or n_successful_checkpoints >= ValiConfig.MIN_CHECKPOINTS_RECEIVED:
                # sort all our successful responses by validator_trust
                sorted_v_trust = sorted(hotkey_to_received_checkpoint.items(), key=lambda item: item[1][0], reverse=True)
                hotkey_to_received_checkpoint = {checkpoint[0]: checkpoint[1] for checkpoint in sorted_v_trust}

                bt.logging.info("Received enough checkpoints, now creating golden.")
                self.created_golden = self.create_golden(hotkey_to_received_checkpoint)
            else:
                bt.logging.info("Not enough checkpoints received to create a golden.")
                self.created_golden = False

        except Exception as e:
            bt.logging.info(f"Error generating golden with error [{e}]")

    def create_golden(self, trusted_checkpoints: dict):
        """
        Simple majority approach
            If a position’s uuid exists on the majority of validators, that position is kept.
            If an order uuid exists in the majority of positions, that order is kept.
                Choose the order with the median price.
        """
        for hotkey, chk in trusted_checkpoints.items():
            bt.logging.info(f"{hotkey} sent checkpoint {self.checkpoint_summary(chk[1])}")
            bt.logging.info("--------------------------------------------------")

        position_manager = PositionManager(
            config=None,
            metagraph=None,
            running_unit_tests=False
        )
        golden_eliminations = position_manager.get_eliminations_from_disk()
        golden_positions = defaultdict(lambda: defaultdict(list))

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}
        miner_to_uuids = defaultdict(lambda: defaultdict(set))  # {miner_hotkey: {positions:[position_uuid], orders:[order_uuid]}

        valid_checkpoints = {}

        # parse each checkpoint to count occurrences of each position and order
        for hotkey, checkpoint in trusted_checkpoints.items():
            # get the first 10 up-to-date checkpoints
            if len(valid_checkpoints) >= ValiConfig.TOP_N_CHECKPOINTS:
                break

            checkpoint_position_counts = defaultdict(int)
            checkpoint_order_counts = defaultdict(lambda: defaultdict(int))
            checkpoint_order_data = defaultdict(list)
            checkpoint_miner_to_uuids = defaultdict(lambda: defaultdict(set))

            # add this checkpoint's data if the checkpoint is up-to-date
            latest_order_ms = self.parse_checkpoint_and_last_order_time(checkpoint[1], checkpoint_position_counts, checkpoint_order_counts, checkpoint_order_data, checkpoint_miner_to_uuids)

            if TimeUtil.now_in_millis() - latest_order_ms < 1000 * 60 * 60 * 24:
                for position_uuid in checkpoint_position_counts:
                    position_counts[position_uuid] += checkpoint_position_counts[position_uuid]

                    for order_uuid, count in checkpoint_order_counts[position_uuid].items():
                        order_counts[position_uuid][order_uuid] += count
                        order_data[order_uuid].extend(checkpoint_order_data[order_uuid])

                for miner_hotkey, uuids in checkpoint_miner_to_uuids.items():
                    miner_to_uuids[miner_hotkey]["positions"].update(checkpoint_miner_to_uuids[miner_hotkey]["positions"])
                    miner_to_uuids[miner_hotkey]["orders"].update(checkpoint_miner_to_uuids[miner_hotkey]["orders"])

                valid_checkpoints[hotkey] = checkpoint
            else:
                bt.logging.info(f"Checkpoint from validator {hotkey} is stale with newest order timestamp {latest_order_ms}, {round((TimeUtil.now_in_millis() - latest_order_ms)/(1000 * 60 * 60))} hrs ago, Skipping.")

        if len(valid_checkpoints) == 0:
            bt.logging.info(f"All {len(trusted_checkpoints)} checkpoints are stale, unable to build golden.")
            return False
        else:
            bt.logging.info(f"Building golden from [{len(valid_checkpoints)}/{len(trusted_checkpoints)}] up-to-date checkpoints.")

        # miners who are still running legacy code. do not want to include them in checkpoint
        legacy_miners = self.find_legacy_miners(len(valid_checkpoints), order_counts, miner_to_uuids, position_counts, order_data)

        # get the set of position_uuids that appear in the majority of checkpoints
        positions_threshold = self.consensus_threshold(len(valid_checkpoints))
        majority_positions = {position_uuid for position_uuid, count in position_counts.items()
                              if count >= positions_threshold}
        seen_positions = set()
        seen_orders = set()

        unmatched_positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all unmatched positions on validator]}}}

        for validator_hotkey, checkpoint in valid_checkpoints.items():
            positions = checkpoint[1]["positions"]
            for miner_hotkey, miner_positions in positions.items():
                # if miner_hotkey in legacy_miners:
                #     continue

                positions_in_majority, unmatched_positions = self.sort_positions(miner_hotkey, miner_positions, majority_positions, seen_positions, seen_orders, position_counts, order_counts, order_data)
                golden_positions[miner_hotkey]["positions"].extend(positions_in_majority)
                for position in unmatched_positions:
                    unmatched_positions_matrix[miner_hotkey][position["trade_pair"][0]][validator_hotkey].append(position)
                    position_uuid = position["position_uuid"]
                    bt.logging.info(
                        f"Position {position_uuid} only appeared [{position_counts[position_uuid]}/{len(valid_checkpoints)}] times on miner {miner_hotkey}. Skipping")

        # insert heuristic matched positions back into the golden's positions
        for position in self.heuristic_resolve_positions(unmatched_positions_matrix):
            bt.logging.info(f"Position {position['position_uuid']} matched, adding back in")
            miner_hotkey = position["miner_hotkey"]
            golden_positions[miner_hotkey]["positions"].append(position)

        # Construct golden and convert defaultdict to dict
        self.golden = {"created_timestamp_ms": TimeUtil.now_in_millis(),
                       "hard_snap_cutoff_ms": TimeUtil.now_in_millis() - 1000 * 60 * 15,
                       "eliminations": golden_eliminations,
                       "positions": {miner: dict(golden_positions[miner]) for miner in golden_positions}}
        bt.logging.info(f"Created golden checkpoint: {self.checkpoint_summary(self.golden)}")
        return True

    def sort_positions(self, miner_hotkey, miner_positions, majority_positions, seen_positions, seen_orders, position_counts, order_counts, order_data):
        """
        determine which positions are in the majority, and which ones should be matched up using a heuristic
        """
        positions_in_majority = []
        unmatched_positions = []

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

                # mark a position as seen, so we don't add it multiple times
                majority_positions.remove(position_uuid)
                seen_positions.add(position_uuid)

                # get the set of order_uuids that appear in the majority of positions for a position_uuid
                orders_threshold = self.consensus_threshold(position_counts[position_uuid])
                majority_orders = {order_uuid for order_uuid, count in order_counts[position_uuid].items()
                                   if count >= orders_threshold}

                # order exists in the majority of positions
                for order_uuid in order_counts[position_uuid].keys():
                    if order_uuid in majority_orders:
                        trade_pair = TradePair.from_trade_pair_id(position["trade_pair"][0])
                        combined_order = self.get_median_order(order_data[order_uuid], trade_pair)
                        new_position.orders.append(combined_order)

                        # mark order_uuid as seen, so we don't add it multiple times
                        majority_orders.remove(order_uuid)
                        seen_orders.add(order_uuid)
                    elif order_uuid not in seen_orders:
                        bt.logging.info(
                            f"Order {order_uuid} with Position {position_uuid} only appeared [{order_counts[position_uuid][order_uuid]}/{position_counts[position_uuid]}] times on miner {miner_hotkey}. Skipping")

                new_position.orders.sort(key=lambda o: o.processed_ms)
                new_position.rebuild_position_with_updated_orders()
                position_dict = json.loads(new_position.to_json_string())
                positions_in_majority.append(position_dict)
            # if our position is not in the majority, it may be from legacy miner code, and we want to match with a heuristic.
            elif (position_uuid not in seen_positions
                  and position_counts[position_uuid] != 0):
                unmatched_positions.append(position)
        return positions_in_majority, unmatched_positions

    def checkpoint_summary(self, checkpoint):
        """
        returns a summary of the checkpoint information
        {miner hotkey: [num of positions, num of orders]
        """
        summary = {}
        for miner in checkpoint['positions']:
            orders = 0
            for pos in checkpoint['positions'][miner]['positions']:
                orders += len(pos["orders"])
            pos_orders = [len(checkpoint['positions'][miner]['positions']), orders]
            summary[miner] = pos_orders
        return summary

    def parse_checkpoint_and_last_order_time(self, checkpoint, checkpoint_position_counts, checkpoint_order_counts, checkpoint_order_data, checkpoint_miner_to_uuids):
        """
        determine the latest order on each validator
        save checkpoint data in the same pass
        """
        latest_order_ms = 0
        # get positions for each miner
        positions = checkpoint["positions"]
        for miner_hotkey, miner_positions in positions.items():
            for position in miner_positions["positions"]:
                position_uuid = position["position_uuid"]
                checkpoint_position_counts[position_uuid] += 1
                checkpoint_miner_to_uuids[miner_hotkey]["positions"].add(position_uuid)

                # count and save orders
                for order in position["orders"]:
                    order_uuid = order["order_uuid"]
                    checkpoint_order_counts[position_uuid][order_uuid] += 1
                    checkpoint_order_data[order_uuid].append(dict(order))
                    checkpoint_miner_to_uuids[miner_hotkey]["orders"].add(order_uuid)

                    latest_order_ms = max(latest_order_ms, order["processed_ms"])
        return latest_order_ms

    def find_legacy_miners(self, num_checkpoints, order_counts, miner_to_uuids, position_counts, order_data):
        """
        detect miners running legacy code. miner with legacy code will have all unique uuid's across validators
        so the counts of each position_uuid and order_uuid would be 1. Mark miners with some unique position/order
        uuid's as potentially running legacy code.
        """
        # create single dict of all orders and counts
        all_order_counts = defaultdict(int)  # {order_uuid: count}
        all_order_counts_list = list(order_counts.values())
        for order_count_dict in all_order_counts_list:
            for order_uuid, count in order_count_dict.items():
                all_order_counts[order_uuid] += count

        legacy_miners = set()            # position/order uuids are all unique across validators
        legacy_miner_candidates = set()  # at least one position/order uuid is unique across validators

        if num_checkpoints > 1:
            for miner_hotkey, uuids in miner_to_uuids.items():
                # number of repeated position_uuids and order_uuids
                num_repeated_pos = sum(1 for pos_uuid in uuids["positions"] if position_counts[pos_uuid] > 1)
                num_repeated_orders = sum(1 for order_uuid in uuids["orders"] if all_order_counts[order_uuid] > 1)
                newest_order_timestamp = max([order_data[order_uuid][0]["processed_ms"] for order_uuid in uuids["orders"]])
                newest_unique_order_timestamp = max([order_data[order_uuid][0]["processed_ms"] for order_uuid in uuids["orders"] if all_order_counts[order_uuid] == 1], default=-1)
                newest_unique_order_uuid = next((order_uuid for order_uuid in uuids["orders"]
                                                if all_order_counts[order_uuid] == 1 and order_data[order_uuid][0]["processed_ms"] == newest_unique_order_timestamp), "")

                # if there are positions or orders that only appear once across all validators
                if num_repeated_pos != len(uuids["positions"]) or num_repeated_orders != len(uuids["orders"]):
                    if (num_repeated_pos == 0) and (num_repeated_orders == 0):
                        legacy_miners.add(miner_hotkey)
                    elif newest_unique_order_timestamp == newest_order_timestamp:
                        legacy_miner_candidates.add(miner_hotkey)
                    bt.logging.info(
                        f"Miner {miner_hotkey} has [{(len(uuids['positions']) - num_repeated_pos)}/{len(uuids['positions'])} legacy positions, {(len(uuids['orders']) - num_repeated_orders)}/{len(uuids['orders'])} legacy orders]. Newest legacy order {newest_unique_order_uuid} at timestamp {newest_unique_order_timestamp}")
                else:
                    bt.logging.info(f"Miner {miner_hotkey} has 0 legacy positions or orders")
        bt.logging.info(f"legacy_miners: {legacy_miners}")
        bt.logging.info(f"legacy_miner_candidates: {legacy_miner_candidates}")
        return legacy_miners

    def heuristic_resolve_positions(self, position_matrix) -> List[Position]:
        """
        takes a matrix of unmatched positions, and returns a list of positions to add back in
        position_matrix:
            {miner hotkey: {trade pair: {validator hotkey: [all unmatched positions on validator]}}}
        """
        resolved_position_uuids = set()

        matched_positions = []
        # want to resolve all the unmatched positions for the validators against each other
        for miner_hotkey, trade_pairs in position_matrix.items():
            for trade_pair, validator in trade_pairs.items():
                for validator_hotkey, position_list in validator.items():
                    for position in position_list:
                        matches = self.find_match(position, trade_pairs[trade_pair], resolved_position_uuids, validator_hotkey)
                        if matches is not None:
                            bt.logging.info(f"Miner hotkey {miner_hotkey} has matches {[m['position_uuid'] for m in matches]}")
                            matched_positions.append(matches[0])
        return matched_positions

    def find_match(self, position, trade_pair_validator_positions: dict, resolved_positions: set, corresponding_validator_hotkey) -> List[Position] | None:
        """
        compares a position from validator corresponding_validator_hotkey to all the positions from all the other validators.
        If we are able to match the position to another, we record all the matches, and return the first one as the matched
        position that we will keep.

        position match heuristic:
            same position type
            same # of orders
            opened and closed around SYNC_LOOK_AROUND_MS of each other
        """
        if position["position_uuid"] in resolved_positions:
            return

        matched_positions = [position]

        for validator_hotkey, position_list in trade_pair_validator_positions.items():
            if validator_hotkey == corresponding_validator_hotkey:
                continue

            for p in position_list:
                if p["position_uuid"] in resolved_positions:
                    continue

                # positions have same position_type, # of orders, and open/close_ms times
                if self.dict_positions_aligned(position, p, validate_num_orders=True):
                    matched_positions.append(p)
                    break

                # positions contain orders with the same order_uuid
                aligned_position = self.positions_order_align(position, p)
                if aligned_position is not None:
                    matched_positions.append(json.loads(aligned_position.to_json_string()))
                    break

        for p in matched_positions:
            resolved_positions.add(p["position_uuid"])

        # if the matched positions exceed threshold, we sort the matches by number of orders and then position_uuid
        # make sure that we have matches other than ourselves
        if len(matched_positions) >= self.consensus_threshold(len(matched_positions)) and len(matched_positions) > 1:
            matched_positions.sort(key=lambda x: (-len(x["orders"]), x["position_uuid"]))
            return matched_positions
        return

    def positions_order_align(self, p1, p2):
        """
        we can align positions which have different position_uuids but contain orders with the same order_uuid
        """
        matched_orders = []
        for o1 in p1["orders"]:
            for o2 in p2["orders"]:
                if o1["order_uuid"] == o2["order_uuid"]:
                    matched_orders.append(o2)

        if len(matched_orders) > 0:
            new_position = Position(miner_hotkey=p2["miner_hotkey"],
                                    position_uuid=p2["position_uuid"],
                                    open_ms=p2["open_ms"],
                                    trade_pair=p2["trade_pair"],
                                    orders=matched_orders)
            new_position.orders.sort(key=lambda x: x.processed_ms)
            new_position.rebuild_position_with_updated_orders()
            return new_position

    def consensus_threshold(self, total_items):
        """
        threshold for including a position or order in the golden
        """
        return min(2, math.ceil(total_items / 2))

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

    def get_validators(self, neurons=None):
        """
        get a list of all validators. defined as:
        stake > 1000 and validator_trust > 0.5
        """
        if self.is_testnet:
            return self.metagraph.axons
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

    def sync_positions_with_cooldown(self):
        now_ms = TimeUtil.now_in_millis()
        # Already performed a sync recently
        if now_ms - self.last_signal_sync_time_ms < 1000 * 60 * 15:
            return
        datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
        # Check if the time is right to sync signals
        if self.is_testnet:
            # every hour in testnet
            if not (47 < datetime_now.minute < 57):
                return
        else:
            # Check if we are between 7:09 AM and 7:19 AM UTC
            # Temp change time to 21:00 UTC so we can see the effects in shadow mode ASAP
            if not (datetime_now.hour == 7 and (18 < datetime_now.minute < 30)):
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

if __name__ == "__main__":
    bt.logging.enable_default()
    position_syncer = P2PSyncer(is_testnet=True)
    asyncio.run(position_syncer.send_checkpoint_requests())
    if position_syncer.created_golden:
        position_syncer.sync_positions(True, candidate_data=position_syncer.golden)