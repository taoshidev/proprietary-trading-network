import asyncio
import base64
import gzip
import json
import math
import traceback
from collections import defaultdict
from typing import List

import bittensor as bt

import template
from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.validator_sync_base import ValidatorSyncBase

class P2PSyncer(ValidatorSyncBase):
    def __init__(self, wallet=None, metagraph=None, is_testnet=None, shutdown_dict=None, signal_sync_lock=None, signal_sync_condition=None, n_orders_being_processed=None, running_unit_tests=False):
        super().__init__(shutdown_dict, signal_sync_lock, signal_sync_condition, n_orders_being_processed, running_unit_tests=running_unit_tests)
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
        position_manager = PositionManager(
            config=None,
            metagraph=None,
            running_unit_tests=False
        )
        golden_eliminations = position_manager.get_eliminations_from_disk()
        golden_positions = defaultdict(lambda: defaultdict(list))

        valid_checkpoints = {}

        # checkpoint is valid/not stale if the last order is recent
        for hotkey, checkpoint in trusted_checkpoints.items():
            # get the first 10 up-to-date checkpoints
            if len(valid_checkpoints) >= ValiConfig.TOP_N_CHECKPOINTS:
                break
            # add this checkpoint's data if the checkpoint is up-to-date
            latest_order_ms = self.last_order_time_in_checkpoint(checkpoint[1])
            if TimeUtil.now_in_millis() - latest_order_ms < 1000 * 60 * 60 * 10:  # validators with no orders processed in 10 hrs are considered stale
                valid_checkpoints[hotkey] = checkpoint[1]
            else:
                bt.logging.info(f"Checkpoint from validator {hotkey} is stale with newest order timestamp {latest_order_ms}, {round((TimeUtil.now_in_millis() - latest_order_ms)/(1000 * 60 * 60))} hrs ago, Skipping.")

        if len(valid_checkpoints) == 0:
            bt.logging.info(f"All {len(trusted_checkpoints)} checkpoints are stale, unable to build golden.")
            return False
        else:
            bt.logging.info(f"Building golden from [{len(valid_checkpoints)}/{len(trusted_checkpoints)}] up-to-date checkpoints.")

        for hotkey, chk in valid_checkpoints.items():
            bt.logging.info(f"{hotkey} sent checkpoint {self.checkpoint_summary(chk)}")
            bt.logging.info("--------------------------------------------------")

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}
        miner_to_uuids = defaultdict(lambda: defaultdict(set))  # {miner_hotkey: {positions:[position_uuid], orders:[order_uuid]}
        miner_counts = defaultdict(int)                         # {miner_hotkey: count}

        positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        orders_matrix = defaultdict(lambda: defaultdict(list))                          # {position_uuid: {validator hotkey: [all orders on validator]}}

        # parse each checkpoint to count occurrences of each position and order
        for hotkey, checkpoint in valid_checkpoints.items():
            self.parse_checkpoint(hotkey, checkpoint, position_counts, order_counts, order_data, miner_to_uuids, miner_counts, positions_matrix, orders_matrix)
        self.prune_position_orders(order_counts)

        # miners who are still running legacy code. do not want to include them in checkpoint
        self.find_legacy_miners(len(valid_checkpoints), order_counts, miner_to_uuids, position_counts, order_data)

        # get the set of position_uuids that appear in the majority of checkpoints
        positions_threshold = self.consensus_threshold(len(valid_checkpoints))
        majority_positions = {position_uuid for position_uuid, count in position_counts.items() if count >= positions_threshold}
        seen_positions = set()
        seen_orders = set()

        for validator_hotkey, checkpoint in valid_checkpoints.items():
            positions = checkpoint["positions"]
            for miner_hotkey, miner_positions in positions.items():
                if miner_counts[miner_hotkey] < positions_threshold:
                    continue

                uuid_matched_positions = self.sort_positions(miner_positions, majority_positions, seen_positions, seen_orders, position_counts, order_counts, order_data, orders_matrix, validator_hotkey)
                golden_positions[miner_hotkey]["positions"].extend(uuid_matched_positions)

        # insert heuristic matched positions back into the golden's positions
        for position in self.heuristic_resolve_positions(positions_matrix, len(valid_checkpoints), seen_positions):
            bt.logging.info(f"Position {position['position_uuid']} on miner {position['miner_hotkey']} matched, adding back in")
            miner_hotkey = position["miner_hotkey"]
            golden_positions[miner_hotkey]["positions"].append(position)

        # Construct golden and convert defaultdict to dict
        self.golden = {"created_timestamp_ms": TimeUtil.now_in_millis(),
                       "hard_snap_cutoff_ms": TimeUtil.now_in_millis() - 1000 * 60 * 15,
                       "eliminations": golden_eliminations,
                       "positions": {miner: dict(golden_positions[miner]) for miner in golden_positions}}
        bt.logging.info(f"Created golden checkpoint: {self.checkpoint_summary(self.golden)}")
        return True

    def sort_positions(self, miner_positions, majority_positions, seen_positions, seen_orders, position_counts, order_counts, order_data, orders_matrix, validator_hotkey):
        """
        determine which positions are in the majority, and which ones should be matched up using a heuristic
        """
        uuid_matched_positions = []
        resolved_orders = set()

        for position in miner_positions["positions"]:
            position_uuid = position["position_uuid"]
            # position exists on majority of validators
            if position_uuid in majority_positions and position_uuid not in seen_positions:
                # create a single combined position, and delete uuid to avoid duplicates
                new_position = Position(**position)
                new_position.orders = []

                # mark a position as seen, so we don't add it multiple times
                seen_positions.add(position_uuid)

                # get the set of order_uuids that appear in the majority of positions for a position_uuid
                orders_threshold = self.consensus_threshold(position_counts[position_uuid])
                majority_orders = {order_uuid for order_uuid, count in order_counts[position_uuid].items()
                                   if count >= orders_threshold}

                # order exists in the majority of positions
                for order_uuid in order_counts[position_uuid].keys():
                    if order_uuid in majority_orders and order_uuid not in seen_orders:
                        trade_pair = TradePair.from_trade_pair_id(position["trade_pair"][0])
                        median_order = self.get_median_order(order_data[order_uuid], trade_pair)
                        new_position.orders.append(median_order)
                        # mark order_uuid as seen, so we don't add it multiple times
                        seen_orders.add(order_uuid)
                    elif order_uuid not in seen_orders:
                        # can have different order_uuids in the same position_uuid
                        # heuristic match up every order, make sure that matched orders do not include any orders that are already in seen_orders
                        matches = self.heuristic_resolve_orders(order_uuid, order_data, position_counts[position_uuid], seen_orders, resolved_orders, orders_matrix, validator_hotkey, position_uuid)
                        if matches is not None:
                            trade_pair = TradePair.from_trade_pair_id(position["trade_pair"][0])
                            matched_order = self.get_median_order(matches, trade_pair)
                            new_position.orders.append(matched_order)
                            seen_orders.update([m["order_uuid"] for m in matches])
                            bt.logging.info(f"Order {order_uuid} with Position {position_uuid} on miner {position['miner_hotkey']} matched with {[m['order_uuid'] for m in matches]}, adding back in")
                        else:
                            bt.logging.info(
                                f"Order {order_uuid} with Position {position_uuid} only appeared [{order_counts[position_uuid][order_uuid]}/{position_counts[position_uuid]}] times on miner {position['miner_hotkey']}. Skipping")

                new_position.orders.sort(key=lambda o: o.processed_ms)
                if len(new_position.orders) > 0 and new_position.orders[0].leverage == 0:
                    bt.logging.info(f"Miner[{new_position.miner_hotkey}] Position [{new_position.position_uuid}] with orders {[o.order_uuid for o in new_position.orders]} first order leverage is 0")
                new_position.rebuild_position_with_updated_orders()
                position_dict = json.loads(new_position.to_json_string())
                uuid_matched_positions.append(position_dict)
        return uuid_matched_positions

    def heuristic_resolve_orders(self, order_uuid: str, order_data: dict, order_in_num_positions: int, seen_orders: set, resolved_orders:set, orders_matrix, corresponding_validator_hotkey, position_uuid):
        """
        heuristic matching for orders with different order_uuids.
        return all the matches if it contains all new orders, and passes threshold
        orders_matrix:
            {position_uuid: {validator hotkey: [all orders on validator]}}
        """
        matches = self.find_matching_orders(order_data[order_uuid][0], resolved_orders, orders_matrix[position_uuid], corresponding_validator_hotkey)
        order_threshold = self.consensus_threshold(order_in_num_positions, heuristic_match=True)
        if (matches is not None
                and len(matches) > order_threshold
                and set([match["order_uuid"] for match in matches]).isdisjoint(seen_orders)):
            # see if no elements from matches have already appeared in uuid_matched_positions
            return matches

    def find_matching_orders(self, order, resolved_orders, orders_matrix, corresponding_validator_hotkey):
        """
        compare an order to all other orders associated with a position, and find all the matches
        sort matches by order_uuid
        """
        if order["order_uuid"] in resolved_orders:
            return
        matched_orders = [order]

        for validator_hotkey, order_list in orders_matrix.items():
            if validator_hotkey == corresponding_validator_hotkey:
                continue

            for o in order_list:
                if o["order_uuid"] in resolved_orders:
                    continue

                if self.dict_orders_aligned(order, o):
                    matched_orders.append(o)
                    break

        for o in matched_orders:
            resolved_orders.add(o["order_uuid"])

        # if the matched positions exceed threshold, we sort the matches by number of orders and then position_uuid
        # make sure that we have matches other than ourselves
        if len(matched_orders) > 1:
            matched_orders.sort(key=lambda x: x["order_uuid"])
            return matched_orders
        return

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

    def last_order_time_in_checkpoint(self, checkpoint):
        """
        determine the latest order on each validator checkpoint
        """
        latest_order_ms = 0
        # get positions for each miner
        positions = checkpoint["positions"]
        for miner_hotkey, miner_positions in positions.items():
            for position in miner_positions["positions"]:
                for order in position["orders"]:
                    latest_order_ms = max(latest_order_ms, order["processed_ms"])
        return latest_order_ms

    def parse_checkpoint(self, validator_hotkey, checkpoint, position_counts, order_counts, order_data, miner_to_uuids, miner_counts, positions_matrix, orders_matrix):
        """
        parse checkpoint data

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}
        miner_to_uuids = defaultdict(lambda: defaultdict(set))  # {miner_hotkey: {positions:[position_uuid], orders:[order_uuid]}
        miner_counts = defaultdict(int)                         # {miner_hotkey: count}

        positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        orders_matrix = defaultdict(lambda: defaultdict(list))                          # {position_uuid: {validator_hotkey: [all orders]}}
        """
        # get positions for each miner
        positions = checkpoint["positions"]
        for miner_hotkey, miner_positions in positions.items():
            miner_counts[miner_hotkey] += 1
            for position in miner_positions["positions"]:
                position_uuid = position["position_uuid"]
                position_counts[position_uuid] += 1
                miner_to_uuids[miner_hotkey]["positions"].add(position_uuid)

                # count and save orders
                for order in position["orders"]:
                    order_uuid = order["order_uuid"]
                    order_counts[position_uuid][order_uuid] += 1
                    order_data[order_uuid].append(dict(order))
                    miner_to_uuids[miner_hotkey]["orders"].add(order_uuid)
                    orders_matrix[position_uuid][validator_hotkey].append(order)

                positions_matrix[miner_hotkey][position["trade_pair"][0]][validator_hotkey].append(position)

    def prune_position_orders(self, order_counts):
        """
        some validators may incorrectly combine multiple positions into one, so an order_uuid may appear under
        multiple positions. We will only keep the order_uuid in the position that it appears the most in.

        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        """
        order_max_count_appears_in_position = defaultdict(tuple)       # {order_uuid: (position_uuid, count)}
        # find the position_uuid with the max count for each order_uuid
        for position, orders in order_counts.items():
            for order, count in orders.items():
                if order not in order_max_count_appears_in_position:
                    order_max_count_appears_in_position[order] = (position, count)
                else:
                    if count > order_max_count_appears_in_position[order][1]:
                        order_max_count_appears_in_position[order] = (position, count)

        # remove order_uuid from all other position_uuids
        for position, orders in order_counts.items():
            for order in list(orders.keys()):
                if position != order_max_count_appears_in_position[order][0]:
                    del orders[order]

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

    def heuristic_resolve_positions(self, positions_matrix, num_checkpoints, seen_positions) -> List[Position]:
        """
        takes a matrix of unmatched positions, and returns a list of positions to add back in
        positions_matrix:
            {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        """
        resolved_position_uuids = set()

        matched_positions = []
        # want to resolve all the unmatched positions for the validators against each other
        for miner_hotkey, trade_pairs in positions_matrix.items():
            for trade_pair, validator in trade_pairs.items():
                for validator_hotkey, position_list in validator.items():
                    for position in position_list:
                        matches = self.find_match(position, trade_pairs[trade_pair], resolved_position_uuids, validator_hotkey)
                        if (matches is not None
                                and len(matches) > self.consensus_threshold(num_checkpoints, heuristic_match=True)
                                and set([match["position_uuid"] for match in matches]).isdisjoint(seen_positions)):
                            # want to find positions with the median number of orders
                            goal_order_count = len(matches[len(matches) // 2]["orders"])
                            # in case the median position misses orders, we
                            # want to take the position with the most orders where
                            # there are at least 2 positions with that many orders
                            for i in range(len(matches)-1):
                                if len(matches[i]["orders"]) == len(matches[i+1]["orders"]):
                                    goal_order_count = max(goal_order_count, len(matches[i]["orders"]))
                                    break
                            matches_with_goal_order_count = [match for match in matches if len(match["orders"]) == goal_order_count]
                            bt.logging.info(f"Miner hotkey {miner_hotkey} has matches {[m['position_uuid'] for m in matches]}. goal_order_count: {goal_order_count}. matches_with_goal_order_count: {matches_with_goal_order_count}.")
                            matched_positions.append(matches_with_goal_order_count[0])
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
                # or positions contain orders that are matched or have the same order_uuid
                if (self.dict_positions_aligned(position, p, validate_num_orders=True) or
                        self.positions_order_aligned(position, p)):
                    matched_positions.append(p)
                    break

        for p in matched_positions:
            resolved_positions.add(p["position_uuid"])

        # we sort the matches by number of orders and then position_uuid
        # make sure that we have matches other than ourselves
        if len(matched_positions) > 1:
            matched_positions.sort(key=lambda x: (-len(x["orders"]), x["position_uuid"]))
            return matched_positions
        return

    def consensus_threshold(self, total_items, heuristic_match=False):
        """
        threshold for including a position or order in the golden
        """
        if heuristic_match:
            return math.floor(total_items / 2)
        else:
            return math.ceil(total_items / 2)

    def get_median_order(self, orders, trade_pair) -> Order:
        """
        select the order with the median price from a list of orders with the same order_uuid
        """
        sorted_orders = sorted(orders, key=lambda o: o["price"])
        median_order = sorted_orders[len(orders)//2]
        median_order["trade_pair"] = trade_pair
        return Order(**median_order)

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
            if not (datetime_now.hour == 1 and (18 < datetime_now.minute < 30)):
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
