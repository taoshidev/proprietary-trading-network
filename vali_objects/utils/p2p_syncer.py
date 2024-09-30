import base64
import gzip
import json
import math
import statistics
import traceback
from collections import defaultdict
from typing import List, Set

import bittensor as bt
from bittensor import AxonInfo, NeuronInfo

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
        self.running_unit_tests = running_unit_tests
        self.min_checkpoints = 1 if running_unit_tests else ValiConfig.MIN_CHECKPOINTS_RECEIVED

    def send_checkpoint_requests(self):
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
            validator_responses = dendrite.query(axons=validator_axons,  synapse=checkpoint_synapse, timeout=60 * 5)

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

            if (n_successful_checkpoints > 0 and self.is_testnet) or n_successful_checkpoints >= self.min_checkpoints:
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

    def create_golden(self, trusted_checkpoints: dict) -> bool:
        """
        Create golden checkpoint from active validators (received order in last 10 hrs)
        """
        position_manager = PositionManager(
            config=None,
            metagraph=None,
            running_unit_tests=False
        )

        valid_checkpoints = {}

        # checkpoint is valid/not stale if the last order is recent
        for hotkey, checkpoint in trusted_checkpoints.items():
            # get the first 10 up-to-date checkpoints
            if len(valid_checkpoints) >= ValiConfig.TOP_N_CHECKPOINTS:
                break
            # add this checkpoint's data if the checkpoint is up-to-date
            latest_order_ms = self.last_order_time_in_checkpoint(checkpoint[1])
            if TimeUtil.now_in_millis() - latest_order_ms < 1000 * 60 * 60 * 10 or self.running_unit_tests:  # validators with no orders processed in 10 hrs are considered stale
                valid_checkpoints[hotkey] = checkpoint[1]
            else:
                bt.logging.info(f"Checkpoint from validator {hotkey} is stale with newest order timestamp {latest_order_ms}, {round((TimeUtil.now_in_millis() - latest_order_ms)/(1000 * 60 * 60))} hrs ago, Skipping.")

        if len(valid_checkpoints) < self.min_checkpoints:
            bt.logging.info(f"Only {len(valid_checkpoints)} checkpoints are not stale, unable to build golden. Min required: {self.min_checkpoints}")
            return False
        else:
            bt.logging.info(f"Building golden from [{len(valid_checkpoints)}/{len(trusted_checkpoints)}] up-to-date checkpoints.")

        for hotkey, chk in valid_checkpoints.items():
            bt.logging.info(f"{hotkey} sent checkpoint {self.checkpoint_summary(chk)}")
            bt.logging.info("--------------------------------------------------")

        golden_eliminations = position_manager.get_eliminations_from_disk()
        golden_positions = self.p2p_sync_positions(valid_checkpoints)
        golden_challengeperiod = self.p2p_sync_challengeperiod(valid_checkpoints)

        self.golden = {
            "created_timestamp_ms": TimeUtil.now_in_millis(),
            "hard_snap_cutoff_ms": TimeUtil.now_in_millis() - 1000 * 60 * 15,
            "eliminations": golden_eliminations,
            "positions": golden_positions,
            "challengeperiod": golden_challengeperiod
        }

        bt.logging.info(f"Created golden checkpoint: {self.checkpoint_summary(self.golden)}")
        return True

    def p2p_sync_challengeperiod(self, valid_checkpoints: dict):
        """
        hotkeys in challenge period determined by simple majority. use the median timestamp for each
        """
        challengeperiod_testing_data = defaultdict(list)        # {hotkey: [time]}
        challengeperiod_success_data = defaultdict(list)        # {hotkey: [time]}

        challengeperiod_testing = defaultdict(int)              # {hotkey: time}
        challengeperiod_success = defaultdict(int)              # {hotkey: time}

        for hotkey, checkpoint in valid_checkpoints.items():
            self.parse_checkpoint_challengeperiod(checkpoint, challengeperiod_testing_data, challengeperiod_success_data)

        threshold = self.consensus_threshold(len(valid_checkpoints))
        majority_testing = {hotkey for hotkey, times in challengeperiod_testing_data.items() if len(times) > threshold}
        majority_success = {hotkey for hotkey, times in challengeperiod_success_data.items() if len(times) > threshold}

        for hotkey in majority_testing:
            challengeperiod_testing[hotkey] = statistics.median_low(challengeperiod_testing_data[hotkey])
        for hotkey in majority_success:
            challengeperiod_success[hotkey] = statistics.median_low(challengeperiod_success_data[hotkey])

        return {"testing": challengeperiod_testing, "success": challengeperiod_success}

    def parse_checkpoint_challengeperiod(self, checkpoint: dict, testing_data: dict, success_data: dict):
        """
        parse checkpoint challengeperiod data
        testing_data = {hotkeys in challengeperiod test: [time]}
        success_data = {hotkeys in challengperiod success: [time]}
        """
        challengeperiod_testing = checkpoint.get("challengeperiod", {}).get("testing", {})
        for hotkey, timestamp in challengeperiod_testing.items():
            testing_data[hotkey].append(timestamp)

        challengeperiod_success = checkpoint.get("challengeperiod", {}).get("success", {})
        for hotkey, timestamp in challengeperiod_success.items():
            success_data[hotkey].append(timestamp)

    def p2p_sync_positions(self, valid_checkpoints: dict):
        """
        Simple majority approach
            If a position’s uuid exists on the majority of validators, that position is kept.
            If an order uuid exists in the majority of positions, that order is kept.
                Choose the order with the median price.
        """
        golden_positions = defaultdict(lambda: defaultdict(list))

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}
        miner_to_uuids = defaultdict(lambda: defaultdict(set))  # {miner_hotkey: {positions:[position_uuid], orders:[order_uuid]}
        miner_counts = defaultdict(int)                         # {miner_hotkey: count}

        positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        orders_matrix = defaultdict(lambda: defaultdict(list))                          # {position_uuid: {validator hotkey: [all orders on validator]}}

        position_validators = defaultdict(set)                  # {position_uuid: {validator hotkeys}}
        order_validators = defaultdict(set)                     # {order_uuid: {validator hotkeys}}

        # parse each checkpoint to count occurrences of each position and order
        for hotkey, checkpoint in valid_checkpoints.items():
            self.parse_checkpoint_positions(hotkey, checkpoint, position_counts, order_counts, order_data, miner_to_uuids, miner_counts, positions_matrix, orders_matrix, position_validators, order_validators)
        self.prune_position_orders(order_counts, orders_matrix)

        # miners who are still running legacy code. do not want to include them in checkpoint
        self.find_legacy_miners(len(valid_checkpoints), order_counts, miner_to_uuids, position_counts, order_data)

        # get the set of position_uuids that appear in the majority of checkpoints
        threshold = self.consensus_threshold(len(valid_checkpoints))
        seen_positions = set()
        seen_orders = set()

        for validator_hotkey, checkpoint in valid_checkpoints.items():
            positions = checkpoint.get("positions", {})
            for miner_hotkey, miner_positions in positions.items():
                if miner_counts[miner_hotkey] <= threshold:
                    continue
                matched_positions = self.construct_common_positions(miner_positions, seen_positions, seen_orders, order_data, orders_matrix, positions_matrix, validator_hotkey, threshold, position_validators, order_validators)
                golden_positions[miner_hotkey]["positions"].extend(matched_positions)

        # convert to dict
        return {miner: dict(golden_positions[miner]) for miner in golden_positions}

    def construct_common_positions(self, miner_positions: dict, seen_positions: Set[str], seen_orders: Set[str], order_data: dict, orders_matrix: dict, positions_matrix: dict, validator_hotkey: str, threshold: int, position_validators: dict, order_validators: dict) -> List[dict]:
        """
        return the positions to add to golden, by attempting to match each position up with others.
        construct each position from its orders. if the order appears in the majority then it is taken, otherwise
        the order is attempted to be matched to other orders using a heuristic.

        order_data = defaultdict(list)                          # {order_uuid: [{order}]}

        positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        orders_matrix = defaultdict(lambda: defaultdict(list))  # {position_uuid: {validator hotkey: [all orders on validator]}}
        """
        uuid_matched_positions = []
        resolved_orders = set()  # separate from seen_orders, because we want to be able to match with seen orders
        resolved_positions = set()

        for position in miner_positions["positions"]:
            position_uuid = position["position_uuid"]
            miner_hotkey = position["miner_hotkey"]
            trade_pair_id = position["trade_pair"][0]

            if position_uuid not in seen_positions:
                # find all matches for a position
                matched_positions = self.find_matching_positions(position, positions_matrix[miner_hotkey][trade_pair_id], resolved_positions, validator_hotkey, position_validators)
                matched_position_uuids = [p["position_uuid"] for p in matched_positions]
                matched_position_uuids_set = set(matched_position_uuids)
                # see if some positions are heuristically matched
                if len(matched_position_uuids_set) != 1:
                    bt.logging.info(f"Positions matched for miner {miner_hotkey}: {matched_position_uuids}")

                # ensure that we have not previously encountered/added this position or its matches
                if len(matched_positions) > threshold and matched_position_uuids_set.isdisjoint(seen_positions):
                    new_position = Position(**position)
                    new_position.orders = []

                    # get all the orders that appear in these positions
                    orders_in_matched_positions = set()
                    # all the validators and orders for the positions
                    matched_positions_orders_matrix = defaultdict(list)
                    for p in matched_positions:
                        orders_in_matched_positions.update([o["order_uuid"] for o in p["orders"]])
                        matched_positions_orders_matrix.update(orders_matrix[p["position_uuid"]])

                    # find all the common orders
                    for order_uuid in orders_in_matched_positions:
                        if order_uuid not in seen_orders:
                            matched_orders = self.find_matching_orders(order_data[order_uuid][0], matched_positions_orders_matrix, resolved_orders, order_validators)
                            matched_order_uuids = [o["order_uuid"] for o in matched_orders]
                            matched_order_uuids_set = set(matched_order_uuids)
                            # see if some orders are heuristically matched
                            if len(matched_order_uuids_set) != 1:
                                bt.logging.info(f"Orders matched for miner {miner_hotkey}: {matched_order_uuids}, across positions: {matched_position_uuids}")

                            # ensure that we have not previously encountered/added this order or its matches
                            if len(matched_orders) > threshold and matched_order_uuids_set.isdisjoint(seen_orders):
                                # add an order to position
                                trade_pair = TradePair.from_trade_pair_id(trade_pair_id)
                                median_order = self.get_median_order(matched_orders, trade_pair)
                                new_position.orders.append(median_order)
                            # add to seen orders
                            seen_orders.update(matched_order_uuids_set)
                    # sort orders by processed time
                    new_position.orders.sort(key=lambda o: o.processed_ms)
                    try:
                        new_position.rebuild_position_with_updated_orders()
                        position_dict = json.loads(new_position.to_json_string())
                        uuid_matched_positions.append(position_dict)
                    except ValueError as v:
                        bt.logging.info(f"Miner [{new_position.miner_hotkey}] Position [{new_position.position_uuid}] Orders {[o.order_uuid for o in new_position.orders]} ValueError {v}")
                # add to seen positions
                seen_positions.update(matched_position_uuids_set)
        return uuid_matched_positions

    def find_matching_orders(self, order: dict, validator_to_orders: dict, resolved_orders: Set[str], order_validators: dict) -> List[dict] | None:
        """
        compare an order to all other orders associated with a position, and find all the matches using a heuristic.
        sort matches by order_uuid.

        validator_to_orders = defaultdict(list)  # {validator hotkey: [all orders on validator]}
        """
        if order["order_uuid"] in resolved_orders:
            return []
        matched_orders = []

        for validator_hotkey, order_list in validator_to_orders.items():
            for o in order_list:
                if o["order_uuid"] in resolved_orders:
                    continue

                # if 2 orders order and o appear under the same checkpoint, they must be distinct and cannot match.
                if o["order_uuid"] != order["order_uuid"] and not order_validators[order["order_uuid"]].isdisjoint(order_validators[o["order_uuid"]]):
                    continue

                # orders must have the same order_uuid or same leverage, order_type, and processed_ms
                if self.dict_orders_aligned(order, o):
                    matched_orders.append(o)
                    break

        for o in matched_orders:
            resolved_orders.add(o["order_uuid"])

        matched_orders.sort(key=lambda x: x["order_uuid"])
        return matched_orders

    def checkpoint_summary(self, checkpoint: dict) -> dict:
        """
        returns a summary of the checkpoint information
        {miner hotkey: [num of positions, num of orders]
        """
        summary = {}
        positions = checkpoint.get("positions", {})
        for miner_hotkey, miner_positions in positions.items():
            orders = 0
            for pos in miner_positions['positions']:
                orders += len(pos["orders"])
            pos_orders = [len(miner_positions['positions']), orders]
            summary[miner_hotkey] = pos_orders
        return summary

    def last_order_time_in_checkpoint(self, checkpoint: dict) -> int:
        """
        determine the latest order on each validator checkpoint
        """
        latest_order_ms = 0
        # get positions for each miner
        positions = checkpoint.get("positions", {})
        for miner_hotkey, miner_positions in positions.items():
            for position in miner_positions["positions"]:
                for order in position["orders"]:
                    latest_order_ms = max(latest_order_ms, order["processed_ms"])
        return latest_order_ms

    def parse_checkpoint_positions(self, validator_hotkey: str, checkpoint: dict, position_counts: dict, order_counts: dict, order_data: dict, miner_to_uuids: dict, miner_counts: dict, positions_matrix: dict, orders_matrix: dict, position_validators: dict, order_validators: dict):
        """
        parse checkpoint data

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}
        miner_to_uuids = defaultdict(lambda: defaultdict(set))  # {miner_hotkey: {positions:[position_uuid], orders:[order_uuid]}
        miner_counts = defaultdict(int)                         # {miner_hotkey: count}

        positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        orders_matrix = defaultdict(lambda: defaultdict(list))                          # {position_uuid: {validator_hotkey: [all orders]}}

        position_validators = defaultdict(set)                  # {position_uuid: [validator hotkeys]}
        order_validators = defaultdict(set)                     # {order_uuid: [validator hotkeys]}
        """
        # get positions for each miner
        positions = checkpoint.get("positions", {})
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
                    order_validators[order_uuid].add(validator_hotkey)

                orders_matrix[position_uuid][validator_hotkey].sort(key=lambda o: o["processed_ms"])
                positions_matrix[miner_hotkey][position["trade_pair"][0]][validator_hotkey].append(position)
                position_validators[position_uuid].add(validator_hotkey)

    def prune_position_orders(self, order_counts: dict, orders_matrix: dict):
        """
        some validators may incorrectly combine multiple positions into one, so an order_uuid may appear under
        multiple positions. We will only keep the order_uuid in the position that it appears the most in.

        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        orders_matrix = defaultdict(lambda: defaultdict(list))  # {position_uuid: {validator hotkey: [all orders on validator]}}
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
                max_position = order_max_count_appears_in_position[order][0]
                if position != max_position:
                    del orders[order]
                    order_counts[max_position][order] += 1

                    for vali_orders in orders_matrix[position].values():
                        vali_orders.remove(order) if order in vali_orders else None

    def find_legacy_miners(self, num_checkpoints: int, order_counts: dict, miner_to_uuids: dict, position_counts: dict, order_data: dict) -> Set[str]:
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

    def find_matching_positions(self, position: dict, trade_pair_validator_positions: dict, resolved_positions: set, corresponding_validator_hotkey: str, position_validators: dict) -> List[dict]:
        """
        compares a position from corresponding_validator_hotkey to all other positions with matching trade pair from all the other validators.
        positions are matched with a heuristic, and returned in a list sorted by number of orders then position_uuid

        position match heuristic:
            same position type
            same # of orders
            opened and closed around SYNC_LOOK_AROUND_MS of each other

        trade_pair_validator_positions: {validator hotkey: [all positions on validator]}
        """
        if position["position_uuid"] in resolved_positions:
            return []

        matched_positions = [position]

        for validator_hotkey, position_list in trade_pair_validator_positions.items():
            if validator_hotkey == corresponding_validator_hotkey:
                continue

            for p in position_list:
                if p["position_uuid"] in resolved_positions:
                    continue

                # if 2 positions position and p appear under the same checkpoint, they must be distinct and cannot match.
                if p["position_uuid"] != position["position_uuid"] and not position_validators[p["position_uuid"]].isdisjoint(position_validators[position["position_uuid"]]):
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
        matched_positions.sort(key=lambda x: (-len(x["orders"]), x["position_uuid"]))
        return matched_positions

    def consensus_threshold(self, total_items: int) -> int:
        """
        threshold for including a position or order in the golden
        """
        return math.floor(total_items / 2)

    def get_median_order(self, orders: List[dict], trade_pair: TradePair) -> Order:
        """
        select the order with the median price from a list of orders with the same order_uuid
        """
        sorted_orders = sorted(orders, key=lambda o: o["price"])
        median_order = sorted_orders[len(orders)//2]
        median_order["trade_pair"] = trade_pair
        return Order(**median_order)

    def get_validators(self, neurons: List[NeuronInfo]=None) -> List[AxonInfo]:
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

    def get_largest_staked_validators(self, top_n_validators: int, neurons: List[NeuronInfo]=None) -> List[AxonInfo]:
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
            # Check if we are between 7:19 AM and 7:29 AM UTC
            if not (datetime_now.hour == 1 and (18 < datetime_now.minute < 30)):
                return

        try:
            bt.logging.info("Sending checkpoint requests")
            self.golden = None
            self.send_checkpoint_requests()
            if self.created_golden:
                bt.logging.info("Golden created. Syncing positions.")
                with self.signal_sync_lock:
                    while self.n_orders_being_processed[0] > 0:
                        self.signal_sync_condition.wait()
                    self.sync_positions(False, candidate_data=self.golden)
        except Exception as e:
            bt.logging.error(f"Error sending checkpoint: {e}")
            bt.logging.error(traceback.format_exc())

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()

if __name__ == "__main__":
    bt.logging.enable_default()
    position_syncer = P2PSyncer(is_testnet=True)
    position_syncer.send_checkpoint_requests()
    if position_syncer.created_golden:
        position_syncer.sync_positions(False, candidate_data=position_syncer.golden)
