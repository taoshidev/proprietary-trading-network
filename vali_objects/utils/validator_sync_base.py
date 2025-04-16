import time
import traceback
from copy import deepcopy
from enum import Enum
from collections import defaultdict

from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
import bittensor as bt

from vali_objects.utils.vali_utils import ValiUtils
AUTO_SYNC_ORDER_LAG_MS = 1000 * 60 * 60 * 24

# Make an enum class that represents how the position sync went. "Nothing", "Updated", "Deleted", "Inserted"
class PositionSyncResult(Enum):
    NOTHING = 0
    UPDATED = 1
    DELETED = 2
    INSERTED = 3

# Create a new type of exception PositionSyncResultException
class PositionSyncResultException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ValidatorSyncBase():
    def __init__(self, shutdown_dict=None, signal_sync_lock=None, signal_sync_condition=None,
                 n_orders_being_processed=None, running_unit_tests=False, position_manager=None,
                 ipc_manager=None):
        self.is_mothership = 'mothership' in ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        self.SYNC_LOOK_AROUND_MS = 1000 * 60 * 3
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict
        self.last_signal_sync_time_ms = 0
        self.signal_sync_lock = signal_sync_lock
        self.signal_sync_condition = signal_sync_condition
        self.n_orders_being_processed = n_orders_being_processed
        if ipc_manager:
            self.perf_ledger_hks_to_invalidate = ipc_manager.dict()
        else:
            self.perf_ledger_hks_to_invalidate = {}  # {hk: timestamp_ms}
        self.init_data()

    def init_data(self):
        self.global_stats = defaultdict(int)

        self.miners_with_order_deletion = set()
        self.miners_with_order_insertion = set()
        self.miners_with_order_matched = set()
        self.miners_with_order_kept = set()

        self.miners_with_position_deletion = set()
        self.miners_with_position_insertion = set()
        self.miners_with_position_matched = set()
        self.miners_with_position_kept = set()
        self.perf_ledger_hks_to_invalidate.clear()

    def sync_positions(self, shadow_mode, candidate_data=None, disk_positions=None) -> dict[str: list[Position]]:
        t0 = time.time()
        self.init_data()
        assert candidate_data, "Candidate data must be provided"

        backup_creation_time_ms = candidate_data['created_timestamp_ms']
        bt.logging.info(f"Automated sync. Shadow mode {shadow_mode}. Found backup creation time: {TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)}")

        candidate_hk_to_positions = {}
        for hk, data in candidate_data['positions'].items():
            positions = data['positions']
            candidate_hk_to_positions[hk] = [Position(**p) for p in positions]

        # The candidate dataset is time lagged. We only delete non-matching data if they occured during the window of the candidate data.
        # We want to account for a few minutes difference of possible orders that came in after a retry.
        if 'hard_snap_cutoff_ms' in candidate_data:
            hard_snap_cutoff_ms = candidate_data['hard_snap_cutoff_ms']
        else:
            hard_snap_cutoff_ms = backup_creation_time_ms - AUTO_SYNC_ORDER_LAG_MS
        bt.logging.info(
            f"Automated sync. hard_snap_cutoff_ms: {TimeUtil.millis_to_formatted_date_str(hard_snap_cutoff_ms)}")

        if self.is_mothership:
            bt.logging.info("Mothership detected")

        if disk_positions is None:
            disk_positions = self.position_manager.get_positions_for_all_miners(sort_positions=True)

        eliminations = candidate_data['eliminations']
        if not self.is_mothership:
            removed = self.position_manager.elimination_manager.sync_eliminations(eliminations)
            for hk in removed:
                self.perf_ledger_hks_to_invalidate[hk] = 0


        challenge_period_data = candidate_data.get('challengeperiod')
        if challenge_period_data:  # Only in autosync as of now.
            orig_testing_keys = set(self.position_manager.challengeperiod_manager.challengeperiod_testing.keys())
            orig_success_keys = set(self.position_manager.challengeperiod_manager.challengeperiod_success.keys())
            new_testing_keys = set(challenge_period_data.get('testing').keys())
            new_success_keys = set(challenge_period_data.get('success').keys())
            bt.logging.info(f"Challengeperiod testing sync keys added: {new_testing_keys-orig_testing_keys}\n"
                            f"Challengeperiod testing sync keys removed: {orig_testing_keys - new_testing_keys}\n"
                            f"Challengeperiod success sync keys added: {new_success_keys - orig_success_keys}\n"
                            f"Challengeperiod success sync keys removed: {orig_success_keys - new_success_keys}")
            if not shadow_mode:
                self.position_manager.challengeperiod_manager.sync_challenege_period_data(challenge_period_data.get('testing', {}),
                                                                                          challenge_period_data.get('success', {}))
        eliminated_hotkeys = set([e['hotkey'] for e in eliminations])
        # For a healthy validator, the existing positions will always be a superset of the candidate positions
        for hotkey, positions in candidate_hk_to_positions.items():
            if self.shutdown_dict:
                return
            if hotkey in eliminated_hotkeys:
                self.global_stats['n_miners_skipped_eliminated'] += 1
                continue
            self.position_manager.dedupe_positions(positions, hotkey)
            self.global_stats['n_miners_synced'] += 1
            candidate_positions_by_trade_pair = self.partition_positions_by_trade_pair(positions)
            existing_positions_by_trade_pair = self.partition_positions_by_trade_pair(disk_positions.get(hotkey, []))
            unified_trade_pairs = set(candidate_positions_by_trade_pair.keys()) | set(existing_positions_by_trade_pair.keys())
            for trade_pair in unified_trade_pairs:
                if self.shutdown_dict:
                    return
                candidate_positions = candidate_positions_by_trade_pair.get(trade_pair, [])
                existing_positions = existing_positions_by_trade_pair.get(trade_pair, [])

                try:
                    position_to_sync_status, min_timestamp_of_change, stats = self.resolve_positions(candidate_positions, existing_positions, trade_pair, hotkey, hard_snap_cutoff_ms)
                    if min_timestamp_of_change != float('inf'):
                        self.perf_ledger_hks_to_invalidate[hotkey] = (
                            min_timestamp_of_change) if hotkey not in self.perf_ledger_hks_to_invalidate else (
                            min(self.perf_ledger_hks_to_invalidate[hotkey], min_timestamp_of_change))
                        if not shadow_mode:
                            self.write_modifications(position_to_sync_status, stats)
                except Exception as e:
                    full_traceback = traceback.format_exc()
                    # Slice the last 1000 characters of the traceback
                    limited_traceback = full_traceback[-1000:]
                    bt.logging.error(f"Error syncing positions for hotkey {hotkey} trade pair {trade_pair.trade_pair}. Error: {e} traceback: {limited_traceback}")
                    # If this is PositionSyncResultException, throw it up. Otherwise, log the error and continue.
                    if isinstance(e, PositionSyncResultException):
                        raise e
                    else:
                        self.global_stats['exceptions_seen'] += 1


        # count sets
        self.global_stats['n_miners_positions_deleted'] = len(self.miners_with_position_deletion)
        self.global_stats['n_miners_positions_inserted'] = len(self.miners_with_position_insertion)
        self.global_stats['n_miners_positions_matched'] = len(self.miners_with_position_matched)
        self.global_stats['n_miners_positions_kept'] = len(self.miners_with_position_kept)

        self.global_stats['n_miners_orders_deleted'] = len(self.miners_with_order_deletion)
        self.global_stats['n_miners_orders_inserted'] = len(self.miners_with_order_insertion)
        self.global_stats['n_miners_orders_matched'] = len(self.miners_with_order_matched)
        self.global_stats['n_miners_orders_kept'] = len(self.miners_with_order_kept)

        # Print self.global_stats
        bt.logging.info("Global stats:")
        for k, v in self.global_stats.items():
            bt.logging.info(f"  {k}: {v}")
        bt.logging.info(f"Position sync took {time.time() - t0} seconds")

    def write_modifications(self, position_to_sync_status, stats):
        # Ensure the enums align with the global stats
        kept_and_matched = stats['kept'] + stats['matched']
        deleted = stats['deleted']
        inserted = stats['inserted']

        # handle having multiple open positions for a hotkey
        # close the older open position
        prev_open_position = None

        #for position, sync_status in position_to_sync_status.items():
        #    position_debug_sting = f'---debug printing pos to ss: {position.trade_pair.trade_pair_id} n_orders {len(position.orders)}'
        #    print(position_debug_sting)
        #    self.debug_print_pos(position)
        #    print('---status', sync_status)

        # Deletions happen first
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.DELETED:
                deleted -= 1
                if not self.is_mothership:
                    self.position_manager.delete_position(position)

        # Updates happen next
        # First close out contradicting positions that happen if a validator is left in a bad state
        #for position, sync_status in position_to_sync_status.items():
        #    if sync_status == PositionSyncResult.UPDATED or sync_status == PositionSyncResult.NOTHING:
        #        if not self.is_mothership:
        #            if position.is_closed_position:
        #                self.position_manager.delete_open_position_if_exists(position)
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.UPDATED:
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)
                        self.position_manager.overwrite_position_on_disk(p)
                kept_and_matched -= 1
        # Insertions happen last so that there is no double open position issue
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.INSERTED:
                inserted -= 1
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)
                        self.position_manager.overwrite_position_on_disk(p)
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.NOTHING:
                kept_and_matched -= 1
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)

        if kept_and_matched != 0:
            raise PositionSyncResultException(f"kept_and_matched: {kept_and_matched} stats {stats}")
        if deleted != 0:
            raise PositionSyncResultException(f"deleted: {deleted} stats {stats}")
        if inserted != 0:
            raise PositionSyncResultException(f"inserted: {inserted} stats {stats}")

    def close_older_open_position(self, p1: Position, p2: Position):
        """
        p1 and p2 are both open positions for a hotkey+trade pair, so we want to close the older one.
        """
        if p2 is None:
            return p1

        self.global_stats['n_positions_closed_duplicate_opens_for_trade_pair'] += 1

        # if p2 is an older position, we close it and return p1 as the newest open position.
        if p2.open_ms < p1.open_ms:
            p2.close_out_position(TimeUtil.now_in_millis())
            self.position_manager.overwrite_position_on_disk(p2)
            return p1
        else:
            p1.close_out_position(TimeUtil.now_in_millis())
            self.position_manager.overwrite_position_on_disk(p1)
            return p2


    def debug_print_pos(self, p):
        print(f'    pos: open {TimeUtil.millis_to_formatted_date_str(p.open_ms)} close {TimeUtil.millis_to_formatted_date_str(p.close_ms) if p.close_ms else "N/A"} uuid {p.position_uuid}')
        for o in p.orders:
            self.debug_print_order(o)
    def debug_print_order(self, o):
        print(f'        order: type {o.order_type} lev {o.leverage} time {TimeUtil.millis_to_formatted_date_str(o.processed_ms)} uuid {o.order_uuid}')

    def positions_aligned(self, p1, p2, timebound_ms=None, validate_num_orders=False):
        p1_initial_position_type = p1.orders[0].order_type
        p2_initial_position_type = p2.orders[0].order_type
        if validate_num_orders and len(p1.orders) != len(p2.orders):
            return False
        if p1_initial_position_type != p2_initial_position_type:
            return False
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        if p1.is_closed_position and p2.is_closed_position:
            return abs(p1.open_ms - p2.open_ms) < timebound_ms and abs(p1.close_ms - p2.close_ms) < timebound_ms
        elif p1.is_open_position and p2.is_open_position:
            return abs(p1.open_ms - p2.open_ms) < timebound_ms
        return False

    def dict_positions_aligned(self, p1, p2, timebound_ms=None, validate_num_orders=False):
        p1_initial_position_type = p1["orders"][0]["order_type"]
        p2_initial_position_type = p2["orders"][0]["order_type"]
        if validate_num_orders and len(p1["orders"]) != len(p2["orders"]):
            return False
        if p1_initial_position_type != p2_initial_position_type:
            return False
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        if p1["is_closed_position"] and p2["is_closed_position"]:
            return abs(p1["open_ms"] - p2["open_ms"]) < timebound_ms and abs(p1["close_ms"] - p2["close_ms"]) < timebound_ms
        elif not p1["is_closed_position"] and not p2["is_closed_position"]:
            return abs(p1["open_ms"] - p2["open_ms"]) < timebound_ms
        return False

    def positions_aligned_strict(self, p1, p2):
        timebound_ms = 1000 * 10
        return (self.positions_aligned(p1, p2, timebound_ms=timebound_ms) and
                self.all_orders_aligned(p1.orders, p2.orders, timebound_ms=timebound_ms))

    def all_orders_aligned(self, orders1, orders2, timebound_ms=None):
        if len(orders1) != len(orders2):
            return False
        for o1, o2 in zip(orders1, orders2):
            if not self.orders_aligned(o1, o2, timebound_ms=timebound_ms):
                return False
        return True

    def positions_order_aligned(self, p1, p2):
        for o1 in p1["orders"]:
            for o2 in p2["orders"]:
                if self.dict_orders_aligned(o1, o2):
                    return True
        return False

    def orders_aligned(self, o1, o2, timebound_ms=None):
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        return ((o1.leverage == o2.leverage) and
                (o1.order_type == o2.order_type) and
                abs(o1.processed_ms - o2.processed_ms) < timebound_ms)

    def dict_orders_aligned(self, o1, o2, timebound_ms=None):
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        return (o1["order_uuid"] == o2["order_uuid"] or
                ((o1["leverage"] == o2["leverage"]) and
                 (o1["order_type"] == o2["order_type"]) and
                 abs(o1["processed_ms"] - o2["processed_ms"]) < timebound_ms))

    def sync_orders(self, ep, cp, hk, trade_pair, hard_snap_cutoff_ms):
        debug = 1
        existing_orders = ep.orders
        candidate_orders = cp.orders
        min_timestamp_of_order_change = float('inf')
        # Positions are synonymous with an order
        assert existing_orders, existing_orders
        assert candidate_orders, candidate_orders

        ret = []
        matched_candidates_by_uuid = set()
        matched_existing_by_uuid = set()
        kept = list()
        matched = list()
        inserted = list()
        stats = defaultdict(int)
        # First pass. Try to match 1:1 based on uuid
        for c in candidate_orders:
            if c.order_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_orders:
                if e.order_uuid in matched_existing_by_uuid:
                    continue
                if e.order_uuid == c.order_uuid:
                    ret.append(e)
                    matched_candidates_by_uuid |= {c.order_uuid}
                    matched_existing_by_uuid |= {e.order_uuid}
                    stats['matched'] += 1
                    matched.append(e)
                    break

        # Second pass. Try to match 1:1 based on timestamps, leverage, and order type
        for c in candidate_orders:
            if c.order_uuid in matched_candidates_by_uuid:  # already matched
                continue
            for e in existing_orders:
                if e.order_uuid in matched_existing_by_uuid:
                    continue
                if self.orders_aligned(e, c):
                    matched_candidates_by_uuid |= {c.order_uuid}
                    matched_existing_by_uuid |= {e.order_uuid}
                    ret.append(e)
                    stats['matched'] += 1
                    matched.append(e)
                    break

        # Handle insertions (unmatched candidates)
        for o in candidate_orders:
            if o.order_uuid in matched_candidates_by_uuid:
                continue
            ret.append(o)
            inserted.append(o)
            stats['inserted'] += 1
            min_timestamp_of_order_change = min(min_timestamp_of_order_change, o.processed_ms)

        # Handle unmatched existing orders. Delete if they occurred during the window of the candidate data.
        deleted = []
        for o in existing_orders:
            if o.order_uuid in matched_existing_by_uuid:
                continue
            if o.processed_ms < hard_snap_cutoff_ms:
                stats['deleted'] += 1
                deleted.append(o)
                min_timestamp_of_order_change = min(min_timestamp_of_order_change, o.processed_ms)
            else:
                ret.append(o)
                kept.append(o)
                stats['kept'] += 1

        if stats['deleted']:
            self.global_stats['orders_deleted'] += stats['deleted']
            self.miners_with_order_deletion.add(hk)
        if stats['inserted']:
            self.global_stats['orders_inserted'] += stats['inserted']
            self.miners_with_order_insertion.add(hk)
        if stats['matched']:
            self.global_stats['orders_matched'] += stats['matched']
            self.miners_with_order_matched.add(hk)
        if stats['kept']:
            self.global_stats['orders_kept'] += stats['kept']
            self.miners_with_order_kept.add(hk)

        any_changes = stats['inserted'] + stats['deleted']
        if debug and any_changes:
            print(f'hk {hk} trade pair {trade_pair.trade_pair} - Found {len(candidate_orders)} candidates and'
                  f' {len(existing_orders)} existing orders. stats {stats} min_timestamp_of_order_change {min_timestamp_of_order_change}')

            print('  existing:')
            for x in existing_orders:
                self.debug_print_order(x)
            print('  candidate:')
            for x in candidate_orders:
                self.debug_print_order(x)
            if inserted:
                print('  inserted:')
                for x in inserted:
                    self.debug_print_order(x)
            if deleted:
                print('  deleted:')
                for x in deleted:
                    self.debug_print_order(x)
            if kept:
                print('  kept:')
                for x in kept:
                    self.debug_print_order(x)
            if matched:
                print('  matched:')
                print(f'  {len(matched)} matched orders')
                #for x in matched:
                #    self.debug_print_order(x)

        ans = ret
        ans.sort(key=lambda x: x.processed_ms)
        return ans, min_timestamp_of_order_change

    def resolve_positions(self, candidate_positions, existing_positions, trade_pair, hk, hard_snap_cutoff_ms):
        debug = 1
        min_timestamp_of_change = float('inf')  # If this stays as float('inf), no changes happened
        candidate_positions_original = deepcopy(candidate_positions)
        existing_positions_original = deepcopy(existing_positions)
        ret = []
        matched_candidates_by_uuid = set()
        matched_existing_by_uuid = set()
        # Map position_uuid to syncStatus
        position_to_sync_status = {}
        kept = list()
        inserted = list()
        deleted = list()
        matched = list()
        stats = defaultdict(int)

        # There should only be one open position at a time. We trust the candidate data to be correct. Therefore, if
        # there is an open position in the candidate data, we will delete all open positions in the existing data.

        # open_postition_acked = False
        # First pass. Try to match 1:1 based on position_uuid
        for c in candidate_positions:
            if c.position_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_positions:
                if e.position_uuid in matched_existing_by_uuid:
                    continue
                if e.position_uuid == c.position_uuid:
                    # Block the match
                    # if open_postition_acked and e.is_open_position:
                    #     continue

                    e.orders, min_timestamp_of_order_change = self.sync_orders(e, c, hk, trade_pair, hard_snap_cutoff_ms)
                    if min_timestamp_of_order_change != float('inf'):
                        e.rebuild_position_with_updated_orders()
                        min_timestamp_of_change = min(min_timestamp_of_change, min_timestamp_of_order_change)
                        position_to_sync_status[e] = PositionSyncResult.UPDATED
                    else:
                        position_to_sync_status[e] = PositionSyncResult.NOTHING
                    # open_postition_acked |= e.is_open_position
                    ret.append(e)

                    matched_candidates_by_uuid |= {c.position_uuid}
                    matched_existing_by_uuid |= {e.position_uuid}
                    stats['matched'] += 1
                    matched.append(e)
                    break

        # Second pass. Try to match 1:1 based on timestamps
        for c in candidate_positions:
            if c.position_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_positions:
                if e.position_uuid in matched_existing_by_uuid:
                    continue
                if self.positions_aligned(e, c):
                    # Block the match
                    # if open_postition_acked and e.is_open_position:
                    #     continue

                    e.orders, min_timestamp_of_order_change = self.sync_orders(e, c, hk, trade_pair, hard_snap_cutoff_ms)
                    if min_timestamp_of_order_change != float('inf'):
                        e.rebuild_position_with_updated_orders()
                        min_timestamp_of_change = min(min_timestamp_of_change, min_timestamp_of_order_change)
                        position_to_sync_status[e] = PositionSyncResult.UPDATED
                    else:
                        position_to_sync_status[e] = PositionSyncResult.NOTHING
                    # open_postition_acked |= e.is_open_position
                    matched_candidates_by_uuid |= {c.position_uuid}
                    matched_existing_by_uuid |= {e.position_uuid}
                    ret.append(e)
                    matched.append(e)
                    stats['matched'] += 1
                    break


        # Handle insertions (unmatched candidates).
        for p in candidate_positions:
            if p.position_uuid in matched_candidates_by_uuid:
                continue

            # Block the insert
            # if open_postition_acked and p.is_open_position:
            #     self.global_stats['blocked_insert_open_position_acked'] += 1
            #     continue

            stats['inserted'] += 1
            position_to_sync_status[p] = PositionSyncResult.INSERTED
            # open_postition_acked |= p.is_open_position
            min_timestamp_of_change = min(min_timestamp_of_change, p.open_ms)
            ret.append(p)
            inserted.append(p)

        # Handle unmatched existing positions. Delete if they occurred during the window of the candidate data.
        for p in existing_positions:
            if p.position_uuid in matched_existing_by_uuid:
                continue
            if p.open_ms < hard_snap_cutoff_ms:
                stats['deleted'] += 1
                deleted.append(p)
                position_to_sync_status[p] = PositionSyncResult.DELETED
                min_timestamp_of_change = min(min_timestamp_of_change, p.open_ms)
            else:
                # Block the keep and delete it
                # if open_postition_acked and p.is_open_position:
                #     stats['deleted'] += 1
                #     deleted.append(p)
                #     position_to_sync_status[p] = PositionSyncResult.DELETED
                #     min_timestamp_of_change = min(min_timestamp_of_change, p.open_ms)
                #     self.global_stats['blocked_keep_open_position_acked'] += 1
                #     continue

                ret.append(p)
                kept.append(p)
                # open_postition_acked |= p.is_open_position
                stats['kept'] += 1
                position_to_sync_status[p] = PositionSyncResult.NOTHING


        if stats['deleted']:
            self.global_stats['positions_deleted'] += stats['deleted']
            self.miners_with_position_deletion.add(hk)
        if stats['inserted']:
            self.global_stats['positions_inserted'] += stats['inserted']
            self.miners_with_position_insertion.add(hk)
        if stats['matched']:
            self.global_stats['positions_matched'] += stats['matched']
            self.miners_with_position_matched.add(hk)
        if stats['kept']:
            self.global_stats['positions_kept'] += stats['kept']
            self.miners_with_position_kept.add(hk)


        if debug and (stats['inserted'] or stats['deleted']):
            print(f'hk {hk} trade pair {trade_pair.trade_pair} - Found {len(candidate_positions)} candidates and'
                  f' {len(existing_positions)} existing positions. stats {stats}')

            print('  existing positions:')
            for x in existing_positions_original:
                self.debug_print_pos(x)
            print('  candidate positions:')
            for x in candidate_positions_original:
                self.debug_print_pos(x)
            if inserted:
                print('  inserted positions:')
                for x in inserted:
                    self.debug_print_pos(x)
            if deleted:
                print('  deleted positions:')
                for x in deleted:
                    self.debug_print_pos(x)
            if matched:
                print('  matched positions:')
                #print(f'  {len(matched)} matched positions')
                for x in matched:
                    self.debug_print_pos(x)
            if kept:
                print('  kept positions:')
                for x in kept:
                    self.debug_print_pos(x)

        # n_open = len([p for p in ret if p.is_open_position])
        # assert n_open < 2, f"n_open: {n_open}"
        return position_to_sync_status, min_timestamp_of_change, stats

    def partition_positions_by_trade_pair(self, positions: list[Position]) -> dict[str, list[Position]]:
        positions_by_trade_pair = defaultdict(list)
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(deepcopy(position))
        return positions_by_trade_pair

    def split_position_on_flat(self, position: Position) -> list[Position]:
        """
        Takes a position, iterates through the orders, and splits the position into multiple positions
        separated by the FLAT orders in the original position.
        If there are no FLAT orders or the FLAT is at the end, no splitting is done.
        """
        positions = [position]
        all_orders = []
        orders_for_sub_position = []

        for order in position.orders:
            orders_for_sub_position.append(order)

            if order.order_type == OrderType.FLAT:
                all_orders.append(orders_for_sub_position)
                orders_for_sub_position = []

        # last set of orders not ending in FLAT
        if orders_for_sub_position:
            all_orders.append(orders_for_sub_position)

        # return original position if there is no FLAT, or FLAT is last order
        if len(all_orders) == 1:
            return positions

        # truncate orders in original position
        position.orders = all_orders[0]
        position.rebuild_position_with_updated_orders()

        # create new positions for each set of remaining orders
        for position_orders in all_orders[1:]:
            new_position = Position(miner_hotkey=position.miner_hotkey,
                                    position_uuid=position_orders[0].order_uuid,
                                    open_ms=0,
                                    trade_pair=position.trade_pair,
                                    orders=position_orders)
            new_position.rebuild_position_with_updated_orders()
            positions.append(new_position)
        self.global_stats['n_positions_spawned_from_post_flat_orders'] += len(positions) - 1
        return positions
