import time
import traceback
from copy import deepcopy
from enum import Enum
from collections import defaultdict

from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
import bittensor as bt

from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.miner_bucket_enum import MinerBucket
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
                 ipc_manager=None, enable_position_splitting = False, verbose=False, contract_manager=None,
                 live_price_fetcher=None, asset_selection_manager=None
):
        self.verbose = verbose
        self.is_mothership = 'ms' in ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        self.SYNC_LOOK_AROUND_MS = 1000 * 60 * 3
        self.enable_position_splitting = enable_position_splitting
        self.position_manager = position_manager
        self.contract_manager = contract_manager
        self.asset_selection_manager = asset_selection_manager
        self.shutdown_dict = shutdown_dict
        self.last_signal_sync_time_ms = 0
        self.signal_sync_lock = signal_sync_lock
        self.signal_sync_condition = signal_sync_condition
        self.n_orders_being_processed = n_orders_being_processed
        self.live_price_fetcher = live_price_fetcher
        if ipc_manager:
            self.perf_ledger_hks_to_invalidate = ipc_manager.dict()
        else:
            self.perf_ledger_hks_to_invalidate = {}  # {hk: timestamp_ms}
        self.init_data()

    def init_data(self):
        self.global_stats = defaultdict(int)

        # Order tracking sets (by miner)
        self.miners_with_order_deletions = set()
        self.miners_with_order_insertions = set()
        self.miners_with_order_matches = set()
        self.miners_with_order_updates = set()

        # Position tracking sets (by miner)
        self.miners_with_position_deletions = set()
        self.miners_with_position_insertions = set()
        self.miners_with_position_matches = set()
        self.miners_with_position_updates = set()
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
            # Get current eliminations before sync
            old_eliminated_hotkeys = set(x['hotkey'] for x in self.position_manager.elimination_manager.eliminations)
            
            # Sync eliminations and get removed hotkeys
            removed = self.position_manager.elimination_manager.sync_eliminations(eliminations)
            
            # Get new eliminations after sync
            new_eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
            newly_eliminated = new_eliminated_hotkeys - old_eliminated_hotkeys
            
            # Invalidate perf ledgers for both removed and newly eliminated miners
            for hk in removed:
                self.perf_ledger_hks_to_invalidate[hk] = 0
            for hk in newly_eliminated:
                self.perf_ledger_hks_to_invalidate[hk] = 0


        challengeperiod_data = candidate_data.get('challengeperiod', {})
        if challengeperiod_data:  # Only in autosync as of now.
            orig_testing_keys = set(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
            orig_success_keys = set(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP))

            challengeperiod_dict = ChallengePeriodManager.parse_checkpoint_dict(challengeperiod_data)
            new_testing_keys = {
                    hotkey for hotkey, (bucket, _, _, _) in challengeperiod_dict.items()
                    if bucket is MinerBucket.CHALLENGE
                    }
            new_success_keys = {
                    hotkey for hotkey, (bucket, _, _, _) in challengeperiod_dict.items()
                    if bucket is MinerBucket.MAINCOMP
                    }

            bt.logging.info(f"Challengeperiod testing sync keys added: {new_testing_keys-orig_testing_keys}\n"
                            f"Challengeperiod testing sync keys removed: {orig_testing_keys - new_testing_keys}\n"
                            f"Challengeperiod success sync keys added: {new_success_keys - orig_success_keys}\n"
                            f"Challengeperiod success sync keys removed: {orig_success_keys - new_success_keys}")
            if not shadow_mode:
                self.position_manager.challengeperiod_manager.sync_challenge_period_data(challengeperiod_data)

        # Sync miner account sizes if available and contract manager is present
        miner_account_sizes_data = candidate_data.get('miner_account_sizes', {})
        if miner_account_sizes_data and hasattr(self, 'contract_manager') and self.contract_manager:
            if not shadow_mode:
                bt.logging.info(f"Syncing {len(miner_account_sizes_data)} miner account size records from auto sync")
                self.contract_manager.sync_miner_account_sizes_data(miner_account_sizes_data)
        elif miner_account_sizes_data:
            bt.logging.warning("Miner account sizes data found but contract manager not available for sync")

        eliminated_hotkeys = set([e['hotkey'] for e in eliminations])
        # For a healthy validator, the existing positions will always be a superset of the candidate positions
        for hotkey, positions in candidate_hk_to_positions.items():
            if self.shutdown_dict:
                return
            if hotkey in eliminated_hotkeys:
                self.global_stats['n_miners_skipped_eliminated'] += 1
                continue
            
            # Deduplicate candidate positions BEFORE processing
            deduped_positions = self._dedupe_candidate_positions(positions)
            candidate_hk_to_positions[hotkey] = deduped_positions
            
            self.global_stats['n_miners_synced'] += 1
            candidate_positions_by_trade_pair = self.partition_positions_by_trade_pair(deduped_positions)
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
                    bt.logging.error(f"Error syncing positions for hotkey {hotkey} trade pair {trade_pair}. Error: {e} traceback: {limited_traceback}")
                    # If this is PositionSyncResultException, throw it up. Otherwise, log the error and continue.
                    if isinstance(e, PositionSyncResultException):
                        raise e
                    else:
                        self.global_stats['exceptions_seen'] += 1

        # Sync asset selections if available
        asset_selections_data = candidate_data.get('asset_selections', {})
        if asset_selections_data and self.asset_selection_manager:
            bt.logging.info(f"Syncing {len(asset_selections_data)} miner asset selections from auto sync")
            if not shadow_mode:
                bt.logging.info(f"Syncing {len(asset_selections_data)} miner asset selection records from auto sync")
                self.asset_selection_manager.sync_miner_asset_selection_data(asset_selections_data)
        elif asset_selections_data:
            bt.logging.warning("Asset selections data found but no AssetSelectionManager available for sync")

        # Reorganized stats with clear, grouped naming
        # Overview
        self.global_stats['miners_processed'] = self.global_stats['n_miners_synced']
        self.global_stats['miners_eliminated_skipped'] = self.global_stats['n_miners_skipped_eliminated']
        
        # Position outcomes (by unique miners)
        self.global_stats['miners_with_position_updates'] = len(self.miners_with_position_updates)
        self.global_stats['miners_with_position_matches'] = len(self.miners_with_position_matches)
        self.global_stats['miners_with_position_insertions'] = len(self.miners_with_position_insertions)
        self.global_stats['miners_with_position_deletions'] = len(self.miners_with_position_deletions)
        
        # Order outcomes (by unique miners)
        self.global_stats['miners_with_order_updates'] = len(self.miners_with_order_updates)
        self.global_stats['miners_with_order_matches'] = len(self.miners_with_order_matches)
        self.global_stats['miners_with_order_insertions'] = len(self.miners_with_order_insertions)
        self.global_stats['miners_with_order_deletions'] = len(self.miners_with_order_deletions)
        
        # Note: Raw counts use existing legacy keys directly (no duplication)

        # Print reorganized stats with clear grouping
        bt.logging.info("=" * 60)
        bt.logging.info("AUTOSYNC STATISTICS")
        bt.logging.info("=" * 60)
        
        # Overview section
        bt.logging.info("SYNC OVERVIEW:")
        bt.logging.info(f"  miners_processed: {self.global_stats.get('miners_processed', 0)}")
        bt.logging.info(f"  miners_eliminated_skipped: {self.global_stats.get('miners_eliminated_skipped', 0)}")
        bt.logging.info(f"  sync_duration_seconds: {time.time() - t0:.2f}")
        
        # Position outcomes
        bt.logging.info("POSITION OUTCOMES (by unique miners):")
        bt.logging.info(f"  miners_with_position_updates: {self.global_stats.get('miners_with_position_updates', 0)}")
        bt.logging.info(f"  miners_with_position_matches: {self.global_stats.get('miners_with_position_matches', 0)}")
        bt.logging.info(f"  miners_with_position_insertions: {self.global_stats.get('miners_with_position_insertions', 0)}")
        bt.logging.info(f"  miners_with_position_deletions: {self.global_stats.get('miners_with_position_deletions', 0)}")
        
        # Order outcomes
        bt.logging.info("ORDER OUTCOMES (by unique miners):")
        bt.logging.info(f"  miners_with_order_updates: {self.global_stats.get('miners_with_order_updates', 0)}")
        bt.logging.info(f"  miners_with_order_matches: {self.global_stats.get('miners_with_order_matches', 0)}")
        bt.logging.info(f"  miners_with_order_insertions: {self.global_stats.get('miners_with_order_insertions', 0)}")
        bt.logging.info(f"  miners_with_order_deletions: {self.global_stats.get('miners_with_order_deletions', 0)}")
        
        # Order-level operation counts (more actionable metrics)
        bt.logging.info("ORDER OPERATIONS (total counts):")
        bt.logging.info(f"  total_orders_inserted: {self.global_stats.get('orders_inserted', 0)}")
        bt.logging.info(f"  total_orders_updated: {self.global_stats.get('orders_updated', 0)}")
        bt.logging.info(f"  total_orders_deleted: {self.global_stats.get('orders_deleted', 0)}")
        bt.logging.info(f"  total_orders_matched: {self.global_stats.get('orders_matched', 0)}")
        
        # Raw counts (total items and miners processed)
        bt.logging.info("RAW COUNTS (total items and miners processed):")
        bt.logging.info(f"  total_positions_processed: {self.global_stats.get('positions_matched', 0)}")
        bt.logging.info(f"  total_orders_processed: {self.global_stats.get('orders_matched', 0)}")
        bt.logging.info(f"  total_miners_processed: {self.global_stats.get('miners_processed', 0)}")
        
        # Other stats (exclude keys we already displayed above)
        excluded_keys = {
            'miners_processed', 'miners_eliminated_skipped',  # Overview keys
            'n_miners_synced', 'n_miners_skipped_eliminated',  # Legacy overview 
            'positions_matched', 'orders_matched',  # Raw counts (already shown)
            'positions_deleted', 'positions_inserted', 'positions_kept',  # Raw position ops
            'orders_deleted', 'orders_inserted', 'orders_updated', 'orders_kept'  # Order ops (already shown)
        }
        other_stats = {k: v for k, v in self.global_stats.items() 
                      if not k.startswith('miners_with_') and k not in excluded_keys}
        if other_stats:
            bt.logging.info("OTHER STATS:")
            for k, v in other_stats.items():
                bt.logging.info(f"  {k}: {v}")
        
        bt.logging.info("=" * 60)

    def write_modifications(self, position_to_sync_status, stats):
        # Track position sync statuses for global stats
        for position, sync_status in position_to_sync_status.items():
            hk = position.miner_hotkey
            if sync_status == PositionSyncResult.UPDATED:
                self.miners_with_position_updates.add(hk)
            elif sync_status == PositionSyncResult.NOTHING:
                self.miners_with_position_matches.add(hk)
            elif sync_status == PositionSyncResult.INSERTED:
                self.miners_with_position_insertions.add(hk)
            elif sync_status == PositionSyncResult.DELETED:
                self.miners_with_position_deletions.add(hk)
            # KEPT status is handled elsewhere in the existing flow

        # Ensure the enums align with the global stats
        kept_and_matched = stats['kept'] + stats['matched']
        deleted = stats['deleted']
        inserted = stats['inserted']

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

        # Handle multiple open positions for a hotkey - track across ALL sync statuses to prevent duplicates
        prev_open_position = None
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
        # Do NOT reset prev_open_position - we need to track it across all sync statuses
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.INSERTED:
                inserted -= 1
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)
                        self.position_manager.overwrite_position_on_disk(p)

        # Handle NOTHING status positions
        # Do NOT reset prev_open_position - we need to track it across all sync statuses
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.NOTHING:
                kept_and_matched -= 1
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)
                        self.position_manager.overwrite_position_on_disk(p)

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
            self.miners_with_order_deletions.add(hk)
        if stats['inserted']:
            self.global_stats['orders_inserted'] += stats['inserted']
            self.miners_with_order_insertions.add(hk)
        if stats['matched']:
            self.global_stats['orders_matched'] += stats['matched']
            self.miners_with_order_matches.add(hk)
        if stats['kept']:
            self.global_stats['orders_kept'] += stats['kept']
        
        # Track order updates (when orders are modified)
        if stats.get('updated', 0) > 0:
            self.global_stats['orders_updated'] = self.global_stats.get('orders_updated', 0) + stats['updated']
            self.miners_with_order_updates.add(hk)

        any_changes = stats['inserted'] + stats['deleted']
        if self.verbose and any_changes:
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
                        e.rebuild_position_with_updated_orders(self.live_price_fetcher)
                        min_timestamp_of_change = min(min_timestamp_of_change, min_timestamp_of_order_change)
                        position_to_sync_status[e] = PositionSyncResult.UPDATED
                    else:
                        position_to_sync_status[e] = PositionSyncResult.NOTHING
                        # Check if position actually needs splitting before forcing write_modifications
                        if self.position_manager and self.position_manager._position_needs_splitting(e):
                            # Force write_modifications to be called for position splitting
                            min_timestamp_of_change = min(min_timestamp_of_change, e.open_ms)
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
                        e.rebuild_position_with_updated_orders(self.live_price_fetcher)
                        min_timestamp_of_change = min(min_timestamp_of_change, min_timestamp_of_order_change)
                        position_to_sync_status[e] = PositionSyncResult.UPDATED
                    else:
                        position_to_sync_status[e] = PositionSyncResult.NOTHING
                        # Check if position actually needs splitting before forcing write_modifications
                        if self.position_manager and self.position_manager._position_needs_splitting(e):
                            # Force write_modifications to be called for position splitting
                            min_timestamp_of_change = min(min_timestamp_of_change, e.open_ms)
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
            self.miners_with_position_deletions.add(hk)
        if stats['inserted']:
            self.global_stats['positions_inserted'] += stats['inserted']
            self.miners_with_position_insertions.add(hk)
        if stats['matched']:
            self.global_stats['positions_matched'] += stats['matched']
            self.miners_with_position_matches.add(hk)
        if stats['kept']:
            self.global_stats['positions_kept'] += stats['kept']
        
        # Track position updates (when positions are modified)
        if stats.get('updated', 0) > 0:
            self.global_stats['positions_updated'] = self.global_stats.get('positions_updated', 0) + stats['updated']
            self.miners_with_position_updates.add(hk)


        if self.verbose and (stats['inserted'] or stats['deleted']):
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

    def _dedupe_candidate_positions(self, positions: list[Position]) -> list[Position]:
        """
        Deduplicate candidate positions in memory before sync processing.
        Similar to position_manager.dedupe_positions but works on in-memory data.
        """
        if not positions:
            return positions
            
        positions_by_trade_pair = defaultdict(list)
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(position)

        deduped_positions = []
        for trade_pair, trade_pair_positions in positions_by_trade_pair.items():
            position_uuid_to_keep = {}
            
            for position in trade_pair_positions:
                if position.position_uuid in position_uuid_to_keep:
                    # Keep the position with more orders (same logic as position_manager.dedupe_positions)
                    existing_position = position_uuid_to_keep[position.position_uuid]
                    if len(position.orders) > len(existing_position.orders):
                        position_uuid_to_keep[position.position_uuid] = position
                    # If current has same or fewer orders, keep the existing one (do nothing)
                else:
                    position_uuid_to_keep[position.position_uuid] = position
            
            # Add the deduplicated positions for this trade pair
            deduped_positions.extend(position_uuid_to_keep.values())
        
        return deduped_positions

    def partition_positions_by_trade_pair(self, positions: list[Position]) -> dict[str, list[Position]]:
        positions_by_trade_pair = defaultdict(list)
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(deepcopy(position))
        return positions_by_trade_pair

    def split_position_on_flat(self, position: Position) -> list[Position]:
        """
        Delegates position splitting to the PositionManager.
        This maintains the autosync logic while using the centralized splitting implementation.
        """
        if not self.position_manager or not self.enable_position_splitting:
            # If no position manager or splitting disabled, return position as-is
            return [position]
        
        # Use the position manager's split method
        positions, split_info = self.position_manager.split_position_on_flat(position, track_stats=False)
        
        # Track statistics for autosync
        if len(positions) > 1:
            self.global_stats['n_positions_spawned_from_post_flat_orders'] += len(positions) - 1
            
            # Track implicit flat splits
            if split_info['implicit_flat_splits'] > 0:
                self.global_stats['n_positions_split_on_implicit_flat'] = \
                    self.global_stats.get('n_positions_split_on_implicit_flat', 0) + split_info['implicit_flat_splits']
        
        return positions
