import io
import json
import time
import zipfile

from collections import defaultdict

import requests

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
import bittensor as bt

from vali_objects.utils.vali_utils import ValiUtils
AUTO_SYNC_ORDER_LAG_MS = 1000 * 60 * 60 * 24


class PositionSyncer:
    def __init__(self, shutdown_dict=None):
        self.SYNC_LOOK_AROUND_MS = 1000 * 60 * 3
        self.position_manager = PositionManager()
        self.position_manager.init_cache_files()
        self.shutdown_dict = shutdown_dict
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
        self.perf_ledger_hks_to_invalidate = {}  # {hk: timestamp_ms}

    def debug_print_pos(self, p):
        print(f'    pos: open {TimeUtil.millis_to_formatted_date_str(p.open_ms)} close {TimeUtil.millis_to_formatted_date_str(p.close_ms) if p.close_ms else "N/A"} uuid {p.position_uuid}')
        for o in p.orders:
            self.debug_print_order(o)
    def debug_print_order(self, o):
        print( f'        order: type {o.order_type} lev {o.leverage} time {TimeUtil.millis_to_formatted_date_str(o.processed_ms)} uuid {o.order_uuid}')

    def positions_aligned(self, p1, p2, timebound_ms=None):
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        if p1.is_closed_position and p2.is_closed_position:
            return abs(p1.open_ms - p2.open_ms) < timebound_ms and abs(p1.close_ms - p2.close_ms) < timebound_ms
        elif p1.is_open_position and p2.is_open_position:
            return abs(p1.open_ms - p2.open_ms) < timebound_ms
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

    def orders_aligned(self, o1, o2, timebound_ms=None):
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        return ((o1.leverage == o2.leverage) and
                (o1.order_type == o2.order_type) and
                abs(o1.processed_ms - o2.processed_ms) < timebound_ms)

    def sync_orders(self, e, c, hk, trade_pair, hard_snap_cutoff_ms):
        debug = 1
        existing_orders = e.orders
        candidate_orders = c.orders
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
            self.miners_with_order_matched.add(hk)


        any_changes = stats['inserted'] + stats['deleted']
        if debug and any_changes:
            print(f'hk {hk} trade pair {trade_pair.trade_pair} - Found {len(candidate_orders)} candidates and'
                  f' {len(existing_orders)} existing orders. stats {stats}')

            print(f'  existing:')
            for x in existing_orders:
                self.debug_print_order(x)
            print(f'  candidate:')
            for x in candidate_orders:
                self.debug_print_order(x)
            if inserted:
                print(f'  inserted:')
                for x in inserted:
                    self.debug_print_order(x)
            if deleted:
                print(f'  deleted:')
                for x in deleted:
                    self.debug_print_order(x)
            if kept:
                print(f'  kept:')
                for x in kept:
                    self.debug_print_order(x)

        ans = ret
        ans.sort(key=lambda x: x.processed_ms)
        return ans, min_timestamp_of_order_change

    def resolve_positions(self, candidate_positions, existing_positions, trade_pair, hk, hard_snap_cutoff_ms):
        # TODO actually write positions.
        debug = 1
        min_timestamp_of_change = float('inf')  # If this stays as float('inf), not changes happened

        # Since candidates come with a lag, this could be expected.
        if not candidate_positions:
            return existing_positions, min_timestamp_of_change
        if not existing_positions:
            return candidate_positions, min_timestamp_of_change

        ret = []
        matched_candidates_by_uuid = set()
        matched_existing_by_uuid = set()
        kept = list()
        inserted = list()
        deleted = list()
        stats = defaultdict(int)
        # First pass. Try to match 1:1 based on position_uuid
        for c in candidate_positions:
            if c.position_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_positions:
                if e.position_uuid == c.position_uuid:
                    e.orders, min_timestamp_of_order_change = self.sync_orders(e, c, hk, trade_pair, hard_snap_cutoff_ms)
                    if min_timestamp_of_order_change != float('inf'):
                        e.rebuild_position_with_updated_orders()
                        min_timestamp_of_change = min(min_timestamp_of_change, min_timestamp_of_order_change)
                    ret.append(e)
                    matched_candidates_by_uuid |= {c.position_uuid}
                    matched_existing_by_uuid |= {e.position_uuid}
                    stats['matched'] += 1
                    break

        # Second pass. Try to match 1:1 based on timestamps
        for c in candidate_positions:
            if c.position_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_positions:
                if e.position_uuid in matched_existing_by_uuid:
                    continue
                if self.positions_aligned(e, c):
                    e.orders, min_timestamp_of_order_change = self.sync_orders(e, c, hk, trade_pair, hard_snap_cutoff_ms)
                    if min_timestamp_of_order_change != float('inf'):
                        e.rebuild_position_with_updated_orders()
                        min(min_timestamp_of_change, min_timestamp_of_order_change)
                    matched_candidates_by_uuid |= {c.position_uuid}
                    matched_existing_by_uuid |= {e.position_uuid}
                    ret.append(e)
                    stats['matched'] += 1
                    break


        # Handle insertions (unmatched candidates).
        for p in candidate_positions:
            if p.position_uuid in matched_candidates_by_uuid:
                continue

            stats['inserted'] += 1
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
                min_timestamp_of_change = min(min_timestamp_of_change, p.open_ms)
            else:
                ret.append(p)
                kept.append(p)
                stats['kept'] += 1


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

            print(f'  existing positions:')
            for x in existing_positions:
                self.debug_print_pos(x)
            print(f'  candidate positions:')
            for x in candidate_positions:
                self.debug_print_pos(x)
            if inserted:
                print(f'  inserted positions:')
                for x in inserted:
                    self.debug_print_pos(x)
            if deleted:
                print(f'  deleted positions:')
                for x in deleted:
                    self.debug_print_pos(x)

        ret.sort(key=self.position_manager.sort_by_close_ms)
        n_open = len([p for p in ret if p.is_open_position])
        assert n_open < 2, f"n_open: {n_open}"
        return ret, min_timestamp_of_change

    def partition_positions_by_trade_pair(self, positions: list[Position]) -> dict[str, list[Position]]:
        positions_by_trade_pair = defaultdict(list)
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(position)
        return positions_by_trade_pair

    def read_validator_checkpoint_from_gcloud_zip(url):
        # URL of the zip file
        url = "https://storage.googleapis.com/validator_checkpoint/validator_checkpoint.zip"
        try:
            # Send HTTP GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Read the content of the zip file from the response
            with io.BytesIO(response.content) as zip_buffer:
                with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                    # Ensure there is at least one file in the zip archive
                    if zip_file.namelist():
                        # Assume the JSON file is the first file in the list
                        with zip_file.open(zip_file.namelist()[0]) as json_file:
                            # Load JSON data from the file
                            json_data = json.load(json_file)
                            return json_data
                    else:
                        bt.logging.error("No files found in the zip archive.")
                        return None
        except requests.HTTPError as e:
            bt.logging.error(f"HTTP Error: {e}")
        except zipfile.BadZipFile:
            bt.logging.error("The downloaded file is not a zip file or it is corrupted.")
        except json.JSONDecodeError:
            bt.logging.error("Error decoding JSON from the file.")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e}")
        return None


    def sync_positions(self, candidate_data=None, disk_positions=None) -> dict[str: list[Position]]:
        t0 = time.time()
        self.init_data()
        if candidate_data is None:
            candidate_data = self.read_validator_checkpoint_from_gcloud_zip()
            if not candidate_data:
                bt.logging.error("Unable to read validator checkpoint file. Sync canceled")
                return
        backup_creation_time_ms = candidate_data['created_timestamp_ms']
        bt.logging.info(f"Automated sync. Found backup creation time: {TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)}")

        candidate_hk_to_positions = {}

        # The candidate dataset is time lagged. We only delete non-matching data if they occured during the window of the candidate data.
        # We want to account for a few minutes difference of possible orders that came in after a retry.
        hard_snap_cutoff_ms = backup_creation_time_ms - AUTO_SYNC_ORDER_LAG_MS
        bt.logging.info(
            f"Automated sync. hard_snap_cutoff_ms: {TimeUtil.millis_to_formatted_date_str(hard_snap_cutoff_ms)}")

        if 'mothership' in ValiUtils.get_secrets():
            bt.logging.info(f"Mothership detected. Skipping position sync.")
            # TODO: reenable return

        if disk_positions is None:
            disk_positions = self.position_manager.get_all_disk_positions_for_all_miners(only_open_positions=False,
                                                                                sort_positions=True)

        # Only add positions/orders to the disk positions. never delete from disk positions.
        # Step 1. segment positions by trade pair
        # Step 2. identify positions that need to be inserted
        # Step 3. identify positions that need to be updated. try to match on uuid and fall back to leverage+order_type and timestamp (~9 s range).


        eliminations = candidate_data['eliminations']
        self.position_manager.write_eliminations_to_disk(eliminations)
        eliminated_hotkeys = set([e['hotkey'] for e in eliminations])

        # For a healthy validator, the existing positions will always be a superset of the candidate positions
        for hotkey, positions in candidate_hk_to_positions.items():
            if self.shutdown_dict:
                return
            if hotkey in eliminated_hotkeys:
                self.global_stats['n_miners_skipped_eliminated'] += 1
                continue
            self.global_stats['n_miners_synced'] += 1
            candidate_positions_by_trade_pair = self.partition_positions_by_trade_pair(positions)
            existing_positions_by_trade_pair = self.partition_positions_by_trade_pair(disk_positions.get(hotkey, []))
            for trade_pair in candidate_positions_by_trade_pair.keys():
                if self.shutdown_dict:
                    return
                candidate_positions = candidate_positions_by_trade_pair.get(trade_pair, [])
                existing_positions = existing_positions_by_trade_pair.get(trade_pair, [])
                # TODO: dont need to return anything. just write.
                synced_positions, min_timestamp_of_change = self.resolve_positions(candidate_positions, existing_positions, trade_pair, hotkey, hard_snap_cutoff_ms)
                if min_timestamp_of_change != float('inf'):
                    self.perf_ledger_hks_to_invalidate[hotkey] = (
                        min_timestamp_of_change) if hotkey not in self.perf_ledger_hks_to_invalidate else (
                        min(self.perf_ledger_hks_to_invalidate[hotkey], min_timestamp_of_change))
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
        bt.logging.info(f"Global stats:")
        for k, v in self.global_stats.items():
            bt.logging.info(f"  {k}: {v}")
        bt.logging.info(f"Position sync took {time.time() - t0} seconds")


if __name__ == "__main__":
    position_syncer = PositionSyncer()
    position_syncer.sync_positions()
