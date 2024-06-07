import argparse
import io
import json
import os.path
import shutil
import time
import zipfile

import traceback
from collections import defaultdict
from datetime import datetime

import requests

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import bittensor as bt

from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from google.cloud import storage

DEBUG = 0

def backup_validation_directory():
    dir_to_backup = ValiBkpUtils.get_vali_dir()
    # Write to the backup location. Make sure it is a function of the date. No dashes. Days and months get 2 digits.
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_location = ValiBkpUtils.get_vali_bkp_dir() + date_str + '/'
    # Sync directory to the backup location using python shutil
    shutil.copytree(dir_to_backup, backup_location)
    bt.logging.info(f"backed up {dir_to_backup} to {backup_location}")


def force_validator_to_restore_from_checkpoint(validator_hotkey, metagraph, config, secrets):
    try:
        time_ms = TimeUtil.now_in_millis()
        if time_ms > 1716644087000 + 1000 * 60 * 60 * 2:  # Only perform under a targeted time as checkpoint goes stale quickly.
            return

        if "mothership" in secrets:
            bt.logging.warning(f"Validator {validator_hotkey} is the mothership. Not forcing restore.")
            return

        #if config.subtensor.network == "test":  # Only need do this in mainnet
        #    bt.logging.warning("Not forcing validator to restore from checkpoint in testnet.")
        #    return

        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in metagraph.neurons}
        my_trust = hotkey_to_v_trust.get(validator_hotkey)
        if my_trust is None:
            bt.logging.warning(f"Validator {validator_hotkey} not found in metagraph. Cannot determine trust.")
            return

        # Good enough
        #if my_trust > 0.5:
        #    return

        bt.logging.warning(f"Validator {validator_hotkey} trust is {my_trust}. Forcing restore.")
        regenerate_miner_positions(perform_backup=True, backup_from_data_dir=True, ignore_timestamp_checks=True)
        bt.logging.info('Successfully forced validator to restore from checkpoint.')

    except Exception as e:
        bt.logging.error(f"Error forcing validator to restore from checkpoint: {e}")
        bt.logging.error(traceback.format_exc())




class PositionSyncer:
    def __init__(self):
        self.SYNC_LOOK_AROUND_MS = 1000 * 60 * 5
        self.position_manager = PositionManager()
        self.position_manager.init_cache_files()
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

    def debug_print_pos(self, p):
        print(f'    pos: open {TimeUtil.millis_to_formatted_date_str(p.open_ms)} close {TimeUtil.millis_to_formatted_date_str(p.close_ms) if p.close_ms else "N/A"}')
        for o in p.orders:
            self.debug_print_order(o)
    def debug_print_order(self, o):
        print( f'        order: type {o.order_type} lev {o.leverage} time {TimeUtil.millis_to_formatted_date_str(o.processed_ms)}')

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

    def sync_orders(self, e, c, hk, trade_pair):
        debug = 1
        existing_orders = e.orders
        candidate_orders = c.orders
        # Positions are synonymous with an order
        assert existing_orders, existing_orders
        assert candidate_orders, candidate_orders

        ret = []
        taken = set()
        e_kept = list()
        e_matched = list()
        if debug:
            c_kept = list()
            c_inserted = list()
        stats = defaultdict(int)
        # First pass. Try to match 1:1 based on uuid
        for e in existing_orders:
            for c in candidate_orders:
                if e.order_uuid == c.order_uuid:
                    ret.append(e)
                    taken |= {e.order_uuid}
                    stats['matched'] += 1
                    e_matched.append(e)
                    break

        # Second pass. Try to match 1:1 based on timestamps
        for e in existing_orders:
            if e.order_uuid in taken:  # already matched
                continue
            match_found = False
            for c in candidate_orders:
                if c.order_uuid in taken:
                    continue
                if self.orders_aligned(e, c):
                    match_found = True
                    taken |= {c.order_uuid}
                    ret.append(e)
                    stats['matched'] += 1
                    e_matched.append(e)
                    break

            if not match_found:
                ret.append(e)
                stats['kept'] += 1

        # Handle insertions (unmatched candidates)
        for o in candidate_orders:
            if o.order_uuid in taken:
                continue
            ret.append(o)
            if debug:
                c_inserted.append(o)
            stats['inserted'] += 1

        if stats['deleted']: # Not possible as of right now.
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


        e_deleted = [] #[e for e in existing_orders if e not in e_kept]
        if debug and (stats['inserted'] or stats['deleted']):
            print(f'hk {hk} trade pair {trade_pair.trade_pair} - Found {len(candidate_orders)} candidates and'
                  f' {len(existing_orders)} existing positions. stats {stats}')

            print(f'  existing:')
            for x in existing_orders:
                self.debug_print_order(x)
            print(f'  candidate:')
            for x in candidate_orders:
                self.debug_print_order(x)
            if c_inserted:
                print(f'  inserted:')
                for x in c_inserted:
                    self.debug_print_order(x)
            if e_deleted:
                print(f'  deleted:')
                for x in e_deleted:
                    self.debug_print_order(x)
            if 0:#e_kept:
                print(f'  kept (e/c):')
                for e, c in zip(e_kept, c_kept):
                    self.debug_print_order(e)
                    self.debug_print_order(c)

        # TODO: remove items from e_deleted that have a match in ret or to_add_back
        ans = ret
        ans.sort(key=lambda x: x.processed_ms)
        any_changes = stats['inserted']# + stats['deleted']
        return ans, any_changes

    # Handle the case where a position exists in "to_add_back" and corresponds to something in candidate_positions
    def resolve_positions(self, candidate_positions, existing_positions, trade_pair, hk):
        # TODO actually write positions.
        debug = 0

        # Since candidates come with a lag, this could be expected.
        if not candidate_positions:
            return existing_positions
        if not existing_positions:
            return candidate_positions

        ret = []
        taken = set()
        e_kept = list()
        if debug:
            c_kept = list()
            c_inserted = list()
            deleted = list()
        stats = defaultdict(int)
        # First pass. Try to match 1:1 based on position_uuid
        for e in existing_positions:
            for c in candidate_positions:
                if e.position_uuid == c.position_uuid:
                    e.orders, any_changes = self.sync_orders(e, c, hk, trade_pair)
                    if any_changes:
                        e.rebuild_position_with_updated_orders()
                    ret.append(e)
                    taken |= {e.position_uuid}
                    stats['matched'] += 1
                    e_kept.append(e)
                    break

        # Second pass. Try to match 1:1 based on timestamps
        unmatched = []
        for e in existing_positions:
            if e.position_uuid in taken:  # already matched
                continue
            match_found = False
            for c in candidate_positions:
                if c.position_uuid in taken:
                    continue
                if self.positions_aligned(e, c):
                    e.orders, any_changes = self.sync_orders(e, c, hk, trade_pair)
                    if any_changes:
                        e.rebuild_position_with_updated_orders()
                    match_found = True
                    taken |= {c.position_uuid}
                    ret.append(e)
                    stats['matched'] += 1
                    e_kept.append(e)
                    if debug:
                        c_kept.append(c)
                    break

            if not match_found:
                unmatched.append(e)


       # Handle unmatched existing positions (possibly fragmented)
        valid_order_uuids = set()
        for p in candidate_positions:
            for o in p.orders:
                valid_order_uuids.add(o.order_uuid)

        for p in unmatched:
            is_fragmented = any([o.order_uuid not in valid_order_uuids for o in p.orders])
            if is_fragmented:
                stats['deleted'] += 1
                if debug:
                    deleted.append(p)
            else:
                stats['kept'] += 1
                ret.append(p)
                if debug:
                    c_kept.append(p)


        # Handle insertions (unmatched candidates).
        for p in candidate_positions:
            if p.position_uuid in taken:
                continue

            stats['inserted'] += 1
            ret.append(p)
            if debug:
                c_inserted.append(p)


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
            if c_inserted:
                print(f'  inserted positions:')
                for x in c_inserted:
                    self.debug_print_pos(x)
            if deleted:
                print(f'  deleted positions:')
                for x in deleted:
                    self.debug_print_pos(x)

        ans = ret
        ans.sort(key=self.position_manager.sort_by_close_ms)
        n_open = len([p for p in ans if p.is_open_position])
        assert n_open < 2, f"n_open: {n_open}"
        return ans

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
            bt.logging.info(f"Automated restore. Found backup creation time: {TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)}")

        if 'mothership' in ValiUtils.get_secrets():
            bt.logging.info(f"Mothership detected. Skipping position sync.")
            return

        if disk_positions is None:
            disk_positions = self.position_manager.get_all_disk_positions_for_all_miners(only_open_positions=False,
                                                                                sort_positions=True)

        # Only add positions/orders to the disk positions. never delete from disk positions.
        # Step 1. segment positions by trade pair
        # Step 2. identify positions that need to be inserted
        # Step 3. identify positions that need to be updated. try to match on uuid and fall back to leverage+order_type and timestamp (~9 s range).


        eliminations = candidate_data['eliminations']
        eliminated_hotkeys = set([e['hotkey'] for e in eliminations])
        for hotkey, json_positions in candidate_data['positions'].items():
            if hotkey in eliminated_hotkeys:
                self.global_stats['n_miners_skipped_eliminated'] += 1
                continue
            self.global_stats['n_miners_synced'] += 1
            positions = [Position(**json_positions_dict) for json_positions_dict in json_positions['positions']]
            positions.sort(key=self.position_manager.sort_by_close_ms)
            candidate_positions_by_trade_pair = self.partition_positions_by_trade_pair(positions)
            existing_positions_by_trade_pair = self.partition_positions_by_trade_pair(disk_positions.get(hotkey, []))
            for trade_pair in existing_positions_by_trade_pair.keys():
                candidate_positions = candidate_positions_by_trade_pair.get(trade_pair, [])
                existing_positions = existing_positions_by_trade_pair[trade_pair]
                # TODO: handle orders. dont need to return anything. just write.
                synced_positions = self.resolve_positions(candidate_positions, existing_positions, trade_pair, hotkey)
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


def regenerate_miner_positions(perform_backup=True, backup_from_data_dir=False, ignore_timestamp_checks=False):
    backup_file_path = ValiBkpUtils.get_backup_file_path(use_data_dir=backup_from_data_dir)
    try:
        data = json.loads(ValiBkpUtils.get_file(backup_file_path))
        if isinstance(data, str):
            data = json.loads(data)
    except Exception as e:
        bt.logging.error(f"Unable to read validator checkpoint file. {e}")
        return False

    bt.logging.info(f"Found validator backup file with the following attributes:")
    # Log every key and value apir in the data except for positions, eliminations, and plagiarism scores
    for key, value in data.items():
        # Check is the value is of type dict or list. If so, print the size of the dict or list
        if isinstance(value, dict) or isinstance(value, list):
            # Log the size of the positions, eliminations, and plagiarism scores
            bt.logging.info(f"    {key}: {len(value)} entries")
        else:
            bt.logging.info(f"    {key}: {value}")
    backup_creation_time_ms = data['created_timestamp_ms']

    if DEBUG:
        from vali_objects.utils.live_price_fetcher import LivePriceFetcher
        secrets = ValiUtils.get_secrets()

        live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        position_manager = PositionManager(live_price_fetcher=live_price_fetcher, perform_price_adjustment=False,
                                           perform_order_corrections=True, generate_correction_templates=False,
                                           apply_corrections_template=False, perform_fee_structure_update=False)
        #position_manager.perform_price_recalibration(time_per_batch_s=10000000)
    else:
        position_manager = PositionManager()
        position_manager.init_cache_files()

    challengeperiod_manager = ChallengePeriodManager(config=None, metagraph=None)
    # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
    # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup smallest timestamp is
    # older than the local smallest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
    smallest_disk_ms, largest_disk_ms = (
        position_manager.get_extreme_position_order_processed_on_disk_ms())
    smallest_backup_ms = data['youngest_order_processed_ms']
    largest_backup_ms = data['oldest_order_processed_ms']
    try:
        formatted_backup_creation_time = TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)
        formatted_disk_date_largest = TimeUtil.millis_to_formatted_date_str(largest_disk_ms)
        formatted_backup_date_largest = TimeUtil.millis_to_formatted_date_str(largest_backup_ms)
        formatted_disk_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_disk_ms)
        formatted_backup_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_backup_ms)
    except:
        formatted_backup_creation_time = backup_creation_time_ms
        formatted_disk_date_largest = largest_disk_ms
        formatted_backup_date_largest = largest_backup_ms
        formatted_disk_date_smallest = smallest_disk_ms
        formatted_backup_date_smallest = smallest_backup_ms

    bt.logging.info(f"Timestamp analysis of backup vs disk (UTC):")
    bt.logging.info(f"    backup_creation_time: {formatted_backup_creation_time}")
    bt.logging.info(f"    smallest_disk_order_timestamp: {formatted_disk_date_smallest}")
    bt.logging.info(f"    smallest_backup_order_timestamp: {formatted_backup_date_smallest}")
    bt.logging.info(f"    oldest_disk_order_timestamp: {formatted_disk_date_largest}")
    bt.logging.info(f"    oldest_backup_order_timestamp: {formatted_backup_date_largest}")

    if ignore_timestamp_checks:
        bt.logging.info('Forcing validator restore no timestamp checks from backup_file_path: ' + backup_file_path)
        pass
    elif smallest_disk_ms >= smallest_backup_ms and largest_disk_ms <= backup_creation_time_ms:
        pass  # Ready for update!
    elif largest_disk_ms > backup_creation_time_ms:
        bt.logging.error(f"Please re-pull the backup file before restoring. Backup {formatted_backup_creation_time} appears to be older than the disk {formatted_disk_date_largest}.")
        return False
    elif smallest_disk_ms < smallest_backup_ms:
        #bt.logging.error("Your local filesystem has older orders than the backup. Please reach out to the team ASAP before regenerating. You may be holding irrecoverable data!")
        #return False
        pass  # Deregistered miners can trip this check. We will allow the regeneration to proceed.
    else:
        bt.logging.error("Problem with backup file detected. Please reach out to the team ASAP")
        return False


    n_existing_position = position_manager.get_number_of_miners_with_any_positions()
    n_existing_eliminations = position_manager.get_number_of_eliminations()
    msg = (f"Detected {n_existing_position} hotkeys with positions, {n_existing_eliminations} eliminations")
    bt.logging.info(msg)

    bt.logging.info("Backing up and overwriting all existing positions, eliminations, and plagiarism scores.")
    if perform_backup:
        backup_validation_directory()

    bt.logging.info(f"regenerating {len(data['positions'].keys())} hotkeys")
    position_manager.clear_all_miner_positions_from_disk()
    for hotkey, json_positions in data['positions'].items():
        # sort positions by close_ms otherwise, writing a closed position after an open position for the same
        # trade pair will delete the open position
        positions = [Position(**json_positions_dict) for json_positions_dict in json_positions['positions']]
        assert len(positions) > 0, f"no positions for hotkey {hotkey}"
        positions.sort(key=position_manager.sort_by_close_ms)
        ValiBkpUtils.make_dir(ValiBkpUtils.get_miner_all_positions_dir(hotkey))
        for p_obj in positions:
            #bt.logging.info(f'creating position {p_obj}')
            position_manager.save_miner_position_to_disk(p_obj)

        # Validate that the positions were written correctly
        disk_positions = position_manager.get_all_miner_positions(hotkey, sort_positions=True)
        #bt.logging.info(f'disk_positions: {disk_positions}, positions: {positions}')
        n_disk_positions = len(disk_positions)
        n_memory_positions = len(positions)
        memory_p_uuids = set([p.position_uuid for p in positions])
        disk_p_uuids = set([p.position_uuid for p in disk_positions])
        assert n_disk_positions == n_memory_positions, f"n_disk_positions: {n_disk_positions}, n_memory_positions: {n_memory_positions}"
        assert memory_p_uuids == disk_p_uuids, f"memory_p_uuids: {memory_p_uuids}, disk_p_uuids: {disk_p_uuids}"


    bt.logging.info(f"regenerating {len(data['eliminations'])} eliminations")
    position_manager.write_eliminations_to_disk(data['eliminations'])

    perf_ledgers = data.get('perf_ledgers', {})
    bt.logging.info(f"regenerating {len(perf_ledgers)} perf ledgers")
    PerfLedgerManager.save_perf_ledgers_to_disk(perf_ledgers)

    ## Now sync challenge period with the disk
    challengeperiod = data.get('challengeperiod', {})
    challengeperiod_manager.challengeperiod_testing = challengeperiod.get('testing', {})  
    challengeperiod_manager.challengeperiod_success = challengeperiod.get('success', {})

    challengeperiod_manager._write_challengeperiod_from_memory_to_disk()
    return True

if __name__ == "__main__":
    #position_syncer = PositionSyncer()
    #position_syncer.sync_positions()
    #assert 0
    t0 = time.time()
    # Check commandline arg "disable_backup" to disable backup.
    parser = argparse.ArgumentParser(description="Regenerate miner positions with optional backup disabling.")
    # Add disable_backup argument, default is 0 (False), change type to int
    parser.add_argument('--backup', type=int, default=1,
                        help='Set to 0 to disable backup during regeneration process.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Use the disable_backup argument to control backup
    perform_backup = bool(args.backup)
    bt.logging.info("regenerating miner positions")
    if not perform_backup:
        bt.logging.warning("backup disabled")
    passed = regenerate_miner_positions(perform_backup)
    if passed:
        bt.logging.info("regeneration complete in %.2f seconds" % (time.time() - t0))
    else:
        bt.logging.error("regeneration failed")
