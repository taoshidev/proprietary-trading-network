"""
Emissions Ledger Manager - Efficiently manages continuous emissions tracking

This module provides a manager class that runs continuously to track emissions
for all hotkeys in a subnet, using delta updates to avoid reprocessing historical data.

Key Features:
- Delta updates: Only processes new 12-hour chunks since last update
- State persistence: Saves/loads ledgers to disk for recovery
- Background loop: Runs continuously with configurable intervals
- New hotkey detection: Automatically tracks new miners joining subnet
- Efficient bulk updates: Processes all hotkeys in parallel batches

Standalone Usage:
    python -m vali_objects.vali_dataclasses.emissions_ledger_manager --netuid 8
"""
import os
import json
import gzip
import time
import threading
import traceback
from typing import Dict, List, Optional, Set
from pathlib import Path
import bittensor as bt

from vali_objects.vali_dataclasses.emissions_ledger import (
    EmissionsLedger,
    EmissionsCheckpoint
)


class EmissionsLedgerManager:
    """
    Manages continuous emissions tracking with efficient delta updates.

    This manager maintains emissions ledgers for all hotkeys in a subnet,
    updating only new chunks to minimize blockchain queries.
    """

    # Default update interval: 6 hours (half a day)
    DEFAULT_UPDATE_INTERVAL_SECONDS = 6 * 60 * 60

    # Default persistence directory
    DEFAULT_PERSISTENCE_DIR = "validation/emissions_ledgers"

    def __init__(
        self,
        network: str = "finney",
        netuid: int = 8,
        persistence_dir: Optional[str] = None,
        update_interval_seconds: int = DEFAULT_UPDATE_INTERVAL_SECONDS,
        auto_persist: bool = True,
        archive_endpoints: Optional[List[str]] = None
    ):
        """
        Initialize EmissionsLedgerManager.

        Args:
            network: Bittensor network name ("finney", "test", "local")
            netuid: Subnet UID to track (default: 8)
            persistence_dir: Directory to save/load ledgers (default: validation/emissions_ledgers)
            update_interval_seconds: How often to update ledgers (default: 6 hours)
            auto_persist: Automatically save ledgers after updates (default: True)
            archive_endpoints: List of archive node endpoints for historical queries
        """
        self.network = network
        self.netuid = netuid
        self.update_interval_seconds = update_interval_seconds
        self.auto_persist = auto_persist

        # Setup persistence directory
        if persistence_dir is None:
            persistence_dir = self.DEFAULT_PERSISTENCE_DIR
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        bt.logging.info(f"EmissionsLedgerManager initialized for network={network}, netuid={netuid}")
        bt.logging.info(f"Persistence directory: {self.persistence_dir}")
        bt.logging.info(f"Update interval: {update_interval_seconds} seconds ({update_interval_seconds/3600:.1f} hours)")

        # Initialize emissions ledger
        self.emissions_ledger = EmissionsLedger(
            network=network,
            netuid=netuid,
            archive_endpoints=archive_endpoints
        )

        # Track last update time per hotkey
        self.last_update_time: Dict[str, int] = {}  # hotkey -> timestamp_ms

        # Background thread control
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

        # Load existing state if available
        self._load_state()

    def _get_state_file_path(self) -> Path:
        """Get the path to the state file"""
        return self.persistence_dir / f"emissions_ledgers_netuid_{self.netuid}.json.gz"

    def _save_state(self):
        """Save emissions ledgers and metadata to disk"""
        try:
            state = {
                'network': self.network,
                'netuid': self.netuid,
                'last_update_time': self.last_update_time,
                'ledgers': {}
            }

            # Convert ledgers to serializable format
            for hotkey, checkpoints in self.emissions_ledger.emissions_ledgers.items():
                state['ledgers'][hotkey] = [cp.to_dict() for cp in checkpoints]

            # Save as compressed JSON
            state_file = self._get_state_file_path()
            state_json = json.dumps(state, indent=2)

            with gzip.open(state_file, 'wt', encoding='utf-8') as f:
                f.write(state_json)

            bt.logging.info(f"Saved emissions ledgers state: {len(state['ledgers'])} hotkeys to {state_file}")

        except Exception as e:
            bt.logging.error(f"Error saving emissions ledgers state: {e}")
            bt.logging.error(traceback.format_exc())

    def _load_state(self):
        """Load emissions ledgers and metadata from disk"""
        try:
            state_file = self._get_state_file_path()

            if not state_file.exists():
                bt.logging.info("No existing emissions ledgers state found, starting fresh")
                return

            bt.logging.info(f"Loading emissions ledgers state from {state_file}")

            with gzip.open(state_file, 'rt', encoding='utf-8') as f:
                state = json.load(f)

            # Validate state
            if state.get('network') != self.network:
                bt.logging.warning(f"State file network mismatch: {state.get('network')} vs {self.network}")
                return

            if state.get('netuid') != self.netuid:
                bt.logging.warning(f"State file netuid mismatch: {state.get('netuid')} vs {self.netuid}")
                return

            # Load last update times
            self.last_update_time = state.get('last_update_time', {})

            # Load ledgers
            ledgers_data = state.get('ledgers', {})
            for hotkey, checkpoints_data in ledgers_data.items():
                checkpoints = []
                for cp_dict in checkpoints_data:
                    checkpoint = EmissionsCheckpoint(
                        chunk_start_ms=cp_dict['chunk_start_ms'],
                        chunk_end_ms=cp_dict['chunk_end_ms'],
                        chunk_emissions=cp_dict['chunk_emissions'],
                        cumulative_emissions=cp_dict['cumulative_emissions'],
                        block_start=cp_dict.get('block_start'),
                        block_end=cp_dict.get('block_end')
                    )
                    checkpoints.append(checkpoint)

                self.emissions_ledger.emissions_ledgers[hotkey] = checkpoints

            bt.logging.info(f"Loaded {len(ledgers_data)} emissions ledgers from state file")

        except Exception as e:
            bt.logging.error(f"Error loading emissions ledgers state: {e}")
            bt.logging.error(traceback.format_exc())

    def get_active_hotkeys(self) -> Set[str]:
        """
        Get the set of currently active hotkeys in the subnet.

        Returns:
            Set of hotkey SS58 addresses
        """
        try:
            # Try metagraph API first
            metagraph = self.emissions_ledger.subtensor.metagraph(netuid=self.netuid)
            active_hotkeys = set(metagraph.hotkeys)
            bt.logging.info(f"Found {len(active_hotkeys)} active hotkeys in subnet {self.netuid}")
            return active_hotkeys
        except Exception as e:
            bt.logging.warning(f"Metagraph API unavailable ({e}), using direct chain queries")
            # Fallback to direct chain queries
            hotkeys_list = self.emissions_ledger._get_all_hotkeys_from_chain()
            active_hotkeys = set(hotkeys_list)
            if active_hotkeys:
                bt.logging.info(f"Found {len(active_hotkeys)} active hotkeys in subnet {self.netuid}")
            else:
                bt.logging.error("Could not retrieve active hotkeys from chain")
            return active_hotkeys

    def update_single_hotkey(self, hotkey: str, verbose: bool = False) -> bool:
        """
        Update emissions ledger for a single hotkey using delta update.

        Only processes new chunks since the last update, making this very efficient.

        Args:
            hotkey: SS58 address of the hotkey
            verbose: Enable detailed logging

        Returns:
            True if update was successful, False otherwise
        """
        try:
            current_time_ms = int(time.time() * 1000)

            # Get existing ledger for this hotkey
            existing_checkpoints = self.emissions_ledger.get_emissions_ledger(hotkey)

            # Determine start time for update
            if existing_checkpoints:
                # Delta update: start from the end of last checkpoint
                last_checkpoint = existing_checkpoints[-1]
                start_time_ms = last_checkpoint.chunk_end_ms

                if verbose:
                    bt.logging.info(f"Delta update for {hotkey}: {len(existing_checkpoints)} existing checkpoints")
                    bt.logging.info(f"Resuming from {start_time_ms}")
            else:
                # Full update: start from registration
                start_time_ms = None
                if verbose:
                    bt.logging.info(f"Full update for {hotkey}: building from registration")

            # Check if we need to update (at least one new chunk available)
            if existing_checkpoints:
                next_chunk_start, next_chunk_end = EmissionsLedger.get_chunk_boundaries(current_time_ms)

                # If we're still in the same chunk as last update, skip
                if last_checkpoint.chunk_end_ms >= next_chunk_start:
                    if verbose:
                        bt.logging.debug(f"No new chunks for {hotkey}, skipping")
                    return True

            # Build/update ledger
            new_checkpoints = self.emissions_ledger.build_emissions_ledger_for_hotkey(
                hotkey=hotkey,
                start_time_ms=start_time_ms,
                end_time_ms=current_time_ms,
                verbose=verbose
            )

            # Track update time
            self.last_update_time[hotkey] = current_time_ms

            if new_checkpoints:
                bt.logging.info(f"Updated {hotkey}: added {len(new_checkpoints) - len(existing_checkpoints)} new checkpoints")

            return True

        except Exception as e:
            bt.logging.error(f"Error updating emissions for {hotkey}: {e}")
            if verbose:
                bt.logging.error(traceback.format_exc())
            return False

    def update_all_hotkeys(self, verbose: bool = False) -> Dict[str, bool]:
        """
        Update emissions ledgers for all active hotkeys in the subnet.

        Uses delta updates for existing hotkeys and full updates for new ones.

        Args:
            verbose: Enable detailed logging

        Returns:
            Dictionary mapping hotkeys to update success status
        """
        bt.logging.info("Starting bulk emissions ledger update")
        start_time = time.time()

        # Get current active hotkeys
        active_hotkeys = self.get_active_hotkeys()

        if not active_hotkeys:
            bt.logging.warning("No active hotkeys found in subnet")
            return {}

        # Separate new vs existing hotkeys
        existing_hotkeys = set(self.emissions_ledger.emissions_ledgers.keys())
        new_hotkeys = active_hotkeys - existing_hotkeys

        if new_hotkeys:
            bt.logging.info(f"Detected {len(new_hotkeys)} new hotkeys to track")

        bt.logging.info(f"Updating {len(active_hotkeys)} total hotkeys "
                       f"({len(existing_hotkeys)} existing, {len(new_hotkeys)} new)")

        # Update all hotkeys
        results = {}
        for i, hotkey in enumerate(active_hotkeys, 1):
            bt.logging.info(f"Processing {i}/{len(active_hotkeys)}: {hotkey}")

            success = self.update_single_hotkey(hotkey, verbose=verbose)
            results[hotkey] = success

            # Small delay to avoid overwhelming the RPC
            if i < len(active_hotkeys):
                time.sleep(0.5)

        # Remove ledgers for hotkeys no longer active
        inactive_hotkeys = existing_hotkeys - active_hotkeys
        if inactive_hotkeys:
            bt.logging.info(f"Removing {len(inactive_hotkeys)} inactive hotkeys from tracking")
            for hotkey in inactive_hotkeys:
                if hotkey in self.emissions_ledger.emissions_ledgers:
                    del self.emissions_ledger.emissions_ledgers[hotkey]
                if hotkey in self.last_update_time:
                    del self.last_update_time[hotkey]

        elapsed = time.time() - start_time
        successes = sum(1 for s in results.values() if s)

        bt.logging.info(f"Bulk update complete: {successes}/{len(results)} successful in {elapsed:.1f}s")

        # Auto-persist if enabled
        if self.auto_persist:
            self._save_state()

        return results

    def run_update_loop(self):
        """
        Main update loop that runs continuously in background.

        Periodically updates all hotkeys and handles errors gracefully.
        """
        bt.logging.info("Starting emissions ledger update loop")

        while not self.shutdown_event.is_set():
            try:
                # Perform update
                self.update_all_hotkeys(verbose=False)

                # Wait for next update interval
                bt.logging.info(f"Sleeping for {self.update_interval_seconds/3600:.1f} hours until next update")

                # Use shutdown_event.wait() instead of time.sleep() for responsive shutdown
                if self.shutdown_event.wait(timeout=self.update_interval_seconds):
                    # Shutdown was signaled
                    break

            except Exception as e:
                bt.logging.error(f"Error in emissions ledger update loop: {e}")
                bt.logging.error(traceback.format_exc())

                # Wait before retrying
                bt.logging.info("Waiting 5 minutes before retry")
                if self.shutdown_event.wait(timeout=300):
                    break

        bt.logging.info("Emissions ledger update loop shutting down")

    def start(self, background: bool = True):
        """
        Start the emissions ledger manager.

        Args:
            background: If True, runs in background thread. If False, blocks in current thread.
        """
        if self.running:
            bt.logging.warning("EmissionsLedgerManager is already running")
            return

        self.running = True
        self.shutdown_event.clear()

        if background:
            self.update_thread = threading.Thread(
                target=self.run_update_loop,
                name="EmissionsLedgerUpdater",
                daemon=True
            )
            self.update_thread.start()
            bt.logging.info("EmissionsLedgerManager started in background")
        else:
            bt.logging.info("EmissionsLedgerManager starting in foreground")
            self.run_update_loop()

    def stop(self, wait: bool = True, timeout: float = 30.0):
        """
        Stop the emissions ledger manager.

        Args:
            wait: If True, waits for background thread to finish
            timeout: Maximum time to wait for shutdown (seconds)
        """
        if not self.running:
            bt.logging.warning("EmissionsLedgerManager is not running")
            return

        bt.logging.info("Stopping EmissionsLedgerManager")
        self.running = False
        self.shutdown_event.set()

        if wait and self.update_thread and self.update_thread.is_alive():
            bt.logging.info(f"Waiting up to {timeout}s for update thread to finish")
            self.update_thread.join(timeout=timeout)

            if self.update_thread.is_alive():
                bt.logging.warning("Update thread did not finish within timeout")
            else:
                bt.logging.info("Update thread finished cleanly")

        # Final persist
        if self.auto_persist:
            bt.logging.info("Saving final state before shutdown")
            self._save_state()

        bt.logging.info("EmissionsLedgerManager stopped")

    def get_emissions_for_hotkey(self, hotkey: str) -> List[EmissionsCheckpoint]:
        """
        Get emissions ledger for a specific hotkey.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            List of EmissionsCheckpoints
        """
        return self.emissions_ledger.get_emissions_ledger(hotkey)

    def get_cumulative_emissions(self, hotkey: str) -> float:
        """
        Get total cumulative emissions for a hotkey.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            Total emissions in TAO
        """
        return self.emissions_ledger.get_cumulative_emissions(hotkey)

    def get_all_hotkeys(self) -> List[str]:
        """
        Get list of all tracked hotkeys.

        Returns:
            List of hotkey SS58 addresses
        """
        return list(self.emissions_ledger.emissions_ledgers.keys())

    def print_summary(self):
        """Print a summary of tracked emissions"""
        hotkeys = self.get_all_hotkeys()

        print(f"\n{'='*80}")
        print(f"Emissions Ledger Manager Summary - Netuid {self.netuid}")
        print(f"{'='*80}")
        print(f"Network: {self.network}")
        print(f"Total Hotkeys Tracked: {len(hotkeys)}")
        print(f"Update Interval: {self.update_interval_seconds/3600:.1f} hours")
        print(f"Persistence Directory: {self.persistence_dir}")
        print(f"\nTop 10 Hotkeys by Cumulative Emissions:")
        print(f"{'-'*80}")
        print(f"{'Hotkey':<66} {'Total TAO':>12}")
        print(f"{'-'*80}")

        # Get emissions for all hotkeys and sort
        hotkey_emissions = [(hk, self.get_cumulative_emissions(hk)) for hk in hotkeys]
        hotkey_emissions.sort(key=lambda x: x[1], reverse=True)

        for hotkey, emissions in hotkey_emissions[:10]:
            print(f"{hotkey:<66} {emissions:>12.6f}")

        if len(hotkey_emissions) > 10:
            print(f"{'...':<66} {'...':>12}")

        total_emissions = sum(e for _, e in hotkey_emissions)
        print(f"{'-'*80}")
        print(f"{'TOTAL':<66} {total_emissions:>12.6f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Run Emissions Ledger Manager")
    parser.add_argument("--network", type=str, default="finney", help="Network name (default: finney)")
    parser.add_argument("--netuid", type=int, default=8, help="Subnet UID (default: 8)")
    parser.add_argument("--persistence-dir", type=str, help="Directory to save/load state")
    parser.add_argument("--update-interval", type=int, default=EmissionsLedgerManager.DEFAULT_UPDATE_INTERVAL_SECONDS,
                       help=f"Update interval in seconds (default: {EmissionsLedgerManager.DEFAULT_UPDATE_INTERVAL_SECONDS})")
    parser.add_argument("--archive-endpoint", type=str, action="append", dest="archive_endpoints",
                       help="Archive node endpoint (can be specified multiple times). Example: wss://archive.chain.opentensor.ai:443")
    parser.add_argument("--no-auto-persist", action="store_true", help="Disable automatic state persistence")
    parser.add_argument("--run-once", action="store_true", help="Run once and exit (don't loop)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    bt.logging.enable_info()
    if args.verbose:
        bt.logging.enable_debug()

    # Create manager
    manager = EmissionsLedgerManager(
        network=args.network,
        netuid=args.netuid,
        persistence_dir=args.persistence_dir,
        update_interval_seconds=args.update_interval,
        auto_persist=not args.no_auto_persist,
        archive_endpoints=args.archive_endpoints
    )

    if args.run_once:
        # Run once and exit
        bt.logging.info("Running single update (--run-once mode)")
        manager.update_all_hotkeys(verbose=args.verbose)
        manager.print_summary()
    else:
        # Run continuously
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            bt.logging.info(f"Received signal {signum}, shutting down gracefully...")
            manager.stop(wait=True)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start manager (blocks in foreground)
        manager.start(background=False)
