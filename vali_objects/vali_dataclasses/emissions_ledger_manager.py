"""
Emissions Ledger Manager - Persistence and Daemon for Emissions Tracking

This module provides:
1. Persistence layer for saving/loading emissions ledgers
2. Delta update functionality to avoid redundant computation
3. Background daemon for continuous updates

Architecture:
- EmissionsLedgerPersistence: Handles disk I/O with atomic writes
- ManagedEmissionsLedger: Wrapper with persistence and delta updates
- EmissionsLedgerDaemon: Background process that runs forever

Key Features:
- Implicit checkpoint tracking (no separate metadata file)
- Archive endpoint stored instead of network name
- Delta updates only compute new 12-hour chunks
- Compressed storage with atomic writes
- Automatic backups before overwrites
"""

import gzip
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.emissions_ledger import (
    EmissionsCheckpoint,
    EmissionsLedger,
    EmissionsLedgerManager as EmissionsLedgerBuilder
)


class EmissionsLedgerPersistence:
    """
    Handles persistence for emissions ledgers.

    Responsibilities:
    - Save/load ledgers to/from disk with compression
    - Atomic writes to prevent corruption
    - Create timestamped backups
    - Extract checkpoint metadata from ledgers (implicit, no separate file)
    """

    def __init__(self, running_unit_tests=False, use_compression=True):
        """
        Initialize persistence layer.

        Args:
            running_unit_tests: If True, use test directory
            use_compression: If True, use gzip compression
        """
        self.running_unit_tests = running_unit_tests
        self.use_compression = use_compression
        self.ledger_path = self._get_emissions_ledger_path()

    def _get_emissions_ledger_path(self) -> str:
        """Get path for emissions ledger file."""
        suffix = "/tests" if self.running_unit_tests else ""
        base_path = ValiConfig.BASE_DIR + f"{suffix}/validation/emissions_ledger.json"
        return base_path + ".gz" if self.use_compression else base_path

    def save_ledgers(
        self,
        ledgers: Dict[str, EmissionsLedger],
        netuid: int,
        archive_endpoint: str,
        create_backup: bool = True
    ):
        """
        Save emissions ledgers to disk with atomic write.

        Args:
            ledgers: Dictionary of hotkey -> EmissionsLedger
            netuid: Subnet UID
            archive_endpoint: Archive node URL used for queries
            create_backup: Whether to create timestamped backup before overwrite
        """
        if not ledgers:
            bt.logging.warning("No ledgers to save")
            return

        # Build data structure
        data = {
            "format_version": "1.0",
            "last_update_ms": int(time.time() * 1000),
            "netuid": netuid,
            "archive_endpoint": archive_endpoint,
            "ledgers": {}
        }

        for hotkey, ledger in ledgers.items():
            data["ledgers"][hotkey] = ledger.to_dict()

        # Create backup before overwriting
        if create_backup and os.path.exists(self.ledger_path):
            self._create_backup()

        # Atomic write: temp file -> move
        if self.use_compression:
            self._write_compressed(self.ledger_path, data)
        else:
            ValiBkpUtils.write_file(self.ledger_path, data)

        bt.logging.info(f"Saved {len(ledgers)} emissions ledgers to {self.ledger_path}")

    def load_ledgers(self) -> tuple[Dict[str, EmissionsLedger], dict]:
        """
        Load emissions ledgers from disk.

        Returns:
            Tuple of (ledgers_dict, metadata_dict)
            - ledgers_dict: Dictionary of hotkey -> EmissionsLedger
            - metadata_dict: Contains netuid, archive_endpoint, last_update_ms
        """
        if not os.path.exists(self.ledger_path):
            bt.logging.info("No existing emissions ledger file found")
            return {}, {}

        try:
            # Load data
            if self.use_compression:
                data = self._read_compressed(self.ledger_path)
            else:
                content = ValiBkpUtils.get_file(self.ledger_path)
                data = json.loads(content)

            # Extract metadata
            metadata = {
                "netuid": data.get("netuid"),
                "archive_endpoint": data.get("archive_endpoint"),
                "last_update_ms": data.get("last_update_ms"),
                "format_version": data.get("format_version", "1.0")
            }

            # Reconstruct ledgers
            ledgers = {}
            for hotkey, ledger_dict in data.get("ledgers", {}).items():
                checkpoints = [
                    EmissionsCheckpoint(
                        chunk_start_ms=cp["chunk_start_ms"],
                        chunk_end_ms=cp["chunk_end_ms"],
                        chunk_emissions=cp["chunk_emissions"],
                        cumulative_emissions=cp["cumulative_emissions"],
                        chunk_emissions_tao=cp.get("chunk_emissions_tao", 0.0),
                        cumulative_emissions_tao=cp.get("cumulative_emissions_tao", 0.0),
                        alpha_to_tao_rate_start=cp.get("alpha_to_tao_rate_start"),
                        alpha_to_tao_rate_end=cp.get("alpha_to_tao_rate_end"),
                        block_start=cp.get("block_start"),
                        block_end=cp.get("block_end")
                    )
                    for cp in ledger_dict.get("checkpoints", [])
                ]
                ledgers[hotkey] = EmissionsLedger(hotkey=hotkey, checkpoints=checkpoints)

            bt.logging.info(f"Loaded {len(ledgers)} emissions ledgers from disk")
            return ledgers, metadata

        except Exception as e:
            bt.logging.error(f"Error loading emissions ledgers: {e}")
            raise

    def get_checkpoint_info(self, ledgers: Dict[str, EmissionsLedger]) -> dict:
        """
        Extract checkpoint metadata from ledgers (implicit tracking).

        Args:
            ledgers: Dictionary of hotkey -> EmissionsLedger

        Returns:
            Dictionary with last_computed_chunk_end_ms, last_computed_block, etc.
        """
        if not ledgers:
            return {
                "last_computed_chunk_end_ms": 0,
                "last_computed_block": 0,
                "total_checkpoints": 0,
                "hotkeys_tracked": 0
            }

        last_chunk_end_ms = 0
        last_block = 0
        total_checkpoints = 0

        for ledger in ledgers.values():
            if ledger.checkpoints:
                last_checkpoint = ledger.checkpoints[-1]
                last_chunk_end_ms = max(last_chunk_end_ms, last_checkpoint.chunk_end_ms)
                if last_checkpoint.block_end:
                    last_block = max(last_block, last_checkpoint.block_end)
                total_checkpoints += len(ledger.checkpoints)

        return {
            "last_computed_chunk_end_ms": last_chunk_end_ms,
            "last_computed_block": last_block,
            "total_checkpoints": total_checkpoints,
            "hotkeys_tracked": len(ledgers)
        }

    def _create_backup(self):
        """Create timestamped backup of current ledger file."""
        backup_dir = ValiBkpUtils.get_vali_bkp_dir() + "emissions_ledgers/"
        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}emissions_ledger_{timestamp}.json.gz"

        try:
            shutil.copy2(self.ledger_path, backup_path)
            bt.logging.info(f"Created backup: {backup_path}")
        except Exception as e:
            bt.logging.warning(f"Failed to create backup: {e}")

    def _write_compressed(self, path: str, data: dict):
        """Write JSON data compressed with gzip (atomic write via temp file)."""
        temp_path = path + ".tmp"
        with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, cls=ValiBkpUtils.CustomEncoder)
        shutil.move(temp_path, path)

    def _read_compressed(self, path: str) -> dict:
        """Read compressed JSON data."""
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)


class ManagedEmissionsLedger:
    """
    Managed wrapper around EmissionsLedgerBuilder with persistence and delta updates.

    This class:
    - Loads existing ledgers at startup
    - Performs delta updates (only new chunks)
    - Saves ledgers after updates
    - Provides clean API for validator integration
    """

    # Stay 12 hours behind current time (allow two chunks for data finality)
    DEFAULT_LAG_TIME_MS = 12 * 60 * 60 * 1000

    def __init__(
        self,
        netuid: int = 8,
        archive_endpoint: str = "wss://archive.chain.opentensor.ai:443",
        rate_limit_per_second: float = 1.0,
        running_unit_tests: bool = False
    ):
        """
        Initialize managed emissions ledger.

        Args:
            netuid: Subnet UID
            archive_endpoint: Archive node URL for blockchain queries
            rate_limit_per_second: Max queries per second
            running_unit_tests: If True, use test directories
        """
        self.netuid = netuid
        self.archive_endpoint = archive_endpoint
        self.rate_limit_per_second = rate_limit_per_second
        self.running_unit_tests = running_unit_tests

        # Persistence layer
        self.persistence = EmissionsLedgerPersistence(running_unit_tests=running_unit_tests)

        # Builder (uses archive endpoint from persisted data or provided)
        self.builder = EmissionsLedgerBuilder(
            network="finney",  # Network is implicit from archive endpoint
            netuid=netuid,
            archive_endpoints=[archive_endpoint],
            rate_limit_per_second=rate_limit_per_second
        )

        # In-memory ledgers
        self.ledgers: Dict[str, EmissionsLedger] = {}

    def load_from_disk(self):
        """
        Load existing ledgers from disk.

        Returns:
            Number of ledgers loaded
        """
        self.ledgers, metadata = self.persistence.load_ledgers()

        if metadata:
            bt.logging.info(
                f"Loaded emissions ledgers: "
                f"{len(self.ledgers)} hotkeys, "
                f"last update: {TimeUtil.millis_to_formatted_date_str(metadata.get('last_update_ms', 0))}"
            )

        return len(self.ledgers)

    def save_to_disk(self, create_backup: bool = True):
        """
        Save current ledgers to disk.

        Args:
            create_backup: Whether to create timestamped backup
        """
        self.persistence.save_ledgers(
            ledgers=self.ledgers,
            netuid=self.netuid,
            archive_endpoint=self.archive_endpoint,
            create_backup=create_backup
        )

    def get_checkpoint_info(self) -> dict:
        """
        Get checkpoint metadata from current ledgers.

        Returns:
            Dict with last_computed_chunk_end_ms, last_computed_block, etc.
        """
        return self.persistence.get_checkpoint_info(self.ledgers)

    def build_delta_update(self, lag_time_ms: Optional[int] = None) -> int:
        """
        Build ONLY new chunks since last checkpoint (delta update).

        This is the key method for incremental updates. It:
        1. Gets last computed chunk from existing ledgers
        2. Only builds new chunks from last_computed_chunk_end_ms to (now - lag_time_ms)
        3. Appends new checkpoints to existing ledgers
        4. Saves updated ledgers back to disk

        Args:
            lag_time_ms: Stay this far behind current time (default: 12 hours)

        Returns:
            Number of new chunks added
        """
        if lag_time_ms is None:
            lag_time_ms = self.DEFAULT_LAG_TIME_MS

        start_time = time.time()

        # Get checkpoint info from existing ledgers
        checkpoint_info = self.get_checkpoint_info()
        last_computed_chunk_end_ms = checkpoint_info["last_computed_chunk_end_ms"]

        # Calculate new time range
        current_time_ms = int(time.time() * 1000)
        end_time_ms = current_time_ms - lag_time_ms  # Stay behind current time

        # Check if there are new chunks to compute
        if last_computed_chunk_end_ms == 0:
            # First run - perform full build
            bt.logging.info("No existing checkpoint found - performing initial full build")
            return self._build_full(end_time_ms)

        if last_computed_chunk_end_ms >= end_time_ms:
            bt.logging.info(
                f"No new chunks to compute. "
                f"Last computed: {TimeUtil.millis_to_formatted_date_str(last_computed_chunk_end_ms)}, "
                f"Target end: {TimeUtil.millis_to_formatted_date_str(end_time_ms)}"
            )
            return 0

        # Build ONLY new chunks
        start_time_ms = last_computed_chunk_end_ms

        bt.logging.info(
            f"Delta update: computing chunks from "
            f"{TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
            f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)}"
        )

        new_ledgers = self.builder.build_all_emissions_ledgers_optimized(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            verbose=False
        )

        # Count chunks added
        chunks_added = 0

        # Merge new checkpoints into existing ledgers
        for hotkey, new_ledger in new_ledgers.items():
            if hotkey in self.ledgers:
                # Append new checkpoints to existing ledger
                existing_ledger = self.ledgers[hotkey]
                for checkpoint in new_ledger.checkpoints:
                    existing_ledger.add_checkpoint(checkpoint)
                    chunks_added += 1
            else:
                # New hotkey - add entire ledger
                self.ledgers[hotkey] = new_ledger
                chunks_added += len(new_ledger.checkpoints)

        # Save to disk
        self.save_to_disk(create_backup=True)

        elapsed = time.time() - start_time
        bt.logging.info(
            f"Delta update completed in {elapsed:.2f}s - "
            f"added {chunks_added} chunks across {len(new_ledgers)} hotkeys"
        )

        return chunks_added

    def _build_full(self, end_time_ms: int, lookback_days: int = 10) -> int:
        """
        Perform full build (used for initial run).

        Args:
            end_time_ms: End time for build
            lookback_days: How many days to look back

        Returns:
            Number of chunks built
        """
        bt.logging.info(f"Building full emissions ledgers ({lookback_days} day lookback)")

        # Store builder's ledgers directly
        self.builder.build_all_emissions_ledgers_optimized(
            start_time_offset_days=lookback_days,
            end_time_ms=end_time_ms,
            verbose=False
        )

        # Copy from builder to our managed ledgers dict
        self.ledgers = self.builder.emissions_ledgers

        # Save to disk
        self.save_to_disk(create_backup=False)  # No backup on first run

        total_chunks = sum(len(ledger.checkpoints) for ledger in self.ledgers.values())
        bt.logging.info(f"Full build complete - {total_chunks} total chunks")

        return total_chunks

    def get_ledger(self, hotkey: str) -> Optional[EmissionsLedger]:
        """Get emissions ledger for a specific hotkey."""
        return self.ledgers.get(hotkey)

    def get_cumulative_emissions(self, hotkey: str) -> float:
        """Get cumulative alpha emissions for a hotkey."""
        ledger = self.get_ledger(hotkey)
        return ledger.get_cumulative_emissions() if ledger else 0.0

    def get_cumulative_emissions_tao(self, hotkey: str) -> float:
        """Get cumulative TAO emissions for a hotkey."""
        ledger = self.get_ledger(hotkey)
        return ledger.get_cumulative_emissions_tao() if ledger else 0.0


class EmissionsLedgerDaemon:
    """
    Background daemon that continuously updates emissions ledgers.

    Runs forever, checking every hour if new chunks are available.
    Performs delta updates to avoid redundant computation.

    Usage:
        daemon = EmissionsLedgerDaemon(netuid=8, archive_endpoint="wss://...")
        daemon.run()  # Blocks forever
    """

    # Check for new chunks every hour
    CHECK_INTERVAL_SECONDS = 3600

    # Stay 12 hours behind current time (two 12-hour chunks for finality)
    LAG_TIME_MS = 12 * 60 * 60 * 1000

    def __init__(
        self,
        netuid: int = 8,
        archive_endpoint: str = "wss://archive.chain.opentensor.ai:443",
        rate_limit_per_second: float = 1.0,
        check_interval_seconds: Optional[int] = None,
        lag_time_ms: Optional[int] = None
    ):
        """
        Initialize daemon.

        Args:
            netuid: Subnet UID
            archive_endpoint: Archive node URL
            rate_limit_per_second: Max queries per second
            check_interval_seconds: How often to check for new chunks (default: 3600s = 1 hour)
            lag_time_ms: How far behind current time to stay (default: 12 hours)
        """
        self.running = True
        self.check_interval = check_interval_seconds or self.CHECK_INTERVAL_SECONDS
        self.lag_time_ms = lag_time_ms or self.LAG_TIME_MS

        self.manager = ManagedEmissionsLedger(
            netuid=netuid,
            archive_endpoint=archive_endpoint,
            rate_limit_per_second=rate_limit_per_second
        )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        bt.logging.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def run(self):
        """Main daemon loop - runs forever until interrupted."""
        bt.logging.info("=" * 80)
        bt.logging.info("Emissions Ledger Daemon Started")
        bt.logging.info("=" * 80)
        bt.logging.info(f"NetUID: {self.manager.netuid}")
        bt.logging.info(f"Archive Endpoint: {self.manager.archive_endpoint}")
        bt.logging.info(f"Rate Limit: {self.manager.rate_limit_per_second} req/sec")
        bt.logging.info(f"Check Interval: {self.check_interval}s ({self.check_interval/3600:.1f} hours)")
        bt.logging.info(f"Lag Time: {self.lag_time_ms/1000/3600:.1f} hours behind current time")
        bt.logging.info("=" * 80)

        # Load existing data
        try:
            bt.logging.info("Loading existing ledgers from disk...")
            num_loaded = self.manager.load_from_disk()

            if num_loaded > 0:
                checkpoint_info = self.manager.get_checkpoint_info()
                bt.logging.info(
                    f"Loaded {num_loaded} ledgers, "
                    f"last checkpoint: {TimeUtil.millis_to_formatted_date_str(checkpoint_info['last_computed_chunk_end_ms'])}"
                )
            else:
                bt.logging.info("No existing ledgers found - will perform initial build on first update")
        except Exception as e:
            bt.logging.error(f"Error loading ledgers: {e}")

        # Perform initial delta update
        try:
            bt.logging.info("Performing initial delta update...")
            chunks_added = self.manager.build_delta_update(lag_time_ms=self.lag_time_ms)
            bt.logging.info(f"Initial update complete - added {chunks_added} chunks")
        except Exception as e:
            bt.logging.error(f"Initial update failed: {e}")

        # Main loop
        while self.running:
            try:
                next_check_time = time.time() + self.check_interval
                next_check_str = datetime.fromtimestamp(next_check_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                bt.logging.info(f"Next check at: {next_check_str}")

                # Sleep in small intervals to allow graceful shutdown
                while self.running and time.time() < next_check_time:
                    time.sleep(10)

                if not self.running:
                    break

                # Perform delta update
                bt.logging.info("Checking for new chunks...")
                chunks_added = self.manager.build_delta_update(lag_time_ms=self.lag_time_ms)

                if chunks_added > 0:
                    bt.logging.info(f"Added {chunks_added} new chunks")
                else:
                    bt.logging.info("No new chunks available")

            except Exception as e:
                bt.logging.error(f"Error in daemon loop: {e}")
                # Continue running even if one update fails
                time.sleep(60)

        bt.logging.info("Emissions Ledger Daemon stopped")


# Entry point for running as standalone daemon
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emissions Ledger Background Daemon")
    parser.add_argument("--netuid", type=int, default=8, help="Subnet UID (default: 8)")
    parser.add_argument("--archive-endpoint", type=str,
                       default="wss://archive.chain.opentensor.ai:443",
                       help="Archive node endpoint URL")
    parser.add_argument("--rate-limit", type=float, default=1.0,
                       help="Max queries per second (default: 1.0)")
    parser.add_argument("--check-interval", type=int, default=3600,
                       help="Check interval in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--lag-hours", type=float, default=12.0,
                       help="Hours behind current time to stay (default: 12)")

    args = parser.parse_args()

    bt.logging.enable_info()

    lag_time_ms = int(args.lag_hours * 60 * 60 * 1000)

    daemon = EmissionsLedgerDaemon(
        netuid=args.netuid,
        archive_endpoint=args.archive_endpoint,
        rate_limit_per_second=args.rate_limit,
        check_interval_seconds=args.check_interval,
        lag_time_ms=lag_time_ms
    )

    daemon.run()
