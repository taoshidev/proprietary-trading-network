"""
Penalty Ledger - Tracks penalty checkpoints aligned with performance ledger checkpoints

This module builds penalty ledgers for miners based on their performance ledgers and positions.
Penalties include drawdown threshold, risk profile, and minimum collateral penalties.

"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime, timezone
import time
import signal
import gzip
import json
import os
import shutil
from vali_objects.position import Position
from vali_objects.utils.asset_selection_manager import AssetSelectionManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, TP_ID_PORTFOLIO
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.position_filter import PositionFilter
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.vali_config import ValiConfig
from time_util.time_util import TimeUtil
from ptn_api.slack_notifier import SlackNotifier
import bittensor as bt


class PenaltyInputType(Enum):
    LEDGER = auto()
    POSITIONS = auto()
    PSEUDO_POSITIONS = auto()
    COLLATERAL = auto()
    ASSET_LEDGER = auto()


@dataclass
class PenaltyConfig:
    function: callable
    input_type: PenaltyInputType


class PenaltyCheckpoint:
    """Stores penalty values aligned with a PerfCheckpoint timestamp"""

    def __init__(
        self,
        last_processed_ms: int,
        drawdown_penalty: float = 1.0,
        risk_profile_penalty: float = 1.0,
        min_collateral_penalty: float = 1.0,
        risk_adjusted_performance_penalty: float = 1.0,
        total_penalty: float = 1.0,
        challenge_period_status: str = "unknown"
    ):
        self.last_processed_ms = int(last_processed_ms)
        self.drawdown_penalty = float(drawdown_penalty)
        self.risk_profile_penalty = float(risk_profile_penalty)
        self.min_collateral_penalty = float(min_collateral_penalty)
        self.risk_adjusted_performance_penalty = float(risk_adjusted_performance_penalty)
        self.total_penalty = float(total_penalty)
        self.challenge_period_status = str(challenge_period_status)

    def __eq__(self, other):
        if not isinstance(other, PenaltyCheckpoint):
            return False
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'item'):  # numpy types
                result[key] = value.item()
            else:
                result[key] = value
        return result

class PenaltyLedger:
    """
    Penalty ledger for a SINGLE miner.

    Stores the complete penalty history as a series of checkpoints aligned with
    performance ledger checkpoints.
    """

    def __init__(self, hotkey: str, checkpoints: Optional[List[PenaltyCheckpoint]] = None):
        """
        Initialize penalty ledger for a single miner.

        Args:
            hotkey: SS58 address of the miner's hotkey
            checkpoints: Optional list of penalty checkpoints
        """
        self.hotkey = hotkey
        self.checkpoints: List[PenaltyCheckpoint] = checkpoints or []

    def add_checkpoint(self, checkpoint: PenaltyCheckpoint, target_cp_duration_ms: int):
        """
        Add a checkpoint to the ledger.

        Validates that the new checkpoint is after the previous checkpoint.

        Args:
            checkpoint: The checkpoint to add
            target_cp_duration_ms: Target checkpoint duration in milliseconds

        Raises:
            AssertionError: If the checkpoint timestamp is not after the previous checkpoint
        """
        assert checkpoint.last_processed_ms % target_cp_duration_ms == 0, (
            f"Checkpoint timestamp {checkpoint.last_processed_ms} must align with target_cp_duration_ms "
            f"{target_cp_duration_ms} for {self.hotkey}"
        )
        if self.checkpoints:
            prev_checkpoint = self.checkpoints[-1]
            # First check it's after previous checkpoint
            assert checkpoint.last_processed_ms > prev_checkpoint.last_processed_ms, (
                f"Checkpoint timestamp must be after previous checkpoint for {self.hotkey}: "
                f"new checkpoint at {checkpoint.last_processed_ms}, "
                f"but previous checkpoint at {prev_checkpoint.last_processed_ms}"
            )
            # Then check exact spacing
            assert checkpoint.last_processed_ms - prev_checkpoint.last_processed_ms == target_cp_duration_ms, (
                f"Checkpoint spacing must be exactly {target_cp_duration_ms}ms for {self.hotkey}: "
                f"new checkpoint at {checkpoint.last_processed_ms}, "
                f"previous at {prev_checkpoint.last_processed_ms}, "
                f"spacing is {checkpoint.last_processed_ms - prev_checkpoint.last_processed_ms}ms"
            )

        self.checkpoints.append(checkpoint)

    def get_latest_checkpoint(self) -> Optional[PenaltyCheckpoint]:
        """Get the most recent checkpoint, or None if no checkpoints exist."""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_checkpoint_at_time(self, timestamp_ms: int, target_cp_duration_ms: int) -> Optional[PenaltyCheckpoint]:
        """
        Get the checkpoint at a specific timestamp (efficient O(1) lookup).

        Uses index calculation instead of scanning since checkpoints are evenly-spaced
        and contiguous (enforced by strict add_checkpoint validation).

        Args:
            timestamp_ms: Exact timestamp to query (should match last_processed_ms)
            target_cp_duration_ms: Target checkpoint duration in milliseconds

        Returns:
            Checkpoint at the exact timestamp, or None if not found

        Raises:
            ValueError: If checkpoint exists at calculated index but timestamp doesn't match (data corruption)
        """
        if not self.checkpoints:
            return None

        # Calculate expected index based on first checkpoint and duration
        first_checkpoint_ms = self.checkpoints[0].last_processed_ms

        # Check if timestamp is before first checkpoint
        if timestamp_ms < first_checkpoint_ms:
            return None

        # Calculate index (checkpoints are evenly spaced by target_cp_duration_ms)
        time_diff = timestamp_ms - first_checkpoint_ms
        if time_diff % target_cp_duration_ms != 0:
            # Timestamp doesn't align with checkpoint boundaries
            return None

        index = time_diff // target_cp_duration_ms

        # Check if index is within bounds
        if index >= len(self.checkpoints):
            return None

        # Validate the checkpoint at this index has the expected timestamp
        checkpoint = self.checkpoints[index]
        if checkpoint.last_processed_ms != timestamp_ms:
            from time_util.time_util import TimeUtil
            raise ValueError(
                f"Data corruption detected for {self.hotkey}: "
                f"checkpoint at index {index} has last_processed_ms {checkpoint.last_processed_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(checkpoint.last_processed_ms)}), "
                f"but expected {timestamp_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(timestamp_ms)}). "
                f"Checkpoints are not properly contiguous."
            )

        return checkpoint

    def to_dict(self) -> dict:
        """
        Convert ledger to dictionary for serialization.

        Returns:
            Dictionary with hotkey and all checkpoints
        """
        return {
            'hotkey': self.hotkey,
            'total_checkpoints': len(self.checkpoints),
            'checkpoints': [cp.to_dict() for cp in self.checkpoints]
        }

    @staticmethod
    def from_dict(data: dict) -> 'PenaltyLedger':
        """
        Reconstruct ledger from dictionary.

        Args:
            data: Dictionary containing ledger data

        Returns:
            Reconstructed PenaltyLedger
        """
        checkpoints = []
        for cp_dict in data.get('checkpoints', []):
            checkpoint = PenaltyCheckpoint(
                last_processed_ms=cp_dict['last_processed_ms'],
                drawdown_penalty=cp_dict.get('drawdown_penalty', 1.0),
                risk_profile_penalty=cp_dict.get('risk_profile_penalty', 1.0),
                min_collateral_penalty=cp_dict.get('min_collateral_penalty', 1.0),
                risk_adjusted_performance_penalty=cp_dict.get('risk_adjusted_performance_penalty', 1.0),
                total_penalty=cp_dict.get('total_penalty', 1.0),
                challenge_period_status=cp_dict.get('challenge_period_status', 'unknown')
            )
            checkpoints.append(checkpoint)

        return PenaltyLedger(hotkey=data['hotkey'], checkpoints=checkpoints)

class PenaltyLedgerManager:
    """
    Manages penalty ledgers aligned with performance checkpoints.
    Reads positions and perf ledgers to build penalty checkpoints.
    """

    # Default check interval for daemon mode (12 hours in seconds)
    DEFAULT_CHECK_INTERVAL_SECONDS = 12 * 60 * 60

    # Define the penalties configuration
    PENALTIES_CONFIG = {
        'drawdown_threshold': PenaltyConfig(
            function=LedgerUtils.max_drawdown_threshold_penalty,
            input_type=PenaltyInputType.LEDGER
        ),
        'risk_profile': PenaltyConfig(
            function=PositionPenalties.risk_profile_penalty,
            input_type=PenaltyInputType.POSITIONS
        ),
        'min_collateral': PenaltyConfig(
            function=ValidatorContractManager.min_collateral_penalty,
            input_type=PenaltyInputType.COLLATERAL
        ),
        'risk_adjusted_performance': PenaltyConfig(
            function=PositionPenalties.risk_adjusted_performance_penalty,
            input_type=PenaltyInputType.ASSET_LEDGER
        )
    }

    def __init__(
        self,
        position_manager: PositionManager,
        perf_ledger_manager: PerfLedgerManager,
        contract_manager: ValidatorContractManager,
        asset_selection_manager: AssetSelectionManager,
        challengeperiod_manager=None,
        ipc_manager=None,
        running_unit_tests: bool = False,
        slack_webhook_url=None,
        run_daemon: bool = False,
        validator_hotkey: Optional[str] = None
    ):
        """
        Initialize PenaltyLedgerManager with managers for positions, performance ledgers, and collateral.

        Args:
            position_manager: Manager for reading miner positions
            perf_ledger_manager: Manager for reading performance ledgers
            contract_manager: Manager for reading miner collateral/account sizes
            asset_selection_manager: Manager for tracking miner asset class selections
            challengeperiod_manager: Optional manager for challenge period status (for real-time status)
            ipc_manager: Optional IPC manager for multiprocessing
            running_unit_tests: Whether this is being run in unit tests
            slack_webhook_url: Optional Slack webhook URL for failure notifications
            run_daemon: If True, automatically start daemon process (default: False)
        """
        self.position_manager = position_manager
        self.perf_ledger_manager = perf_ledger_manager
        self.contract_manager = contract_manager
        self.asset_selection_manager = asset_selection_manager
        self.challengeperiod_manager = challengeperiod_manager
        self.running_unit_tests = running_unit_tests

        # Storage for penalty checkpoints per miner
        self.penalty_ledgers: Dict[str, PenaltyLedger] = ipc_manager.dict() if ipc_manager else {}

        # Daemon control
        self.running = False

        # Slack notifications
        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)

        # Load existing ledgers from disk
        self.load_from_disk()
        if run_daemon:
            self._start_daemon_process()

    def _start_daemon_process(self):
        """Start the daemon process for continuous updates."""
        import multiprocessing
        daemon_process = multiprocessing.Process(
            target=self.run_daemon_forever,
            args=(),
            kwargs={'verbose': False}
        )
        daemon_process.daemon = True
        daemon_process.start()
        bt.logging.info("Started PenaltyLedgerManager daemon process")

    def get_positions_at_date(self, cutoff_date_ms: int, positions: List[Position]) -> List[Position]:
        """
        Get all positions that are open at a given date (in ms).

        This utility function is critical for build_penalty_ledgers to work.
        Uses PositionFilter.filter_single_position_simple for filtering.

        Args:
            cutoff_date_ms: Timestamp in milliseconds to filter positions
            positions: List of positions to filter

        Returns:
            List of positions filtered by date
        """
        filtered_positions = []

        for position in positions:
            filtered_position = PositionFilter.filter_single_position_simple(position, cutoff_date_ms)
            if filtered_position:
                filtered_positions.append(filtered_position)

        return filtered_positions

    # ============================================================================
    # PERSISTENCE METHODS
    # ============================================================================

    def _get_ledger_path(self) -> str:
        """Get path for penalty ledger file."""
        suffix = "/tests" if self.running_unit_tests else ""
        base_path = ValiConfig.BASE_DIR + f"{suffix}/validation/penalty_ledger.json"
        return base_path + ".gz"

    def save_to_disk(self, create_backup: bool = True):
        """
        Save penalty ledgers to disk with atomic write.

        Args:
            create_backup: Whether to create timestamped backup before overwrite
        """
        if not self.penalty_ledgers:
            bt.logging.warning("No penalty ledgers to save")
            return

        ledger_path = self._get_ledger_path()

        # Build data structure
        data = {
            "format_version": "1.0",
            "last_update_ms": int(time.time() * 1000),
            "ledgers": {}
        }

        for hotkey, ledger in self.penalty_ledgers.items():
            data["ledgers"][hotkey] = ledger.to_dict()

        # Atomic write: temp file -> move
        self._write_compressed(ledger_path, data)

        bt.logging.info(f"Saved {len(self.penalty_ledgers)} penalty ledgers to {ledger_path}")

    def load_from_disk(self) -> int:
        """
        Load existing ledgers from disk.

        Returns:
            Number of ledgers loaded
        """
        ledger_path = self._get_ledger_path()

        if not os.path.exists(ledger_path):
            bt.logging.info("No existing penalty ledger file found")
            return 0

        # Load data
        data = self._read_compressed(ledger_path)

        # Extract metadata
        metadata = {
            "last_update_ms": data.get("last_update_ms"),
            "format_version": data.get("format_version", "1.0")
        }

        # Reconstruct ledgers
        for hotkey, ledger_dict in data.get("ledgers", {}).items():
            ledger = PenaltyLedger.from_dict(ledger_dict)
            self.penalty_ledgers[hotkey] = ledger

        bt.logging.info(
            f"Loaded {len(self.penalty_ledgers)} penalty ledgers, "
            f"metadata: {metadata}, "
            f"last update: {TimeUtil.millis_to_formatted_date_str(metadata.get('last_update_ms', 0))}"
        )

        return len(self.penalty_ledgers)


    def _write_compressed(self, path: str, data: dict):
        """Write JSON data compressed with gzip (atomic write via temp file)."""
        temp_path = path + ".tmp"
        with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        shutil.move(temp_path, path)

    def _read_compressed(self, path: str) -> dict:
        """Read compressed JSON data."""
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def get_last_processed_ms(self, miner_hotkey: str) -> int:
        """
        Get the last processed timestamp for a miner's penalty ledger.

        This is a helper method to modularize delta update logic.

        Args:
            miner_hotkey: The miner's hotkey

        Returns:
            Last processed timestamp in milliseconds, or 0 if no checkpoints exist
        """
        if miner_hotkey not in self.penalty_ledgers:
            return 0

        penalty_ledger = self.penalty_ledgers[miner_hotkey]
        if not penalty_ledger.checkpoints:
            return 0

        last_checkpoint = penalty_ledger.get_latest_checkpoint()
        return last_checkpoint.last_processed_ms

    # ============================================================================
    # DAEMON MODE
    # ============================================================================

    def _calculate_next_aligned_time(self) -> float:
        """
        Calculate the next 12-hour UTC-aligned timestamp (00:00 or 12:00 UTC).

        Returns:
            Unix timestamp of the next aligned time
        """
        now_utc = datetime.now(timezone.utc)

        # Get current hour in UTC
        current_hour = now_utc.hour

        # Determine next aligned time (00:00 or 12:00 UTC)
        if current_hour < 12:
            # Next alignment is 12:00 today
            next_aligned = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            # Next alignment is 00:00 tomorrow
            from datetime import timedelta
            next_aligned = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        return next_aligned.timestamp()

    def run_daemon_forever(self, check_interval_seconds: Optional[int] = None, verbose: bool = False):
        """
        Run as daemon - continuously update penalty ledgers forever.

        Checks for new performance checkpoints at 12-hour UTC-aligned intervals (00:00 and 12:00 UTC).
        Handles graceful shutdown on SIGINT/SIGTERM.

        Features:
        - Delta updates (only processes new checkpoints since last update)
        - UTC-aligned refresh (at 00:00 and 12:00 UTC for accurate data per checkpoint)
        - Graceful shutdown
        - Automatic retry on failures

        Args:
            check_interval_seconds: Deprecated - now uses 12-hour UTC alignment instead
            verbose: Enable detailed logging
        """
        self.running = True

        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            bt.logging.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        bt.logging.info("=" * 80)
        bt.logging.info("Penalty Ledger Manager - Daemon Mode (UTC-Aligned)")
        bt.logging.info("=" * 80)
        bt.logging.info("Refresh Schedule: 00:00 UTC and 12:00 UTC (12-hour intervals)")
        bt.logging.info(f"Delta Update Mode: Enabled (resumes from last checkpoint)")
        bt.logging.info(f"Slack Notifications: {'Enabled' if self.slack_notifier.webhook_url else 'Disabled'}")
        bt.logging.info("=" * 80)

        # Track consecutive failures for exponential backoff
        consecutive_failures = 0
        initial_backoff_seconds = 300  # Start with 5 minutes
        max_backoff_seconds = 3600  # Max 1 hour
        backoff_multiplier = 2

        time.sleep(120) # Initial delay to stagger large ipc reads

        # Main loop
        while self.running:
            try:
                bt.logging.info("Starting penalty ledger delta update...")
                start_time = time.time()

                # Perform delta update (only new checkpoints)
                self.build_penalty_ledgers(verbose=verbose, delta_update=True)

                elapsed = time.time() - start_time
                bt.logging.info(f"Delta update completed in {elapsed:.2f}s")

                # Success - reset failure counter
                if consecutive_failures > 0:
                    bt.logging.info(f"Recovered after {consecutive_failures} failure(s)")
                    # Send recovery alert with VM/git/hotkey context
                    self.slack_notifier.send_ledger_recovery_alert("Penalty Ledger", consecutive_failures)

                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1

                # Calculate backoff for logging
                backoff_seconds = min(
                    initial_backoff_seconds * (backoff_multiplier ** (consecutive_failures - 1)),
                    max_backoff_seconds
                )

                bt.logging.error(
                    f"Error in daemon loop (failure #{consecutive_failures}): {e}",
                    exc_info=True
                )

                # Send Slack alert with VM/git/hotkey context
                self.slack_notifier.send_ledger_failure_alert(
                    "Penalty Ledger",
                    consecutive_failures,
                    e,
                    backoff_seconds
                )

            # Calculate sleep time and sleep
            if self.running:
                if consecutive_failures > 0:
                    # Exponential backoff on failure
                    backoff_seconds = min(
                        initial_backoff_seconds * (backoff_multiplier ** (consecutive_failures - 1)),
                        max_backoff_seconds
                    )
                    next_check_time = time.time() + backoff_seconds
                    next_check_str = datetime.fromtimestamp(next_check_time, tz=timezone.utc).strftime(
                        '%Y-%m-%d %H:%M:%S UTC')
                    bt.logging.warning(
                        f"Retrying after {consecutive_failures} failure(s). "
                        f"Backoff: {backoff_seconds}s. Next attempt at: {next_check_str}"
                    )
                    # Sleep in small intervals to allow graceful shutdown
                    while self.running and time.time() < next_check_time:
                        time.sleep(10)
                else:
                    # Normal interval - align to next 12-hour UTC boundary
                    # Recalculate target time periodically for precision
                    next_check_time = self._calculate_next_aligned_time()
                    next_check_str = datetime.fromtimestamp(next_check_time, tz=timezone.utc).strftime(
                        '%Y-%m-%d %H:%M:%S UTC')
                    time_until_next = next_check_time - time.time()
                    bt.logging.info(f"Next check at: {next_check_str} (in {time_until_next / 3600:.2f} hours)")

                    # Sleep in smaller chunks (60s) and recalculate target time periodically
                    # This ensures we hit the UTC boundary as precisely as possible
                    last_log_time = time.time()
                    while self.running:
                        current_time = time.time()

                        # Recalculate next boundary every iteration to account for drift
                        next_check_time = self._calculate_next_aligned_time()
                        time_remaining = next_check_time - current_time

                        # If we've reached or passed the boundary, break and run update
                        if time_remaining <= 0:
                            bt.logging.info("Reached UTC boundary - triggering update")
                            break

                        # Log progress every hour
                        if current_time - last_log_time >= 3600:
                            bt.logging.info(f"Time until next boundary: {time_remaining / 3600:.2f} hours")
                            last_log_time = current_time

                        # Sleep for 60 seconds or remaining time, whichever is smaller
                        # This provides good balance between precision and CPU efficiency
                        sleep_duration = min(60.0, time_remaining)
                        time.sleep(sleep_duration)

        bt.logging.info("Penalty Ledger Manager daemon stopped")

    def build_penalty_ledgers(self, verbose: bool = False, delta_update: bool = True):
        """
        Build penalty ledgers for all checkpoints in all performance ledgers.

        This function iterates through all checkpoints in each miner's portfolio perf ledger
        and computes the penalties at each checkpoint time using the positions at that time.

        Supports delta updates: only processes checkpoints after the last processed checkpoint.

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints since last update. If False, rebuild from scratch.
        """
        if not delta_update:
            self.penalty_ledgers.clear()
            bt.logging.info("Full rebuild mode: clearing existing penalty ledgers")

        # Read all perf ledgers from perf ledger manager
        all_perf_ledgers: Dict[str, Dict[str, PerfLedger]] = self.perf_ledger_manager.get_perf_ledgers(
            portfolio_only=False
        )
        all_positions: Dict[str, List[Position]] = self.position_manager.get_positions_for_all_miners()

        # OPTIMIZATION: Fetch entire active_miners dict once upfront to avoid O(n) IPC calls
        # Instead of calling get_miner_bucket() for each miner (which makes an IPC call each time),
        # we fetch the entire dict once and do local lookups
        challenge_period_statuses = {}
        if self.challengeperiod_manager:
            # Make a single IPC call to get the entire dict
            active_miners_snapshot = dict(self.challengeperiod_manager.active_miners)
            # Extract just the bucket status for each hotkey (first element of the tuple)
            challenge_period_statuses = {
                hotkey: bucket_tuple[0].value if bucket_tuple and bucket_tuple[0] else "unknown"
                for hotkey, bucket_tuple in active_miners_snapshot.items()
            }

        bt.logging.info(
            f"Building penalty ledgers for {len(all_perf_ledgers)} hotkeys "
            f"({'delta update' if delta_update else 'full rebuild'})"
        )

        hotkeys_processed = 0
        total_checkpoints_added = 0

        for miner_hotkey, ledger_dict in all_perf_ledgers.items():
            # Get portfolio ledger for this miner
            portfolio_ledger = ledger_dict.get(TP_ID_PORTFOLIO)

            if not portfolio_ledger or not portfolio_ledger.cps:
                raise ValueError(f"No portfolio ledger found for miner {miner_hotkey}")

            # Get or create penalty ledger for this miner
            if miner_hotkey in self.penalty_ledgers:
                penalty_ledger = self.penalty_ledgers[miner_hotkey]
            else:
                penalty_ledger = PenaltyLedger(miner_hotkey)

            # Determine starting point for delta updates using helper method
            last_processed_ms = self.get_last_processed_ms(miner_hotkey) if delta_update else 0
            if delta_update and last_processed_ms > 0:
                if verbose:
                    bt.logging.info(
                        f"Delta update for {miner_hotkey}: resuming from {last_processed_ms}"
                    )

            # Get miner's collateral/account size
            miner_account_size = self.contract_manager.miner_account_sizes.get(miner_hotkey, 0)
            if miner_account_size is None:
                miner_account_size = 0

            # Iterate through checkpoints in the portfolio ledger (only new ones if delta_update)
            checkpoints_processed = 0
            for checkpoint in portfolio_ledger.cps:
                # Skip checkpoints we've already processed in delta update mode
                if delta_update:
                    if checkpoint.last_update_ms <= last_processed_ms: # We have already processed this checkpoint
                        continue
                    if checkpoint.accum_ms != portfolio_ledger.target_cp_duration_ms:
                        # This is still an active checkpoint - skip it
                        continue

                checkpoint_ms = checkpoint.last_update_ms

                # Get positions at this checkpoint time
                miner_positions = all_positions.get(miner_hotkey, [])
                miner_positions_at_checkpoint = self.get_positions_at_date(checkpoint_ms, miner_positions)

                # Calculate each penalty
                penalties = {}
                total_penalty = 1.0

                for penalty_name, penalty_config in self.PENALTIES_CONFIG.items():
                    penalty_value = 1.0

                    try:
                        if penalty_config.input_type == PenaltyInputType.LEDGER:
                            # Use the portfolio ledger up to this checkpoint
                            # Create a temporary ledger with only checkpoints up to current time
                            temp_ledger = PerfLedger(
                                initialization_time_ms=portfolio_ledger.initialization_time_ms,
                                max_return=portfolio_ledger.max_return,
                                target_cp_duration_ms=portfolio_ledger.target_cp_duration_ms,
                                target_ledger_window_ms=portfolio_ledger.target_ledger_window_ms,
                                cps=[cp for cp in portfolio_ledger.cps if cp.last_update_ms <= checkpoint_ms],
                                tp_id=portfolio_ledger.tp_id
                            )
                            penalty_value = penalty_config.function(temp_ledger)

                        elif penalty_config.input_type == PenaltyInputType.POSITIONS:
                            penalty_value = penalty_config.function(miner_positions_at_checkpoint)

                        elif penalty_config.input_type == PenaltyInputType.COLLATERAL:
                            penalty_value = penalty_config.function(miner_account_size)

                        elif penalty_config.input_type == PenaltyInputType.ASSET_LEDGER:
                            segmentation_machine = AssetSegmentation({miner_hotkey: ledger_dict})
                            accumulated_penalty = 1

                            asset_class = self.asset_selection_manager.asset_selections.get(miner_hotkey);
                            if not asset_class:
                                accumulated_penalty = 0
                            else:
                                subcategories = ValiConfig.ASSET_CLASS_BREAKDOWN[asset_class].get("subcategory_weights", {}).keys()

                                subcategory_penalties = []
                                for subcategory in subcategories:
                                    asset_ledger = segmentation_machine.segmentation(subcategory).get(miner_hotkey)
                                    if not asset_ledger or not asset_ledger.cps:
                                        continue
                                    subcategory_penalty = penalty_config.function(asset_ledger, asset_class)
                                    subcategory_penalties.append(subcategory_penalty)

                                if subcategory_penalties:
                                    category_penalty = sum(subcategory_penalties) / len(subcategory_penalties)
                                    accumulated_penalty *= category_penalty

                            penalty_value = accumulated_penalty

                    except Exception as e:
                        if verbose:
                            bt.logging.warning(
                                f"Error computing {penalty_name} for miner {miner_hotkey} at {checkpoint_ms}: {e}"
                            )
                        penalty_value = 1.0

                    penalties[penalty_name] = penalty_value
                    total_penalty *= penalty_value

                # Get challenge period status (real-time if available, otherwise "unknown")
                challenge_period_status = "unknown"
                if challenge_period_statuses:
                    # For historical checkpoints, use "unknown"
                    # Only populate status for the most recent checkpoint (real-time)
                    latest_cp = portfolio_ledger.cps[-1] if portfolio_ledger.cps else None
                    if latest_cp and checkpoint_ms == latest_cp.last_update_ms:
                        # This is the most recent checkpoint - get real-time status from local dict
                        # (no IPC call needed - we fetched all statuses upfront)
                        challenge_period_status = challenge_period_statuses.get(miner_hotkey, "unknown")

                # Create penalty checkpoint
                penalty_checkpoint = PenaltyCheckpoint(
                    last_processed_ms=checkpoint_ms,
                    drawdown_penalty=penalties.get('drawdown_threshold', 1.0),
                    risk_profile_penalty=penalties.get('risk_profile', 1.0),
                    min_collateral_penalty=penalties.get('min_collateral', 1.0),
                    risk_adjusted_performance_penalty=penalties.get('risk_adjusted_performance', 1.0),
                    total_penalty=total_penalty,
                    challenge_period_status=challenge_period_status
                )

                # Add checkpoint to ledger (validates ordering)
                penalty_ledger.add_checkpoint(penalty_checkpoint, portfolio_ledger.target_cp_duration_ms)
                checkpoints_processed += 1

            # IMPORTANT: For IPC-managed dicts, we must retrieve, mutate, and reassign
            # to propagate changes (managed dicts don't track nested mutations)
            if checkpoints_processed > 0:
                self.penalty_ledgers[miner_hotkey] = penalty_ledger  # Reassign to trigger IPC update
                hotkeys_processed += 1
                total_checkpoints_added += checkpoints_processed
                if verbose:
                    bt.logging.info(
                        f"Processed {checkpoints_processed} new penalty checkpoints for miner {miner_hotkey} "
                        f"(total: {len(penalty_ledger.checkpoints)})"
                    )

        bt.logging.info(
            f"Built penalty ledgers: {hotkeys_processed} hotkeys processed, "
            f"{total_checkpoints_added} new checkpoints added, "
            f"{len(self.penalty_ledgers)} total ledgers"
        )

        # Save to disk after building
        self.save_to_disk()

    def get_penalty_ledger(self, miner_hotkey: str) -> Optional[PenaltyLedger]:
        """
        Get the penalty ledger for a specific miner.

        Args:
            miner_hotkey: The miner's hotkey

        Returns:
            PenaltyLedger for the miner, or None if not found
        """
        return self.penalty_ledgers.get(miner_hotkey, None)

