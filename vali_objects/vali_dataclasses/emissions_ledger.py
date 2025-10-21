"""
Emissions Ledger - Tracks theta (TAO) emissions for hotkeys in 12-hour UTC chunks

This module builds emissions ledgers by querying on-chain data to track how much theta
has been awarded to each hotkey over its entire history since registration.

Emissions are tracked in 12-hour chunks aligned with UTC day:
- Chunk 1: 00:00 UTC - 12:00 UTC
- Chunk 2: 12:00 UTC - 00:00 UTC (next day)

Each checkpoint stores both the emissions for that specific 12-hour chunk and
the cumulative emissions up to that point.

Architecture:
- EmissionsCheckpoint: Data for a single 12-hour chunk
- EmissionsLedger: Emissions history for a SINGLE hotkey
- EmissionsLedgerManager: Builds and manages ledgers for multiple hotkeys

Standalone Usage:
    python -m vali_objects.vali_dataclasses.emissions_ledger --hotkey <hotkey> --netuid 8
"""
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import bittensor as bt
import time
# Import scalecodec for SS58 encoding
import scalecodec

@dataclass
class EmissionsCheckpoint:
    """
    Stores emissions data for a 12-hour UTC chunk.

    Attributes:
        chunk_start_ms: Start timestamp of the 12-hour chunk (milliseconds)
        chunk_end_ms: End timestamp of the 12-hour chunk (milliseconds)
        chunk_emissions: Alpha tokens earned during this specific 12-hour chunk
        cumulative_emissions: Total alpha tokens earned from registration up to end of this chunk
        chunk_emissions_tao: Approximate TAO value of chunk emissions (using avg conversion rate)
        cumulative_emissions_tao: Total approximate TAO value from registration up to end of this chunk
        alpha_to_tao_rate_start: Alpha-to-TAO conversion rate at chunk start
        alpha_to_tao_rate_end: Alpha-to-TAO conversion rate at chunk end
        block_start: Block number at chunk start (for verification)
        block_end: Block number at chunk end (for verification)
    """
    chunk_start_ms: int
    chunk_end_ms: int
    chunk_emissions: float
    cumulative_emissions: float
    chunk_emissions_tao: float = 0.0
    cumulative_emissions_tao: float = 0.0
    alpha_to_tao_rate_start: Optional[float] = None
    alpha_to_tao_rate_end: Optional[float] = None
    block_start: Optional[int] = None
    block_end: Optional[int] = None

    def __eq__(self, other):
        if not isinstance(other, EmissionsCheckpoint):
            return False
        return (
            self.chunk_start_ms == other.chunk_start_ms
            and self.chunk_end_ms == other.chunk_end_ms
            and abs(self.chunk_emissions - other.chunk_emissions) < 1e-9
            and abs(self.cumulative_emissions - other.cumulative_emissions) < 1e-9
        )

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'chunk_start_ms': self.chunk_start_ms,
            'chunk_end_ms': self.chunk_end_ms,
            'chunk_start_utc': datetime.fromtimestamp(self.chunk_start_ms / 1000, tz=timezone.utc).isoformat(),
            'chunk_end_utc': datetime.fromtimestamp(self.chunk_end_ms / 1000, tz=timezone.utc).isoformat(),
            'chunk_emissions': self.chunk_emissions,
            'cumulative_emissions': self.cumulative_emissions,
            'chunk_emissions_tao': self.chunk_emissions_tao,
            'cumulative_emissions_tao': self.cumulative_emissions_tao,
            'alpha_to_tao_rate_start': self.alpha_to_tao_rate_start,
            'alpha_to_tao_rate_end': self.alpha_to_tao_rate_end,
            'block_start': self.block_start,
            'block_end': self.block_end,
        }


class EmissionsLedger:
    """
    Emissions ledger for a SINGLE hotkey.

    Stores the complete emissions history as a series of 12-hour checkpoints,
    along with methods to query and visualize the data.
    """

    def __init__(self, hotkey: str, checkpoints: Optional[List[EmissionsCheckpoint]] = None):
        """
        Initialize emissions ledger for a single hotkey.

        Args:
            hotkey: SS58 address of the hotkey
            checkpoints: Optional list of emission checkpoints
        """
        self.hotkey = hotkey
        self.checkpoints: List[EmissionsCheckpoint] = checkpoints or []

    def add_checkpoint(self, checkpoint: EmissionsCheckpoint):
        """Add a checkpoint to the ledger."""
        self.checkpoints.append(checkpoint)

    def get_cumulative_emissions(self) -> float:
        """
        Get total cumulative alpha emissions for this hotkey.

        Returns:
            Total alpha emissions (float)
        """
        if not self.checkpoints:
            return 0.0
        return self.checkpoints[-1].cumulative_emissions

    def get_cumulative_emissions_tao(self) -> float:
        """
        Get total cumulative TAO emissions for this hotkey.

        Returns:
            Total TAO emissions (float)
        """
        if not self.checkpoints:
            return 0.0
        return self.checkpoints[-1].cumulative_emissions_tao

    def to_dict(self) -> dict:
        """
        Convert ledger to dictionary for serialization.

        Returns:
            Dictionary with hotkey and all checkpoints
        """
        return {
            'hotkey': self.hotkey,
            'total_checkpoints': len(self.checkpoints),
            'cumulative_emissions': self.get_cumulative_emissions(),
            'cumulative_emissions_tao': self.get_cumulative_emissions_tao(),
            'checkpoints': [cp.to_dict() for cp in self.checkpoints]
        }

    def print_summary(self):
        """Print a formatted summary of emissions for this hotkey."""
        if not self.checkpoints:
            print(f"\nNo emissions data found for {self.hotkey}")
            return

        print(f"\n{'='*80}")
        print(f"Emissions Summary for {self.hotkey}")
        print(f"{'='*80}")
        print(f"Total Checkpoints: {len(self.checkpoints)}")
        print(f"Total Emissions: {self.get_cumulative_emissions():.6f} alpha (~{self.get_cumulative_emissions_tao():.6f} TAO)")
        print(f"\nFirst 5 Checkpoints:")
        print(f"{'Chunk Start (UTC)':<25} {'Chunk End (UTC)':<25} {'Chunk Alpha':>15} {'Cumulative Alpha':>15}")
        print(f"{'-'*80}")

        for checkpoint in self.checkpoints[:5]:
            start_dt = datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(checkpoint.chunk_end_ms / 1000, tz=timezone.utc)
            print(f"{start_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                  f"{end_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                  f"{checkpoint.chunk_emissions:>15.6f} "
                  f"{checkpoint.cumulative_emissions:>15.6f}")

        if len(self.checkpoints) > 10:
            print(f"{'...':<25} {'...':<25} {'...':>15} {'...':>15}")
            print(f"\nLast 5 Checkpoints:")
            print(f"{'Chunk Start (UTC)':<25} {'Chunk End (UTC)':<25} {'Chunk Alpha':>15} {'Cumulative Alpha':>15}")
            print(f"{'-'*80}")

            for checkpoint in self.checkpoints[-5:]:
                start_dt = datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc)
                end_dt = datetime.fromtimestamp(checkpoint.chunk_end_ms / 1000, tz=timezone.utc)
                print(f"{start_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                      f"{end_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                      f"{checkpoint.chunk_emissions:>15.6f} "
                      f"{checkpoint.cumulative_emissions:>15.6f}")

        print(f"{'='*80}\n")

    def plot_emissions(self, save_path: Optional[str] = None):
        """
        Plot emissions data using matplotlib.

        Creates two subplots:
        1. Bar chart of chunk emissions over time (alpha & TAO)
        2. Line chart of cumulative emissions over time (alpha & TAO)

        Args:
            save_path: Optional path to save the plot (default: display only)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            bt.logging.warning("matplotlib not installed, skipping plot. Install with: pip install matplotlib")
            return

        if not self.checkpoints:
            bt.logging.warning(f"No emissions data found for {self.hotkey}, skipping plot")
            return

        # Extract data for plotting
        chunk_starts = [datetime.fromtimestamp(cp.chunk_start_ms / 1000, tz=timezone.utc) for cp in self.checkpoints]
        chunk_emissions = [cp.chunk_emissions for cp in self.checkpoints]
        cumulative_emissions = [cp.cumulative_emissions for cp in self.checkpoints]
        chunk_emissions_tao = [cp.chunk_emissions_tao for cp in self.checkpoints]
        cumulative_emissions_tao = [cp.cumulative_emissions_tao for cp in self.checkpoints]

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Emissions Analysis for {self.hotkey[:16]}...{self.hotkey[-8:]}', fontsize=14, fontweight='bold')

        # Subplot 1: Chunk Emissions (Bar Chart with dual y-axes)
        ax1_alpha = ax1
        ax1_alpha.bar(chunk_starts, chunk_emissions, width=0.4, alpha=0.6, color='steelblue', edgecolor='navy', label='Alpha')
        ax1_alpha.set_xlabel('Date (UTC)', fontsize=11)
        ax1_alpha.set_ylabel('Alpha per Chunk', fontsize=11, color='steelblue')
        ax1_alpha.tick_params(axis='y', labelcolor='steelblue')
        ax1_alpha.set_title('Emissions per 12-Hour Chunk (Alpha & TAO)', fontsize=12, fontweight='bold')
        ax1_alpha.grid(True, alpha=0.3, linestyle='--')

        # TAO emissions on right y-axis
        ax1_tao = ax1_alpha.twinx()
        ax1_tao.plot(chunk_starts, chunk_emissions_tao, linewidth=2, color='orange', marker='o',
                    markersize=4, alpha=0.8, label='TAO')
        ax1_tao.set_ylabel('TAO per Chunk', fontsize=11, color='orange')
        ax1_tao.tick_params(axis='y', labelcolor='orange')

        # Format x-axis
        ax1_alpha.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1_alpha.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.checkpoints)//20)))
        plt.setp(ax1_alpha.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add statistics text box
        total_emissions = cumulative_emissions[-1] if cumulative_emissions else 0
        total_emissions_tao = cumulative_emissions_tao[-1] if cumulative_emissions_tao else 0
        avg_chunk = sum(chunk_emissions) / len(chunk_emissions) if chunk_emissions else 0
        avg_chunk_tao = sum(chunk_emissions_tao) / len(chunk_emissions_tao) if chunk_emissions_tao else 0
        nonzero_chunks = sum(1 for e in chunk_emissions if e > 0)

        stats_text = f'Total: {total_emissions:.6f} alpha (~{total_emissions_tao:.6f} TAO)\n'
        stats_text += f'Avg/Chunk: {avg_chunk:.6f} alpha (~{avg_chunk_tao:.6f} TAO)\n'
        stats_text += f'Active Chunks: {nonzero_chunks}/{len(self.checkpoints)}'

        ax1_alpha.text(0.98, 0.97, stats_text,
                transform=ax1_alpha.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Subplot 2: Cumulative Emissions (Line Chart with dual y-axes)
        ax2_alpha = ax2
        ax2_alpha.plot(chunk_starts, cumulative_emissions, linewidth=2, color='darkgreen', marker='o',
                      markersize=3, alpha=0.7, label='Alpha')
        ax2_alpha.fill_between(chunk_starts, cumulative_emissions, alpha=0.2, color='lightgreen')
        ax2_alpha.set_xlabel('Date (UTC)', fontsize=11)
        ax2_alpha.set_ylabel('Cumulative Alpha', fontsize=11, color='darkgreen')
        ax2_alpha.tick_params(axis='y', labelcolor='darkgreen')
        ax2_alpha.set_title('Cumulative Emissions Over Time (Alpha & TAO)', fontsize=12, fontweight='bold')
        ax2_alpha.grid(True, alpha=0.3, linestyle='--')

        # TAO emissions on right y-axis
        ax2_tao = ax2_alpha.twinx()
        ax2_tao.plot(chunk_starts, cumulative_emissions_tao, linewidth=2, color='darkorange', marker='s',
                    markersize=3, alpha=0.7, label='TAO')
        ax2_tao.fill_between(chunk_starts, cumulative_emissions_tao, alpha=0.2, color='peachpuff')
        ax2_tao.set_ylabel('Cumulative TAO', fontsize=11, color='darkorange')
        ax2_tao.tick_params(axis='y', labelcolor='darkorange')

        # Format x-axis
        ax2_alpha.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2_alpha.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.checkpoints)//20)))
        plt.setp(ax2_alpha.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add final value annotations
        if chunk_starts and cumulative_emissions:
            ax2_alpha.annotate(f'{total_emissions:.6f} alpha',
                        xy=(chunk_starts[-1], cumulative_emissions[-1]),
                        xytext=(-60, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkgreen'))

            ax2_tao.annotate(f'{total_emissions_tao:.6f} TAO',
                        xy=(chunk_starts[-1], cumulative_emissions_tao[-1]),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='darkorange',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='peachpuff', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkorange'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            bt.logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


class EmissionsLedgerManager:
    """
    Manages emissions tracking for Bittensor hotkeys.

    Queries on-chain data to build a historical record of emissions received
    by hotkeys, organized into 12-hour UTC-aligned chunks.

    The ledger tracks emissions from the time a miner first registered on the subnet
    until the current block, aggregating data into consistent time windows.
    """

    # Bittensor blocks are produced every ~12 seconds
    SECONDS_PER_BLOCK = 12

    # 12 hours in milliseconds
    CHUNK_DURATION_MS = 12 * 60 * 60 * 1000

    # Blocks per 12-hour chunk (approximate)
    BLOCKS_PER_CHUNK = int(CHUNK_DURATION_MS / 1000 / SECONDS_PER_BLOCK)

    # Default offset in days from current time for emissions tracking
    DEFAULT_START_TIME_OFFSET_DAYS = 10

    def __init__(
        self,
        network: str = "finney",
        netuid: int = 8,
        archive_endpoints: Optional[List[str]] = None,
        rate_limit_per_second: float = 1.0
    ):
        """
        Initialize EmissionsLedger with blockchain connection.

        Args:
            network: Bittensor network name ("finney", "test", "local")
            netuid: Subnet UID to query (default: 8 for mainnet PTN)
            archive_endpoints: List of archive node endpoints for historical queries.
                Example: ["wss://archive.chain.opentensor.ai:443"]
            rate_limit_per_second: Maximum queries per second (default: 1.0 for official endpoints)
        """
        self.network = network
        self.netuid = netuid
        self.rate_limit_per_second = rate_limit_per_second
        self.last_query_time = 0.0

        # Initialize subtensor connection
        bt.logging.info(f"Connecting to network: {network}, netuid: {netuid}")

        # Use archive endpoint if provided, otherwise use default network
        if archive_endpoints and len(archive_endpoints) > 0:
            chain_endpoint = archive_endpoints[0]
            bt.logging.info(f"Using archive endpoint as primary: {chain_endpoint}")

            # Create proper bittensor config with archive endpoint
            import argparse
            parser = argparse.ArgumentParser()
            bt.subtensor.add_args(parser)
            config = bt.config(parser, args=[])

            # Override the chain endpoint
            config.subtensor.chain_endpoint = chain_endpoint

            # Clear network so it uses our custom endpoint
            config.subtensor.network = None

            self.subtensor = bt.subtensor(config=config)
            bt.logging.info(f"Connected to: {self.subtensor.chain_endpoint}")
        else:
            # Use default network
            self.subtensor = bt.subtensor(network=network)

        # Storage for emissions ledgers (one per hotkey)
        self.emissions_ledgers: Dict[str, EmissionsLedger] = {}

        if rate_limit_per_second < 10:
            bt.logging.warning(f"Rate limit set to {rate_limit_per_second} req/sec - queries will be slow")
        bt.logging.info("EmissionsLedgerManager initialized")

    def _rate_limit(self):
        """
        Enforce rate limiting by sleeping if necessary.

        Ensures we don't exceed rate_limit_per_second requests per second.
        """
        if self.rate_limit_per_second <= 0:
            return  # No rate limiting

        current_time = time.time()
        time_since_last_query = current_time - self.last_query_time
        min_interval = 1.0 / self.rate_limit_per_second

        if time_since_last_query < min_interval:
            sleep_time = min_interval - time_since_last_query
            time.sleep(sleep_time)

        self.last_query_time = time.time()

    @staticmethod
    def get_chunk_boundaries(timestamp_ms: int) -> tuple[int, int]:
        """
        Calculate the 12-hour UTC chunk boundaries for a given timestamp.

        Chunks are aligned to UTC day:
        - Chunk 1: 00:00 UTC - 12:00 UTC
        - Chunk 2: 12:00 UTC - 00:00 UTC (next day)

        Args:
            timestamp_ms: Timestamp in milliseconds

        Returns:
            Tuple of (chunk_start_ms, chunk_end_ms)
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

        # Determine which chunk this timestamp falls into
        if dt.hour < 12:
            # Morning chunk: 00:00 - 12:00
            chunk_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            chunk_end = dt.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            # Afternoon/evening chunk: 12:00 - 00:00 next day
            chunk_start = dt.replace(hour=12, minute=0, second=0, microsecond=0)
            # Next day at 00:00
            chunk_end = (dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))

        chunk_start_ms = int(chunk_start.timestamp() * 1000)
        chunk_end_ms = int(chunk_end.timestamp() * 1000)

        return chunk_start_ms, chunk_end_ms


    def query_alpha_to_tao_rate(self, block_number: int) -> Optional[float]:
        """
        Query the alpha-to-TAO conversion rate at a specific block.

        The conversion rate is calculated from subnet pool reserves:
        alpha_price_in_tao = TAO_reserve / Alpha_reserve

        Args:
            block_number: Block number to query

        Returns:
            Alpha-to-TAO conversion rate, or None if query fails or if querying pre-dTAO blocks
        """
        # Rate limit before block hash query
        self._rate_limit()

        # Get block hash for the specific block
        block_hash = self.subtensor.substrate.get_block_hash(block_number)
        if not block_hash:
            bt.logging.warning(f"Could not get hash for block {block_number}")
            return None

        try:
            # Rate limit before SubnetTAO query
            self._rate_limit()

            # Query TAO reserve
            tao_reserve_query = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='SubnetTAO',
                params=[self.netuid],
                block_hash=block_hash
            )

            # Rate limit before SubnetAlphaIn query
            self._rate_limit()

            # Query Alpha reserve
            alpha_reserve_query = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='SubnetAlphaIn',
                params=[self.netuid],
                block_hash=block_hash
            )
        except Exception as e:
            # Only catch "storage function not found" errors (pre-dTAO blocks)
            # Let all other exceptions propagate (fail fast)
            raise

        if tao_reserve_query is None or alpha_reserve_query is None:
            bt.logging.debug(f"Pool reserve data not available at block {block_number}")
            return None

        # Extract values (stored in RAO)
        tao_reserve_rao = float(tao_reserve_query.value if hasattr(tao_reserve_query, 'value') else tao_reserve_query)
        alpha_reserve_rao = float(alpha_reserve_query.value if hasattr(alpha_reserve_query, 'value') else alpha_reserve_query)

        if alpha_reserve_rao == 0:
            bt.logging.warning(f"Alpha reserve is zero at block {block_number}")
            return None

        # Convert from RAO to tokens (both use 1e9 divisor)
        tao_in_pool = tao_reserve_rao / 1e9
        alpha_in_pool = alpha_reserve_rao / 1e9

        # Calculate alpha price in TAO
        alpha_to_tao_rate = tao_in_pool / alpha_in_pool

        return alpha_to_tao_rate

    def build_all_emissions_ledgers_optimized(
        self,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        start_time_offset_days: int = None,
        verbose: bool = False
    ) -> Dict[str, EmissionsLedger]:
        """
        Build emissions ledgers for ALL hotkeys in the subnet efficiently.

        This is the optimized approach that queries each block once and processes
        all hotkeys simultaneously, sharing block hashes, emission vectors, and
        alpha-to-TAO conversion rates across all hotkeys.

        Args:
            start_time_ms: Optional start time in milliseconds (overrides start_time_offset_days)
            end_time_ms: Optional end time (default: current time)
            start_time_offset_days: Number of days to look back from now (default: 10)
            verbose: Enable detailed logging

        Returns:
            Dictionary mapping hotkeys to their EmissionsLedger objects
        """
        start_exec_time = time.time()
        bt.logging.info("Building emissions ledgers for all hotkeys (optimized mode)")

        # Rate limit before initial query
        self._rate_limit()

        # Verify max UIDs is still 256 (sanity check - has never changed in Bittensor history)
        current_max_uids_result = self.subtensor.substrate.query(
            module='SubtensorModule',
            storage_function='SubnetworkN',
            params=[self.netuid]
        )
        current_max_uids = int(current_max_uids_result.value if hasattr(current_max_uids_result, 'value') else current_max_uids_result) if current_max_uids_result else 0
        assert current_max_uids == 256, f"Expected max UIDs to be 256, but got {current_max_uids}. The hardcoded value needs to be updated!"

        # Calculate start time dynamically if not specified
        if start_time_ms is None:
            if start_time_offset_days is None:
                start_time_offset_days = self.DEFAULT_START_TIME_OFFSET_DAYS

            # Calculate start time as (now - offset days)
            current_time = datetime.now(timezone.utc)
            start_time = current_time - timedelta(days=start_time_offset_days)
            start_time_ms = int(start_time.timestamp() * 1000)

            bt.logging.info(f"Using dynamic start date ({start_time_offset_days} days ago): {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Default end time is now
        if end_time_ms is None:
            end_time_ms = int(time.time() * 1000)

        # Get current block and estimate block range
        current_block = self.subtensor.get_current_block()
        current_time_ms = int(time.time() * 1000)

        seconds_from_start = (current_time_ms - start_time_ms) / 1000
        blocks_from_start = int(seconds_from_start / self.SECONDS_PER_BLOCK)
        start_block = current_block - blocks_from_start

        seconds_from_end = (current_time_ms - end_time_ms) / 1000
        blocks_from_end = int(seconds_from_end / self.SECONDS_PER_BLOCK)
        end_block = current_block - blocks_from_end

        # Format dates for logging
        start_date = datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        end_date = datetime.fromtimestamp(end_time_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        bt.logging.info(f"Block range: {start_block} to {end_block} ({start_date} to {end_date})")

        # Initialize EmissionsLedger objects for all hotkeys
        hotkey_ledgers: Dict[str, EmissionsLedger] = {}
        hotkey_cumulative_alpha: Dict[str, float] = defaultdict(float)
        hotkey_cumulative_tao: Dict[str, float] = defaultdict(float)

        # Generate list of 12-hour chunks
        current_chunk_start_ms, current_chunk_end_ms = self.get_chunk_boundaries(start_time_ms)

        # If start time is not at chunk boundary, adjust to next chunk
        if current_chunk_start_ms < start_time_ms:
            current_chunk_start_ms = current_chunk_end_ms
            current_chunk_end_ms = current_chunk_start_ms + self.CHUNK_DURATION_MS

        chunk_count = 0
        hotkeys_processed_cumulative = set()  # Track all unique hotkeys seen so far
        # Track all hotkeys we've seen across all blocks
        all_hotkeys_seen = set()

        while current_chunk_start_ms < end_time_ms:
            chunk_count += 1
            chunk_start_time = time.time()

            # Determine actual time range for this chunk
            actual_start_ms = max(current_chunk_start_ms, start_time_ms)
            actual_end_ms = min(current_chunk_end_ms, end_time_ms)

            # Calculate corresponding blocks
            seconds_from_chunk_start = (actual_start_ms - start_time_ms) / 1000
            seconds_from_chunk_end = (actual_end_ms - start_time_ms) / 1000
            chunk_start_block = start_block + int(seconds_from_chunk_start / self.SECONDS_PER_BLOCK)
            chunk_end_block = start_block + int(seconds_from_chunk_end / self.SECONDS_PER_BLOCK)

            blocks_in_chunk = chunk_end_block - chunk_start_block + 1

            # Calculate emissions for ALL hotkeys in this chunk (optimized)
            chunk_results = self._calculate_emissions_for_all_hotkeys(
                chunk_start_block,
                chunk_end_block,
                all_hotkeys_seen,
            )

            # Track hotkeys with activity in this chunk
            hotkeys_active_in_chunk = set()

            # Update each hotkey's ledger
            # Iterate over ALL hotkeys in chunk_results (includes newly discovered historical hotkeys)
            for hotkey, (alpha_emissions, tao_emissions, rate_start, rate_end) in chunk_results.items():
                # If this is a newly discovered hotkey (not in current metagraph), create ledger for it
                if hotkey not in hotkey_ledgers:
                    #bt.logging.info(f"Discovered historical hotkey not in current metagraph: {hotkey[:16]}...{hotkey[-8:]}")
                    hotkey_ledgers[hotkey] = EmissionsLedger(hotkey)
                    hotkey_cumulative_alpha[hotkey] = 0.0
                    hotkey_cumulative_tao[hotkey] = 0.0

                hotkey_cumulative_alpha[hotkey] += alpha_emissions
                hotkey_cumulative_tao[hotkey] += tao_emissions

                # Track if this hotkey had any emissions in this chunk
                if alpha_emissions > 0:
                    hotkeys_active_in_chunk.add(hotkey)
                    hotkeys_processed_cumulative.add(hotkey)

                checkpoint = EmissionsCheckpoint(
                    chunk_start_ms=current_chunk_start_ms,
                    chunk_end_ms=current_chunk_end_ms,
                    chunk_emissions=alpha_emissions,
                    cumulative_emissions=hotkey_cumulative_alpha[hotkey],
                    chunk_emissions_tao=tao_emissions,
                    cumulative_emissions_tao=hotkey_cumulative_tao[hotkey],
                    alpha_to_tao_rate_start=rate_start,
                    alpha_to_tao_rate_end=rate_end,
                    block_start=chunk_start_block,
                    block_end=chunk_end_block
                )

                hotkey_ledgers[hotkey].add_checkpoint(checkpoint)

            # Also add zero-emission checkpoints for currently registered hotkeys that had no emissions this chunk
            for hotkey in all_hotkeys_seen:
                if hotkey not in chunk_results:
                    # Hotkey exists now but had no emissions in this chunk
                    checkpoint = EmissionsCheckpoint(
                        chunk_start_ms=current_chunk_start_ms,
                        chunk_end_ms=current_chunk_end_ms,
                        chunk_emissions=0.0,
                        cumulative_emissions=hotkey_cumulative_alpha[hotkey],
                        chunk_emissions_tao=0.0,
                        cumulative_emissions_tao=hotkey_cumulative_tao[hotkey],
                        alpha_to_tao_rate_start=chunk_results.get(list(chunk_results.keys())[0], (0, 0, None, None))[2] if chunk_results else None,
                        alpha_to_tao_rate_end=chunk_results.get(list(chunk_results.keys())[0], (0, 0, None, None))[3] if chunk_results else None,
                        block_start=chunk_start_block,
                        block_end=chunk_end_block
                    )
                    hotkey_ledgers[hotkey].add_checkpoint(checkpoint)

            # Calculate chunk processing time
            chunk_elapsed = time.time() - chunk_start_time

            # Format chunk date range for logging
            chunk_start_dt = datetime.fromtimestamp(current_chunk_start_ms / 1000, tz=timezone.utc)
            chunk_end_dt = datetime.fromtimestamp(current_chunk_end_ms / 1000, tz=timezone.utc)
            date_range = f"{chunk_start_dt.strftime('%Y-%m-%d %H:%M')} - {chunk_end_dt.strftime('%Y-%m-%d %H:%M')} UTC"

            # Log progress for every chunk
            bt.logging.info(
                f"Chunk {chunk_count} ({date_range}): "
                f"{blocks_in_chunk} blocks, "
                f"{chunk_elapsed:.2f}s, "
                f"{len(hotkeys_active_in_chunk)} active hotkeys, "
                f"{len(hotkeys_processed_cumulative)} cumulative unique"
            )

            # Move to next chunk
            current_chunk_start_ms = current_chunk_end_ms
            current_chunk_end_ms = current_chunk_start_ms + self.CHUNK_DURATION_MS

        # Store all ledgers
        self.emissions_ledgers = hotkey_ledgers

        elapsed_time = time.time() - start_exec_time
        bt.logging.info(f"Built {chunk_count} chunks for {len(hotkey_ledgers)} hotkeys in {elapsed_time:.2f} seconds")

        # Log summary for each hotkey
        for hotkey, ledger in hotkey_ledgers.items():
            if ledger.checkpoints:
                total_alpha = ledger.get_cumulative_emissions()
                total_tao = ledger.get_cumulative_emissions_tao()
                bt.logging.info(f"  {hotkey[:16]}...{hotkey[-8:]}: {total_alpha:.6f} alpha (~{total_tao:.6f} TAO)")

        return self.emissions_ledgers

    def _get_uid_to_hotkey_at_block(self, block_hash: str) -> Dict[int, str]:
        """
        Query UID-to-hotkey mapping at a specific historical block.

        Args:
            block_hash: Block hash to query at

        Returns:
            Dictionary mapping UID to hotkey at that block
        """
        # Rate limit before query_map
        self._rate_limit()

        # Query all Keys entries for this netuid at once (much more efficient than 256 individual queries)
        keys_map = self.subtensor.substrate.query_map(
            module='SubtensorModule',
            storage_function='Keys',
            params=[self.netuid],
            block_hash=block_hash
        )

        # Convert result to dict[uid -> hotkey]
        uid_to_hotkey = {}
        for key, value in keys_map:
            # Key is the UID directly (integer)
            uid = int(key)

            # Value is a ScaleObj containing the hotkey as bytes
            # Convert to SS58 address string
            if hasattr(value, 'value'):
                # value.value is a list of tuples with byte values
                # Example: [(24, 91, 223, ...)] - extract bytes
                if isinstance(value.value, list) and len(value.value) > 0:
                    hotkey_bytes = bytes(value.value[0]) if isinstance(value.value[0], (list, tuple)) else bytes(value.value)
                else:
                    hotkey_bytes = bytes(value.value)

                # Convert bytes to SS58 address
                hotkey = scalecodec.ss58_encode(hotkey_bytes.hex(), ss58_format=42)

                if hotkey:
                    uid_to_hotkey[uid] = hotkey
            else:
                # Fallback: try to use value as-is
                hotkey = str(value)
                if hotkey:
                    uid_to_hotkey[uid] = hotkey

        return uid_to_hotkey


    def _calculate_emissions_for_all_hotkeys(
        self,
        start_block: int,
        end_block: int,
        all_hotkeys_seen: Optional[set] = None,
    ) -> Dict[str, tuple[float, float, Optional[float], Optional[float]]]:
        """
        Calculate emissions for ALL hotkeys between two blocks (OPTIMIZED).

        This method queries each block once and extracts data for all UIDs,
        sharing block hashes and alpha-to-TAO rates across all hotkeys.

        IMPORTANT: Queries UID-to-hotkey mapping at each historical block to
        correctly attribute emissions even if hotkeys changed UIDs over time.

        Args:
            start_block: Starting block (inclusive)
            end_block: Ending block (inclusive)
            verbose: Enable detailed logging

        Returns:
            Dictionary mapping hotkey to (alpha_emissions, tao_emissions, rate_start, rate_end)
        """
        # Sample blocks at regular intervals
        sample_interval = int(3600 / self.SECONDS_PER_BLOCK)  # ~300 blocks per hour
        sampled_blocks = list(range(start_block, end_block + 1, sample_interval))
        if sampled_blocks[-1] != end_block:
            sampled_blocks.append(end_block)

        # Initialize tracking for emissions
        hotkey_total_alpha = defaultdict(float)
        hotkey_total_tao = defaultdict(float)

        rate_at_start = None
        rate_at_end = None
        previous_block = start_block
        previous_emissions_by_hotkey = {}
        previous_alpha_to_tao_rate = None

        for i, block in enumerate(sampled_blocks):
            # Rate limit before block hash query
            self._rate_limit()

            # Query block hash ONCE
            block_hash = self.subtensor.substrate.get_block_hash(block)
            if not block_hash:
                bt.logging.warning(f"Could not get hash for block {block}")
                raise Exception(f"Block hash query failed for block {block}")

            # Query UID-to-hotkey mapping AT THIS BLOCK (rate limited inside method)
            current_uid_to_hotkey = self._get_uid_to_hotkey_at_block(block_hash)

            # Track new hotkeys
            for hotkey in current_uid_to_hotkey.values():
                if hotkey not in all_hotkeys_seen:
                    all_hotkeys_seen.add(hotkey)

            # Rate limit before Emission query
            self._rate_limit()

            # Query ALL emissions at once (entire vector)
            emission_query = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='Emission',
                params=[self.netuid],
                block_hash=block_hash
            )

            if emission_query is not None:
                emissions_list = emission_query.value if hasattr(emission_query, 'value') else emission_query
            else:
                emissions_list = []


            # Query alpha-to-TAO rate ONCE (shared by all hotkeys)
            alpha_to_tao_rate = self.query_alpha_to_tao_rate(block)

            # Track first and last rates
            if i == 0:
                rate_at_start = alpha_to_tao_rate
            if i == len(sampled_blocks) - 1:
                rate_at_end = alpha_to_tao_rate

            # Fallback to previous rate if query fails
            if alpha_to_tao_rate is None:
                alpha_to_tao_rate = previous_alpha_to_tao_rate if previous_alpha_to_tao_rate is not None else 0.0

            # Process emissions for each UID -> hotkey (at this block)
            current_emissions_by_hotkey = defaultdict(float)
            for uid, hotkey in current_uid_to_hotkey.items():
                # Extract emission for this UID from the vector
                if isinstance(emissions_list, (list, tuple)) and uid < len(emissions_list):
                    emission_rao = float(emissions_list[uid])
                    emission_alpha = emission_rao / 1e9  # Convert RAO to alpha
                    emission_per_block = emission_alpha / 360  # Convert per-tempo to per-block
                elif isinstance(emissions_list, dict) and uid in emissions_list:
                    emission_rao = float(emissions_list[uid])
                    emission_alpha = emission_rao / 1e9
                    emission_per_block = emission_alpha / 360
                else:
                    emission_per_block = 0.0

                current_emissions_by_hotkey[hotkey] += emission_per_block

                # Calculate segment emissions if we have previous data
                if hotkey in previous_emissions_by_hotkey:
                    blocks_elapsed = block - previous_block
                    avg_alpha_rate = (previous_emissions_by_hotkey[hotkey] + current_emissions_by_hotkey[hotkey]) / 2
                    segment_emissions_alpha = avg_alpha_rate * blocks_elapsed
                    hotkey_total_alpha[hotkey] += segment_emissions_alpha

                    # Convert to TAO using average conversion rate for this segment
                    avg_conversion_rate = (previous_alpha_to_tao_rate + alpha_to_tao_rate) / 2
                    segment_emissions_tao = segment_emissions_alpha * avg_conversion_rate
                    hotkey_total_tao[hotkey] += segment_emissions_tao

            # Update state for next iteration
            previous_emissions_by_hotkey = current_emissions_by_hotkey
            previous_alpha_to_tao_rate = alpha_to_tao_rate
            previous_block = block

        # Build result dictionary for ALL hotkeys seen
        results = {}
        for hotkey in all_hotkeys_seen:
            results[hotkey] = (
                hotkey_total_alpha.get(hotkey, 0.0),
                hotkey_total_tao.get(hotkey, 0.0),
                rate_at_start,
                rate_at_end
            )

        return results


    def get_ledger(self, hotkey: str) -> Optional[EmissionsLedger]:
        """
        Get the emissions ledger for a specific hotkey.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            EmissionsLedger object, or None if not found
        """
        return self.emissions_ledgers.get(hotkey)

    def get_cumulative_emissions(self, hotkey: str) -> float:
        """
        Get the total cumulative alpha emissions for a hotkey.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            Total alpha emissions
        """
        ledger = self.get_ledger(hotkey)
        if not ledger:
            return 0.0
        return ledger.get_cumulative_emissions()

    def print_emissions_summary(self, hotkey: str):
        """
        Print a formatted summary of emissions for a hotkey.

        Delegates to the EmissionsLedger object's print_summary() method.

        Args:
            hotkey: SS58 address of the hotkey
        """
        ledger = self.get_ledger(hotkey)
        if ledger:
            ledger.print_summary()
        else:
            print(f"\nNo emissions data found for {hotkey}")

    def plot_emissions(self, hotkey: str, save_path: Optional[str] = None):
        """
        Plot emissions data for a hotkey using matplotlib.

        Delegates to the EmissionsLedger object's plot_emissions() method.

        Args:
            hotkey: SS58 address of the hotkey
            save_path: Optional path to save the plot (default: display only)
        """
        ledger = self.get_ledger(hotkey)
        if ledger:
            ledger.plot_emissions(save_path)
        else:
            bt.logging.warning(f"No emissions data found for {hotkey}, skipping plot")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build emissions ledger for Bittensor hotkeys")
    parser.add_argument("--hotkey", type=str, help="Hotkey to display/focus on (optional, displays one plot)", default=None)
    parser.add_argument("--netuid", type=int, default=8, help="Subnet UID (default: 8)")
    parser.add_argument("--network", type=str, default="finney", help="Network name (default: finney)")
    parser.add_argument("--archive-endpoint", type=str, action="append", dest="archive_endpoints",
                       help="Archive node endpoint (can be specified multiple times). Example: wss://archive.chain.opentensor.ai:443")
    parser.add_argument("--start-time-offset-days", type=int, default=10,
                       help="Number of days to look back from now for emissions tracking (default: 10)")
    parser.add_argument("--show-plot", action="store_true", help="Display plot for specified hotkey (requires --hotkey)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    bt.logging.enable_info()
    if args.verbose:
        bt.logging.enable_debug()

    # Initialize ledger manager
    emissions_ledger_manager = EmissionsLedgerManager(
        network=args.network,
        netuid=args.netuid,
        archive_endpoints=args.archive_endpoints
    )

    # ALWAYS build ALL ledgers using optimized method
    bt.logging.info("Building emissions ledgers for ALL hotkeys in subnet (optimized mode)")
    emissions_ledger_manager.build_all_emissions_ledgers_optimized(
        start_time_offset_days=args.start_time_offset_days,
        verbose=args.verbose
    )

    if len(emissions_ledger_manager.emissions_ledgers) == 0:
        bt.logging.error("No emissions ledgers were built")
        exit(1)

    # Create emissions_ledger_plots directory
    # Get project root (2 levels up from vali_objects/vali_dataclasses/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    plots_dir = os.path.join(project_root, "emissions_ledger_plots")

    os.makedirs(plots_dir, exist_ok=True)
    bt.logging.info(f"Saving plots to: {plots_dir}")

    # Save ALL plots to emissions_ledger_plots/{hotkey}.png
    for hotkey in emissions_ledger_manager.emissions_ledgers.keys():
        plot_path = os.path.join(plots_dir, f"{hotkey}.png")
        try:
            emissions_ledger_manager.plot_emissions(hotkey, save_path=plot_path)
            bt.logging.info(f"Saved plot for {hotkey[:16]}...{hotkey[-8:]}")
        except Exception as e:
            bt.logging.error(f"Error saving plot for {hotkey}: {e}")

    bt.logging.info(f"All plots saved to {plots_dir}")

    # Print summary for specified hotkey or all hotkeys
    if args.hotkey:
        if args.hotkey in emissions_ledger_manager.emissions_ledgers:
            emissions_ledger_manager.print_emissions_summary(args.hotkey)

            # Optionally display the plot for this hotkey
            if args.show_plot:
                bt.logging.info(f"Displaying plot for {args.hotkey}")
                emissions_ledger_manager.plot_emissions(args.hotkey, save_path=None)
        else:
            bt.logging.error(f"Hotkey {args.hotkey} not found in built ledgers")
    else:
        # Print summaries for all hotkeys
        bt.logging.info("Printing summaries for all hotkeys")
        for hotkey in emissions_ledger_manager.emissions_ledgers.keys():
            emissions_ledger_manager.print_emissions_summary(hotkey)
