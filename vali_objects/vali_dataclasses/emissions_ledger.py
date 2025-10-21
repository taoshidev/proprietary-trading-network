"""
Emissions Ledger - Tracks theta (TAO) emissions for hotkeys in 12-hour UTC chunks

This module builds emissions ledgers by querying on-chain data to track how much theta
has been awarded to each hotkey over its entire history since registration.

Emissions are tracked in 12-hour chunks aligned with UTC day:
- Chunk 1: 00:00 UTC - 12:00 UTC
- Chunk 2: 12:00 UTC - 00:00 UTC (next day)

Each checkpoint stores both the emissions for that specific 12-hour chunk and
the cumulative emissions up to that point.

Standalone Usage:
    python -m vali_objects.vali_dataclasses.emissions_ledger --hotkey <hotkey> --netuid 8
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import bittensor as bt
import time


@dataclass
class EmissionsCheckpoint:
    """
    Stores emissions data for a 12-hour UTC chunk.

    Attributes:
        chunk_start_ms: Start timestamp of the 12-hour chunk (milliseconds)
        chunk_end_ms: End timestamp of the 12-hour chunk (milliseconds)
        chunk_emissions: Theta earned during this specific 12-hour chunk
        cumulative_emissions: Total theta earned from registration up to end of this chunk
        block_start: Block number at chunk start (for verification)
        block_end: Block number at chunk end (for verification)
    """
    chunk_start_ms: int
    chunk_end_ms: int
    chunk_emissions: float
    cumulative_emissions: float
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
            'block_start': self.block_start,
            'block_end': self.block_end,
        }


class EmissionsLedger:
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

    # Default start date for emissions tracking: September 1, 2025 00:00:00 UTC
    DEFAULT_START_DATE_MS = int(datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    def __init__(self, network: str = "finney", netuid: int = 8):
        """
        Initialize EmissionsLedger with blockchain connection.

        Args:
            network: Bittensor network name ("finney", "test", "local")
            netuid: Subnet UID to query (default: 8 for mainnet PTN)
        """
        self.network = network
        self.netuid = netuid

        # Initialize subtensor connection
        bt.logging.info(f"Connecting to network: {network}, netuid: {netuid}")
        self.subtensor = bt.subtensor(network=network)

        # Storage for emissions checkpoints per hotkey
        self.emissions_ledgers: Dict[str, List[EmissionsCheckpoint]] = {}

        bt.logging.info("EmissionsLedger initialized")

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

    def _check_subnet_exists(self) -> bool:
        """
        Check if the subnet exists on the chain.

        Returns:
            True if subnet exists, False otherwise
        """
        try:
            # Query SubnetworkN to see if subnet is registered
            result = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='SubnetworkN',
                params=[self.netuid]
            )

            if result is None:
                bt.logging.warning(f"Netuid {self.netuid} does not exist on chain")
                return False

            n = int(result.value if hasattr(result, 'value') else result)
            bt.logging.info(f"Netuid {self.netuid} exists with {n} max UIDs")
            return n > 0

        except Exception as e:
            bt.logging.error(f"Error checking if subnet exists: {e}")
            return False

    def _get_uid_for_hotkey_from_chain(self, hotkey: str) -> Optional[int]:
        """
        Query substrate storage directly to find UID for a hotkey.
        This is a fallback when metagraph API is not available.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            UID if found, None otherwise
        """
        try:
            # Query SubnetworkN storage to get max UIDs
            max_uids_result = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='SubnetworkN',
                params=[self.netuid]
            )

            bt.logging.debug(f"SubnetworkN query result: {max_uids_result} (type: {type(max_uids_result)})")

            if max_uids_result is None:
                bt.logging.warning(f"Could not query SubnetworkN for netuid {self.netuid}")
                return None

            max_uids = int(max_uids_result.value if hasattr(max_uids_result, 'value') else max_uids_result)
            bt.logging.debug(f"Parsed max_uids: {max_uids}")

            bt.logging.info(f"Searching for hotkey {hotkey} in {max_uids} UIDs for netuid {self.netuid}")

            # Iterate through all UIDs to find matching hotkey
            found_hotkeys = []
            for uid in range(max_uids):
                try:
                    keys_result = self.subtensor.substrate.query(
                        module='SubtensorModule',
                        storage_function='Keys',
                        params=[self.netuid, uid]
                    )

                    if keys_result is not None:
                        stored_hotkey = keys_result.value if hasattr(keys_result, 'value') else str(keys_result)
                        found_hotkeys.append((uid, stored_hotkey))

                        bt.logging.debug(f"UID {uid}: {stored_hotkey}")

                        if stored_hotkey == hotkey:
                            bt.logging.info(f"Found UID {uid} for hotkey {hotkey}")
                            return uid
                except Exception as e:
                    bt.logging.debug(f"Error querying UID {uid}: {e}")
                    continue

            # Log what we found for debugging
            bt.logging.warning(f"Hotkey {hotkey} not found in subnet {self.netuid}")
            bt.logging.info(f"Found {len(found_hotkeys)} total hotkeys in subnet")
            if found_hotkeys:
                bt.logging.info(f"First few hotkeys: {found_hotkeys[:3]}")
                bt.logging.info(f"Last few hotkeys: {found_hotkeys[-3:]}")
            return None

        except Exception as e:
            bt.logging.error(f"Error searching for hotkey UID: {e}")
            return None

    def _get_registration_block_from_chain(self, uid: int) -> Optional[int]:
        """
        Query registration block directly from substrate storage.

        Args:
            uid: The UID to query

        Returns:
            Block number at registration, or None if not found
        """
        try:
            block_result = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='BlockAtRegistration',
                params=[self.netuid, uid]
            )

            if block_result is not None:
                block = int(block_result.value if hasattr(block_result, 'value') else block_result)
                return block

            return None

        except Exception as e:
            bt.logging.error(f"Error querying registration block for UID {uid}: {e}")
            return None

    def get_registration_block(self, hotkey: str) -> Optional[int]:
        """
        Get the block number when a hotkey was registered on the subnet.

        First tries to use metagraph API, then falls back to direct substrate queries
        if the API is not available (e.g., on archive nodes).

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            Block number at registration, or None if not registered
        """
        try:
            # Try metagraph API first (faster)
            metagraph = self.subtensor.metagraph(netuid=self.netuid)

            if hotkey not in metagraph.hotkeys:
                bt.logging.warning(f"Hotkey {hotkey} not found in subnet {self.netuid}")
                return None

            uid = metagraph.hotkeys.index(hotkey)
            registration_block = metagraph.block_at_registration[uid]

            bt.logging.info(f"Hotkey {hotkey} registered at block {registration_block}")
            return int(registration_block)

        except Exception as e:
            bt.logging.warning(f"Metagraph API unavailable ({e}), using direct chain queries")

            # Check if subnet exists
            if not self._check_subnet_exists():
                bt.logging.error(f"Netuid {self.netuid} does not exist or has no UIDs. "
                               f"Make sure your local subtensor has netuid {self.netuid} configured.")
                return None

            # Fallback to direct substrate storage queries
            uid = self._get_uid_for_hotkey_from_chain(hotkey)
            if uid is None:
                bt.logging.error(f"Could not find UID for hotkey {hotkey}")
                return None

            registration_block = self._get_registration_block_from_chain(uid)
            if registration_block is not None:
                bt.logging.info(f"Hotkey {hotkey} (UID {uid}) registered at block {registration_block}")
                return registration_block
            else:
                bt.logging.error(f"Could not determine registration block for {hotkey}")
                return None

    def get_block_timestamp(self, block_number: int) -> Optional[int]:
        """
        Get the timestamp (in milliseconds) for a specific block.

        Args:
            block_number: Block number to query

        Returns:
            Timestamp in milliseconds, or None if query fails
        """
        try:
            # Query block info from substrate
            block_hash = self.subtensor.substrate.get_block_hash(block_number)
            if not block_hash:
                return None

            block = self.subtensor.substrate.get_block(block_hash=block_hash)
            if not block or 'extrinsics' not in block:
                return None

            # Get timestamp from block
            # Substrate stores timestamps in milliseconds in the Timestamp pallet
            timestamp_call = self.subtensor.substrate.query(
                module='Timestamp',
                storage_function='Now',
                block_hash=block_hash
            )

            if timestamp_call:
                return int(timestamp_call.value)
            else:
                raise ValueError("Timestamp not found in block")

            bt.logging.debug(f"Using estimated timestamp for block {block_number}")
            return estimated_timestamp_ms

        except Exception as e:
            bt.logging.error(f"Error getting timestamp for block {block_number}: {e}")
            return None

    def query_emissions_at_block(self, hotkey: str, block_number: int) -> Optional[float]:
        """
        Query the emissions rate for a hotkey at a specific block.

        This queries the historical state of the metagraph at the given block
        to determine the emissions the hotkey was receiving at that time.

        Args:
            hotkey: SS58 address of the hotkey
            block_number: Block number to query

        Returns:
            Emissions per block (in TAO), or None if query fails
        """
        try:
            # Get block hash for the specific block
            block_hash = self.subtensor.substrate.get_block_hash(block_number)
            if not block_hash:
                bt.logging.warning(f"Could not get hash for block {block_number}")
                return None

            # Query emission for this hotkey at this block
            # The SubtensorModule stores emission data in the Emission storage
            emission_query = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='Emission',
                params=[self.netuid, hotkey],
                block_hash=block_hash
            )

            if emission_query is not None:
                # Emission is stored in RAO (1 TAO = 10^9 RAO)
                emission_rao = float(emission_query.value) if hasattr(emission_query, 'value') else float(emission_query)
                emission_tao = emission_rao / 1e9
                return emission_tao

            return 0.0

        except Exception as e:
            bt.logging.error(f"Error querying emissions at block {block_number} for {hotkey}: {e}")
            return None

    def calculate_emissions_in_range(self, hotkey: str, start_block: int, end_block: int) -> float:
        """
        Calculate total emissions received by a hotkey between two blocks.

        This method samples emissions at regular intervals and estimates total
        emissions received during the block range.

        Args:
            hotkey: SS58 address of the hotkey
            start_block: Starting block (inclusive)
            end_block: Ending block (inclusive)

        Returns:
            Total emissions (in TAO) received during the block range
        """
        total_emissions = 0.0

        # Sample emissions at regular intervals to estimate total
        # We'll sample every ~1 hour of blocks for accuracy vs performance balance
        sample_interval = int(3600 / self.SECONDS_PER_BLOCK)  # ~300 blocks per hour

        sampled_blocks = list(range(start_block, end_block + 1, sample_interval))
        if sampled_blocks[-1] != end_block:
            sampled_blocks.append(end_block)

        bt.logging.info(f"Sampling {len(sampled_blocks)} blocks between {start_block} and {end_block}")

        previous_emission_rate = None
        previous_block = start_block

        for block in sampled_blocks:
            emission_rate = self.query_emissions_at_block(hotkey, block)

            if emission_rate is None:
                bt.logging.warning(f"Could not query emissions at block {block}, using previous rate")
                emission_rate = previous_emission_rate if previous_emission_rate is not None else 0.0

            if previous_emission_rate is not None:
                # Average the two rates and multiply by blocks elapsed
                blocks_elapsed = block - previous_block
                avg_rate = (previous_emission_rate + emission_rate) / 2
                chunk_emissions = avg_rate * blocks_elapsed
                total_emissions += chunk_emissions

            previous_emission_rate = emission_rate
            previous_block = block

        return total_emissions

    def build_emissions_ledger_for_hotkey(
        self,
        hotkey: str,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        verbose: bool = False
    ) -> List[EmissionsCheckpoint]:
        """
        Build emissions ledger for a single hotkey.

        Queries historical blockchain data to construct a complete emissions history
        organized into 12-hour UTC chunks.

        Args:
            hotkey: SS58 address of the hotkey
            start_time_ms: Optional start time (default: registration time)
            end_time_ms: Optional end time (default: current time)
            verbose: Enable detailed logging

        Returns:
            List of EmissionsCheckpoints chronologically ordered
        """
        bt.logging.info(f"Building emissions ledger for hotkey: {hotkey}")

        # Use default start date (Sept 1, 2025) if not specified
        if start_time_ms is None:
            start_time_ms = self.DEFAULT_START_DATE_MS
            bt.logging.info(f"Using default start date: {datetime.fromtimestamp(start_time_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # Estimate block number from default start time
            current_block = self.subtensor.get_current_block()
            current_time_ms = int(time.time() * 1000)
            seconds_diff = (current_time_ms - start_time_ms) / 1000
            blocks_diff = int(seconds_diff / self.SECONDS_PER_BLOCK)
            start_block = current_block - blocks_diff

            bt.logging.info(f"Estimated start block: {start_block}")
        else:
            # Estimate block number from timestamp
            current_block = self.subtensor.get_current_block()
            current_time_ms = int(time.time() * 1000)
            seconds_diff = (current_time_ms - start_time_ms) / 1000
            blocks_diff = int(seconds_diff / self.SECONDS_PER_BLOCK)
            start_block = current_block - blocks_diff

        # Default end time is now
        if end_time_ms is None:
            end_time_ms = int(time.time() * 1000)
            end_block = self.subtensor.get_current_block()
        else:
            # Estimate block number from timestamp
            current_block = self.subtensor.get_current_block()
            current_time_ms = int(time.time() * 1000)
            seconds_diff = (current_time_ms - end_time_ms) / 1000
            blocks_diff = int(seconds_diff / self.SECONDS_PER_BLOCK)
            end_block = current_block - blocks_diff

        bt.logging.info(f"Tracking emissions from {start_time_ms} (block {start_block}) to {end_time_ms} (block {end_block})")

        # Generate list of 12-hour chunks
        checkpoints: List[EmissionsCheckpoint] = []
        cumulative_emissions = 0.0

        current_chunk_start_ms, current_chunk_end_ms = self.get_chunk_boundaries(start_time_ms)

        # If start time is not at chunk boundary, adjust to next chunk
        if current_chunk_start_ms < start_time_ms:
            current_chunk_start_ms = current_chunk_end_ms
            current_chunk_end_ms = current_chunk_start_ms + self.CHUNK_DURATION_MS

        chunk_count = 0
        while current_chunk_start_ms < end_time_ms:
            chunk_count += 1

            # Determine actual time range for this chunk (clipped to start/end)
            actual_start_ms = max(current_chunk_start_ms, start_time_ms)
            actual_end_ms = min(current_chunk_end_ms, end_time_ms)

            # Calculate corresponding blocks
            seconds_from_start = (actual_start_ms - start_time_ms) / 1000
            seconds_from_end = (actual_end_ms - start_time_ms) / 1000
            chunk_start_block = start_block + int(seconds_from_start / self.SECONDS_PER_BLOCK)
            chunk_end_block = start_block + int(seconds_from_end / self.SECONDS_PER_BLOCK)

            if verbose:
                bt.logging.info(f"Processing chunk {chunk_count}: {actual_start_ms} to {actual_end_ms}")

            # Calculate emissions for this chunk
            chunk_emissions = self.calculate_emissions_in_range(
                hotkey,
                chunk_start_block,
                chunk_end_block
            )

            cumulative_emissions += chunk_emissions

            checkpoint = EmissionsCheckpoint(
                chunk_start_ms=current_chunk_start_ms,
                chunk_end_ms=current_chunk_end_ms,
                chunk_emissions=chunk_emissions,
                cumulative_emissions=cumulative_emissions,
                block_start=chunk_start_block,
                block_end=chunk_end_block
            )

            checkpoints.append(checkpoint)

            if verbose:
                bt.logging.info(f"Chunk {chunk_count}: {chunk_emissions:.6f} TAO (cumulative: {cumulative_emissions:.6f} TAO)")

            # Move to next chunk
            current_chunk_start_ms = current_chunk_end_ms
            current_chunk_end_ms = current_chunk_start_ms + self.CHUNK_DURATION_MS

        bt.logging.info(f"Built {len(checkpoints)} emission checkpoints for {hotkey}")
        bt.logging.info(f"Total emissions: {cumulative_emissions:.6f} TAO")

        # Store in ledger
        self.emissions_ledgers[hotkey] = checkpoints

        return checkpoints

    def _get_all_hotkeys_from_chain(self) -> List[str]:
        """
        Query all hotkeys from subnet using direct substrate storage queries.
        This is a fallback when metagraph API is not available.

        Returns:
            List of hotkey SS58 addresses
        """
        try:
            # Query SubnetworkN storage to get max UIDs
            max_uids_result = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='SubnetworkN',
                params=[self.netuid]
            )

            if max_uids_result is None:
                bt.logging.error(f"Could not query SubnetworkN for netuid {self.netuid}")
                return []

            max_uids = int(max_uids_result.value if hasattr(max_uids_result, 'value') else max_uids_result)
            bt.logging.info(f"Querying {max_uids} UIDs from chain")

            hotkeys = []
            for uid in range(max_uids):
                try:
                    keys_result = self.subtensor.substrate.query(
                        module='SubtensorModule',
                        storage_function='Keys',
                        params=[self.netuid, uid]
                    )

                    if keys_result is not None:
                        hotkey = keys_result.value if hasattr(keys_result, 'value') else str(keys_result)
                        if hotkey:
                            hotkeys.append(hotkey)
                except Exception as e:
                    bt.logging.debug(f"Error querying UID {uid}: {e}")
                    continue

            bt.logging.info(f"Found {len(hotkeys)} hotkeys from chain queries")
            return hotkeys

        except Exception as e:
            bt.logging.error(f"Error getting hotkeys from chain: {e}")
            return []

    def build_emissions_ledgers_bulk(
        self,
        hotkeys: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, List[EmissionsCheckpoint]]:
        """
        Build emissions ledgers for multiple hotkeys efficiently.

        This method queries the metagraph once and then processes all hotkeys,
        which is more efficient than building ledgers individually.

        Args:
            hotkeys: List of hotkeys to process (default: all hotkeys in subnet)
            verbose: Enable detailed logging

        Returns:
            Dictionary mapping hotkeys to their emissions checkpoints
        """
        bt.logging.info("Building emissions ledgers in bulk")

        # Get all hotkeys from subnet if not specified
        if hotkeys is None:
            try:
                # Try metagraph API first
                metagraph = self.subtensor.metagraph(netuid=self.netuid)
                hotkeys = list(metagraph.hotkeys)
                bt.logging.info(f"Processing all {len(hotkeys)} hotkeys in subnet {self.netuid}")
            except Exception as e:
                bt.logging.warning(f"Metagraph API unavailable ({e}), using direct chain queries")
                # Fallback to direct chain queries
                hotkeys = self._get_all_hotkeys_from_chain()
                if not hotkeys:
                    bt.logging.error("Could not retrieve hotkeys from chain")
                    return {}
                bt.logging.info(f"Processing all {len(hotkeys)} hotkeys in subnet {self.netuid}")
        else:
            bt.logging.info(f"Processing {len(hotkeys)} specified hotkeys")

        # Process each hotkey
        for i, hotkey in enumerate(hotkeys, 1):
            bt.logging.info(f"Processing hotkey {i}/{len(hotkeys)}: {hotkey}")

            try:
                self.build_emissions_ledger_for_hotkey(hotkey, verbose=verbose)
            except Exception as e:
                bt.logging.error(f"Error processing hotkey {hotkey}: {e}")
                continue

        bt.logging.info(f"Completed bulk processing of {len(self.emissions_ledgers)} hotkeys")

        return self.emissions_ledgers

    def get_emissions_ledger(self, hotkey: str) -> List[EmissionsCheckpoint]:
        """
        Get the emissions ledger for a specific hotkey.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            List of EmissionsCheckpoints, or empty list if not found
        """
        return self.emissions_ledgers.get(hotkey, [])

    def get_cumulative_emissions(self, hotkey: str) -> float:
        """
        Get the total cumulative emissions for a hotkey.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            Total emissions in TAO
        """
        checkpoints = self.get_emissions_ledger(hotkey)
        if not checkpoints:
            return 0.0
        return checkpoints[-1].cumulative_emissions

    def print_emissions_summary(self, hotkey: str):
        """
        Print a formatted summary of emissions for a hotkey.

        Args:
            hotkey: SS58 address of the hotkey
        """
        checkpoints = self.get_emissions_ledger(hotkey)

        if not checkpoints:
            print(f"\nNo emissions data found for {hotkey}")
            return

        print(f"\n{'='*80}")
        print(f"Emissions Summary for {hotkey}")
        print(f"{'='*80}")
        print(f"Total Checkpoints: {len(checkpoints)}")
        print(f"Total Emissions: {checkpoints[-1].cumulative_emissions:.6f} TAO")
        print(f"\nFirst 5 Checkpoints:")
        print(f"{'Chunk Start (UTC)':<25} {'Chunk End (UTC)':<25} {'Chunk TAO':>15} {'Cumulative TAO':>15}")
        print(f"{'-'*80}")

        for checkpoint in checkpoints[:5]:
            start_dt = datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(checkpoint.chunk_end_ms / 1000, tz=timezone.utc)
            print(f"{start_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                  f"{end_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                  f"{checkpoint.chunk_emissions:>15.6f} "
                  f"{checkpoint.cumulative_emissions:>15.6f}")

        if len(checkpoints) > 10:
            print(f"{'...':<25} {'...':<25} {'...':>15} {'...':>15}")
            print(f"\nLast 5 Checkpoints:")
            print(f"{'Chunk Start (UTC)':<25} {'Chunk End (UTC)':<25} {'Chunk TAO':>15} {'Cumulative TAO':>15}")
            print(f"{'-'*80}")

            for checkpoint in checkpoints[-5:]:
                start_dt = datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc)
                end_dt = datetime.fromtimestamp(checkpoint.chunk_end_ms / 1000, tz=timezone.utc)
                print(f"{start_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                      f"{end_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                      f"{checkpoint.chunk_emissions:>15.6f} "
                      f"{checkpoint.cumulative_emissions:>15.6f}")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build emissions ledger for Bittensor hotkeys")
    parser.add_argument("--hotkey", type=str, help="Hotkey to query (SS58 address)", default='5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY')
    parser.add_argument("--netuid", type=int, default=8, help="Subnet UID (default: 8)")
    parser.add_argument("--network", type=str, default="finney", help="Network name (default: finney)")
    parser.add_argument("--bulk", action="store_true", help="Process all hotkeys in subnet")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    bt.logging.enable_info()
    if args.verbose:
        bt.logging.enable_debug()

    # Initialize ledger
    ledger = EmissionsLedger(network=args.network, netuid=args.netuid)

    if args.bulk:
        # Build ledgers for all hotkeys
        ledger.build_emissions_ledgers_bulk(verbose=args.verbose)

        # Print summaries for all
        for hotkey in ledger.emissions_ledgers.keys():
            ledger.print_emissions_summary(hotkey)

    elif args.hotkey:
        # Build ledger for single hotkey
        ledger.build_emissions_ledger_for_hotkey(args.hotkey, verbose=args.verbose)
        ledger.print_emissions_summary(args.hotkey)

    else:
        print("Error: Must specify --hotkey or --bulk")
        parser.print_help()
