import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

import bittensor as bt
from bittensor.utils.balance import Balance

from collateral_sdk import CollateralManager, Network

from vali_objects.vali_config import ValiConfig

class CollateralRecord:
    def __init__(self, account_size, update_time_ms):
        self.account_size = account_size
        self.update_time_ms = update_time_ms
        self.valid_date_timestamp = CollateralRecord.valid_from_ms(update_time_ms)

    @staticmethod
    def valid_from_ms(update_time_ms) -> int:
        """Returns timestamp of start of day (00:00:00 UTC) when this record is valid"""
        dt = datetime.fromtimestamp(update_time_ms / 1000, tz=timezone.utc)
        start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(start_of_day.timestamp() * 1000)

    @property
    def valid_date_str(self) -> str:
        """Returns YYYY-MM-DD format for easy reading"""
        dt = datetime.fromtimestamp(self.valid_date_timestamp / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")

class ContractManager:
    """
    Manages miner collateral through smart contracts integration.
    
    Responsibilities:
    - Read miner deposited collateral from smart contracts
    - Track miner account sizes and capital allocation
    - Calculate withdrawal eligibility based on drawdowns
    - Provide interface for collateral operations
    """
    
    def __init__(
        self,
        network = None,
        program_address: Optional[str] = None,
        owner_address: Optional[str] = None,
        owner_private_key: Optional[str] = None,
        data_dir: Optional[str] = None
    ):
        """
        Initialize ContractManager.
        
        Args:
            network: Network to use (TESTNET, MAINNET, LOCAL)
            program_address: Smart contract address override
            owner_address: Owner address for contract operations
            owner_private_key: Owner private key for transactions
            data_dir: Directory to store miner account data
        """
        self.collateral_manager = CollateralManager(
            network=network,
            program_address=program_address
        )
        
        self.owner_address = owner_address
        self.owner_private_key = owner_private_key
        self.network = network
        
        # Data storage
        self.data_dir = Path(data_dir or ValiConfig.BASE_DIR) / "collateral_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.miner_account_sizes_file = self.data_dir / "miner_account_sizes.pkl"

        # In-memory cache for miner data
        self.miner_account_sizes: Dict[str, List[CollateralRecord]] = {}  # hotkey -> List[CollateralRecord] sorted by updated_time_ms
        self.miner_collateral_cache: Dict[str, Tuple[int, float]] = {}  # hotkey -> (balance_theta, timestamp)
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_account_size_refresh = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_account_sizes()
        
        bt.logging.info(f"ContractManager initialized for network: {network.name}")

    def refresh_account_sizes(self, timestamp_ms: int, hotkeys: list[str] = None) -> None:
        """
        Refresh account sizes by checking if updates are needed and cleaning up old records.

        Args:
            timestamp_ms: Reference timestamp in milliseconds to check against
            hotkeys: Optional list of hotkeys to refresh. If None, refreshes all tracked miners.
        """
        with self._lock:
            target_date_timestamp = CollateralRecord.valid_from_ms(timestamp_ms)
            target_date_str = datetime.fromtimestamp(target_date_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            last_refresh_date_str = datetime.fromtimestamp(self.last_account_size_refresh / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if last_refresh_date_str >= target_date_str:
                bt.logging.info(f"Account sizes already refreshed for {target_date_str}, skipping refresh") # TODO, this may introduce too many logs
                return

            bt.logging.info(f"Starting account size refresh for date: {target_date_str}")

            self._load_account_sizes()

            miners_to_check = hotkeys if hotkeys is not None else list(self.miner_account_sizes.keys()) # TODO, get hotkeys from elsewhere and add validation

            updated_count = 0
            cleaned_count = 0

            for hotkey in miners_to_check:
                if self._needs_account_size_update(hotkey, target_date_timestamp):
                    if self._update_miner_account_size(hotkey, timestamp_ms):
                        updated_count += 1

                cutoff_ms = timestamp_ms - (120 * 24 * 60 * 60 * 1000)
                records_removed = self._cleanup_old_records(hotkey, cutoff_ms)
                cleaned_count += records_removed

            self._save_account_sizes()
            bt.logging.info(f"Account size refresh completed - Updated: {updated_count}, Cleaned: {cleaned_count}")
    
    def _load_account_sizes(self) -> None:
        """Load miner account sizes from disk."""
        try:
            if self.miner_account_sizes_file.exists():
                with open(self.miner_account_sizes_file, 'rb') as f:
                    self.miner_account_sizes = pickle.load(f)
                bt.logging.info(f"Loaded {len(self.miner_account_sizes)} miner account sizes")
        except Exception as e:
            bt.logging.error(f"Failed to load account sizes: {e}")
            self.miner_account_sizes = {}
    
    def _save_account_sizes(self) -> None:
        """Save miner account sizes to disk."""
        try:
            with open(self.miner_account_sizes_file, 'wb') as f:
                pickle.dump(self.miner_account_sizes, f)
        except Exception as e:
            bt.logging.error(f"Failed to save account sizes: {e}")

    def _cleanup_old_records(self, hotkey: str, cutoff_timestamp_ms: int) -> int:
        """
        Remove old collateral records for a miner.
        
        Args:
            hotkey: Miner's hotkey
            cutoff_timestamp_ms: Records older than this timestamp will be removed
            
        Returns:
            Number of records removed
        """
        if hotkey not in self.miner_account_sizes or not self.miner_account_sizes[hotkey]:
            return 0
            
        cutoff_date_timestamp = CollateralRecord.valid_from_ms(cutoff_timestamp_ms)
        original_count = len(self.miner_account_sizes[hotkey])
        
        self.miner_account_sizes[hotkey] = [
            record for record in self.miner_account_sizes[hotkey]
            if record.valid_date_timestamp >= cutoff_date_timestamp
        ]
        
        return original_count - len(self.miner_account_sizes[hotkey])

    def _needs_account_size_update(self, hotkey: str, target_date_timestamp: int) -> bool:
        """
        Check if a miner's account size needs updating.
        
        Args:
            hotkey: Miner's hotkey
            target_date_timestamp: Target date timestamp to check against
            
        Returns:
            True if update is needed
        """
        if hotkey not in self.miner_account_sizes or not self.miner_account_sizes[hotkey]:
            return True
            
        latest_record = max(self.miner_account_sizes[hotkey], key=lambda r: r.valid_date_timestamp)
        return latest_record.valid_date_timestamp < target_date_timestamp

    def _update_miner_account_size(self, hotkey: str, timestamp_ms: int) -> bool:
        """
        Update account size for a single miner.
        
        Args:
            hotkey: Miner's hotkey
            timestamp_ms: Timestamp for the new record
            
        Returns:
            True if update was successful
        """
        current_balance_theta = self.get_miner_collateral_balance(hotkey, use_cache=False)
        account_size = float(current_balance_theta)
        
        new_record = CollateralRecord(account_size, timestamp_ms)
        
        if hotkey not in self.miner_account_sizes:
            self.miner_account_sizes[hotkey] = []
        
        self.miner_account_sizes[hotkey].append(new_record)
        self.miner_account_sizes[hotkey].sort(key=lambda r: r.update_time_ms)
        
        bt.logging.debug(f"Updated account size for {hotkey}: {account_size}")
        return True
    
    def get_miner_collateral_balance(self, hotkey: str, use_cache: bool = True) -> int:
        """
        Get miner's deposited collateral balance in theta.
        
        Args:
            hotkey: Miner's hotkey (SS58 address)
            use_cache: Whether to use cached values if available
            
        Returns:
            Collateral balance in theta
        """
        with self._lock:
            # Check cache first
            if use_cache and hotkey in self.miner_collateral_cache:
                balance, timestamp = self.miner_collateral_cache[hotkey]
                if time.time() - timestamp < self.cache_ttl:
                    return balance

            try:
                balance = self.collateral_manager.balance_of(hotkey)
                bt.logging.debug(f"Fetched collateral balance for {hotkey}: {balance} theta")

                # Update cache
                self.miner_collateral_cache[hotkey] = (balance, time.time())
                return balance

            except Exception as e:
                bt.logging.error(f"Failed to get collateral balance for {hotkey}: {e}")
                # Return cached value if available, otherwise 0
                if hotkey in self.miner_collateral_cache:
                    balance, _ = self.miner_collateral_cache[hotkey]
                    return balance
                return 0

    def get_miner_collateral_balance_tao(self, hotkey: str, use_cache: bool = True) -> float:
        """
        Get miner's deposited collateral balance in TAO.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            use_cache: Whether to use cached values if available

        Returns:
            Collateral balance in TAO
        """
        balance_theta = self.get_miner_collateral_balance(hotkey, use_cache)
        return balance_theta

    def set_miner_account_size(self, hotkey: str, account_size: float) -> None:
        """
        Set the account size for a miner.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            account_size: Account size in USD
        """
        with self._lock:
            current_time_ms = int(time.time() * 1000)
            record = CollateralRecord(account_size, current_time_ms)

            if hotkey not in self.miner_account_sizes:
                self.miner_account_sizes[hotkey] = []

            self.miner_account_sizes[hotkey].append(record)
            # Keep list sorted by updated_time_ms
            self.miner_account_sizes[hotkey].sort(key=lambda r: r.update_time_ms)

            self._save_account_sizes()
            bt.logging.debug(f"Set account size for {hotkey}: ${account_size:,.2f}")

    def get_miner_account_size(self, hotkey: str) -> float:
        """
        Get the account size for a miner.
        
        Args:
            hotkey: Miner's hotkey (SS58 address)
            
        Returns:
            Account size in USD, defaults to ValiConfig.CAPITAL if not set
        """
        with self._lock:
            if hotkey in self.miner_account_sizes and self.miner_account_sizes[hotkey]:
                # Get the latest record (sorted by updated_time_ms)
                latest_record = max(self.miner_account_sizes[hotkey], key=lambda r: r.update_time_ms)
                return latest_record.account_size
            return ValiConfig.CAPITAL

    def calculate_capital_allocation(self, hotkey: str, tao_price_usd: float) -> Tuple[float, float]:
        """
        Calculate miner's capital allocation based on collateral and account size.
        
        Args:
            hotkey: Miner's hotkey (SS58 address)
            tao_price_usd: Current TAO price in USD
            
        Returns:
            Tuple of (allocated_capital_usd, collateral_ratio)
        """
        collateral_tao = self.get_miner_collateral_balance_tao(hotkey)
        collateral_usd = collateral_tao * tao_price_usd
        account_size = self.get_miner_account_size(hotkey)
        
        # Use minimum of collateral value and configured account size
        allocated_capital = min(collateral_usd, account_size)
        collateral_ratio = collateral_usd / account_size if account_size > 0 else 0
        
        bt.logging.debug(
            f"Capital allocation for {hotkey}: "
            f"collateral={collateral_usd:.2f} USD, "
            f"account_size={account_size:.2f} USD, "
            f"allocated={allocated_capital:.2f} USD, "
            f"ratio={collateral_ratio:.3f}"
        )
        
        return allocated_capital, collateral_ratio
    
    def calculate_withdrawable_amount(
        self, 
        hotkey: str,
        pnl: float,
        tao_price_usd: float,
        safety_margin: float = 0.1
    ) -> Tuple[float, float]:
        """
        Calculate how much collateral a miner can withdraw based on drawdowns.
        
        Args:
            hotkey: Miner's hotkey (SS58 address)
            pnl: Miner's unrealized and realized pnl
            tao_price_usd: Current TAO price in USD
            safety_margin: Additional safety margin (default 10%)
            
        Returns:
            Tuple of (withdrawable_tao, required_collateral_tao)
        """
        collateral_tao = self.get_miner_collateral_balance_tao(hotkey)
        account_size = self.get_miner_account_size(hotkey)
        
        # Calculate maximum drawdown as a percentage
        max_drawdown_pct = abs(pnl / account_size) if account_size > 0 else 0
        
        # Calculate required collateral based on drawdown
        # Use the worse of daily or total drawdown limits
        max_allowed_drawdown = min(
            1 - ValiConfig.MAX_DAILY_DRAWDOWN,
            1 - ValiConfig.MAX_TOTAL_DRAWDOWN_V2
        )
        
        # Required collateral should cover potential losses plus safety margin
        required_collateral_usd = account_size * (max_allowed_drawdown + safety_margin)
        required_collateral_tao = required_collateral_usd / tao_price_usd if tao_price_usd > 0 else 0
        
        # Calculate withdrawable amount
        withdrawable_tao = max(0, collateral_tao - required_collateral_tao)
        
        bt.logging.debug(
            f"Withdrawal calculation for {hotkey}: "
            f"collateral={collateral_tao:.6f} TAO, "
            f"required={required_collateral_tao:.6f} TAO, "
            f"withdrawable={withdrawable_tao:.6f} TAO, "
            f"drawdown={max_drawdown_pct:.3%}"
        )
        
        return withdrawable_tao, required_collateral_tao
    
    def is_miner_adequately_collateralized(
        self, 
        hotkey: str, 
        pnl: float,
        tao_price_usd: float,
        min_collateral_ratio: float = 1.2
    ) -> bool:
        """
        Check if miner has adequate collateral for their current position.
        
        Args:
            hotkey: Miner's hotkey (SS58 address)
            pnl: Miner's unrealized and realized pnl
            tao_price_usd: Current TAO price in USD
            min_collateral_ratio: Minimum collateral ratio required
            
        Returns:
            True if adequately collateralized
        """
        _, required_collateral_tao = self.calculate_withdrawable_amount(
            hotkey, pnl, tao_price_usd
        )
        
        collateral_tao = self.get_miner_collateral_balance_tao(hotkey)
        current_ratio = collateral_tao / required_collateral_tao if required_collateral_tao > 0 else float('inf')
        
        is_adequate = current_ratio >= min_collateral_ratio
        
        bt.logging.debug(
            f"Collateral check for {hotkey}: "
            f"ratio={current_ratio:.3f}, "
            f"required={min_collateral_ratio:.3f}, "
            f"adequate={is_adequate}"
        )
        
        return is_adequate
    
    def get_total_collateral(self) -> int:
        """Get total collateral in the contract in theta."""
        try:
            return self.collateral_manager.get_total_collateral()
        except Exception as e:
            bt.logging.error(f"Failed to get total collateral: {e}")
            return 0
    
    def get_slashed_collateral(self) -> int:
        """Get total slashed collateral in theta."""
        try:
            return self.collateral_manager.get_slashed_collateral()
        except Exception as e:
            bt.logging.error(f"Failed to get slashed collateral: {e}")
            return 0
    
    def slash_miner_collateral(
        self, 
        hotkey: str, 
        amount_tao: float,
        reason: str = "Performance violation"
    ) -> bool:
        """
        Slash collateral from a miner (owner only).
        
        Args:
            hotkey: Miner's hotkey (SS58 address)
            amount_tao: Amount to slash in TAO
            reason: Reason for slashing
            
        Returns:
            True if successful
        """
        if not self.owner_address or not self.owner_private_key:
            bt.logging.error("Owner credentials not configured for slashing")
            return False
        
        try:
            amount_theta = Balance.from_tao(amount_tao).theta
            slashed_balance = self.collateral_manager.slash(
                address=hotkey,
                amount=amount_theta,
                owner_address=self.owner_address,
                owner_private_key=self.owner_private_key
            )
            
            # Clear cache for this miner
            if hotkey in self.miner_collateral_cache:
                del self.miner_collateral_cache[hotkey]
            
            bt.logging.warning(
                f"Slashed {slashed_balance.tao:.6f} TAO from {hotkey}. Reason: {reason}"
            )
            return True
            
        except Exception as e:
            bt.logging.error(f"Failed to slash collateral from {hotkey}: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the collateral balance cache."""
        with self._lock:
            self.miner_collateral_cache.clear()
            bt.logging.info("Cleared collateral cache")
    
    def get_cached_miners(self) -> list[str]:
        """Get list of miners with cached collateral data."""
        with self._lock:
            return list(self.miner_collateral_cache.keys())
    
    def get_all_miner_stats(self, tao_price_usd: float) -> Dict[str, Dict]:
        """
        Get comprehensive stats for all tracked miners.
        
        Args:
            tao_price_usd: Current TAO price in USD
            
        Returns:
            Dictionary mapping hotkey to stats dict
        """
        stats = {}
        
        with self._lock:
            all_hotkeys = set(self.miner_account_sizes.keys()) | set(self.miner_collateral_cache.keys())
            
            for hotkey in all_hotkeys:
                collateral_tao = self.get_miner_collateral_balance_tao(hotkey, use_cache=True)
                account_size = self.get_miner_account_size(hotkey)
                allocated_capital, collateral_ratio = self.calculate_capital_allocation(hotkey, tao_price_usd)

                stats[hotkey] = {
                    'collateral_tao': collateral_tao,
                    'collateral_usd': collateral_tao * tao_price_usd,
                    'account_size_usd': account_size,
                    'allocated_capital_usd': allocated_capital,
                    'collateral_ratio': collateral_ratio
                }

        return stats

    def get_recent_account_sizes(self, hotkeys: list[str] = None, timestamp_ms: int = None, refresh_to_current_time: bool = False) -> Dict[str, float]:
        with self._lock:
            self._load_account_sizes()

            if hotkeys is None:
                hotkeys = list(self.miner_account_sizes.keys())

            # Use current time if no timestamp provided
            if timestamp_ms is None:
                timestamp_ms = int(time.time() * 1000)
            
            # Get target day timestamp for validation
            target_day_timestamp = CollateralRecord.valid_from_ms(timestamp_ms)

            # Check which hotkeys need updates
            hotkeys_needing_update = []
            for hotkey in hotkeys:
                if self._needs_account_size_update(hotkey, target_day_timestamp):
                    hotkeys_needing_update.append(hotkey)

            # Refresh account sizes for hotkeys that need updates
            if hotkeys_needing_update:
                self.refresh_account_sizes(timestamp_ms, hotkeys_needing_update)

            # Now collect the results
            result = {}
            for hotkey in hotkeys:
                if hotkey in self.miner_account_sizes and self.miner_account_sizes[hotkey]:
                    # Get the latest record
                    latest_record = max(self.miner_account_sizes[hotkey], key=lambda r: r.valid_date_timestamp)
                    
                    # Only include if the record is up-to-date for target day
                    if latest_record.valid_date_timestamp >= target_day_timestamp:
                        result[hotkey] = latest_record.account_size

            return result

    def get_historical_account_sizes(self, hotkeys: list[str] = None) -> Dict[str, List[CollateralRecord]]:
        with self._lock:
            self._load_account_sizes()

            if hotkeys is None:
                hotkeys = list(self.miner_account_sizes.keys())

            result = {}
            for hotkey in hotkeys:
                if hotkey in self.miner_account_sizes:
                    result[hotkey] = self.miner_account_sizes[hotkey]

            return result