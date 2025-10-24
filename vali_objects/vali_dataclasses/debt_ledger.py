"""
Debt Ledger - Unified view combining emissions, penalties, and performance data

This module provides a unified DebtLedger structure that combines:
- Emissions data (alpha/TAO/USD) from EmissionsLedger
- Penalty multipliers from PenaltyLedger
- Performance metrics (PnL, fees, drawdown) from PerfLedger

The DebtLedger provides a complete financial picture for each miner, making it
easy for the UI to display comprehensive miner statistics.

Architecture:
- DebtCheckpoint: Data for a single point in time
- DebtLedger: Complete debt history for a SINGLE hotkey
- DebtLedgerManager: (TODO) Manages ledgers for multiple hotkeys

Usage:
    # Create a debt ledger for a miner
    ledger = DebtLedger(hotkey="5...")

    # Add a checkpoint combining all data sources
    checkpoint = DebtCheckpoint(
        timestamp_ms=1234567890000,
        # Emissions data
        chunk_emissions_alpha=10.5,
        chunk_emissions_tao=0.05,
        chunk_emissions_usd=25.0,
        # Performance data
        portfolio_return=1.15,
        pnl_gain=1000.0,
        pnl_loss=-200.0,
        # ... other fields
    )
    ledger.add_checkpoint(checkpoint)

Standalone Usage:
Use runnable/local_debt_ledger.py for standalone execution with hard-coded configuration.
Edit the configuration variables at the top of that file to customize behavior.

"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timezone

from vali_objects.vali_dataclasses.emissions_ledger import EmissionsLedgerManager
from vali_objects.vali_dataclasses.penalty_ledger import PenaltyLedgerManager


@dataclass
class DebtCheckpoint:
    """
    Unified checkpoint combining emissions, penalties, and performance data.

    All data is aligned to a single timestamp representing a snapshot in time
    of the miner's complete financial state.

    Attributes:
        # Timing
        timestamp_ms: Checkpoint timestamp in milliseconds

        # Emissions Data (from EmissionsLedger) - chunk data only, no cumulative
        chunk_emissions_alpha: Alpha tokens earned in this chunk
        chunk_emissions_tao: TAO value earned in this chunk
        chunk_emissions_usd: USD value earned in this chunk
        avg_alpha_to_tao_rate: Average alpha-to-TAO conversion rate for this chunk
        avg_tao_to_usd_rate: Average TAO/USD price for this chunk

        # Performance Data (from PerfLedger)
        portfolio_return: Current portfolio return multiplier (1.0 = break-even)
        pnl_gain: Cumulative PnL gain
        pnl_loss: Cumulative PnL loss
        spread_fee_loss: Cumulative spread fee losses
        carry_fee_loss: Cumulative carry fee losses
        max_drawdown: Maximum drawdown (worst loss from peak)
        max_portfolio_value: Maximum portfolio value achieved
        open_ms: Total time with open positions (milliseconds)
        accum_ms: Total accumulated time (milliseconds)
        n_updates: Number of performance updates

        # Penalty Data (from PenaltyLedger)
        drawdown_penalty: Drawdown threshold penalty multiplier
        risk_profile_penalty: Risk profile penalty multiplier
        min_collateral_penalty: Minimum collateral penalty multiplier
        risk_adjusted_performance_penalty: Risk-adjusted performance penalty multiplier
        total_penalty: Combined penalty multiplier (product of all penalties)

        # Derived/Computed Fields
        net_pnl: Net PnL (gain + loss)
        total_fees: Total fees paid (spread + carry)
        return_after_fees: Portfolio return after all fees
        weighted_score: Final score after applying all penalties
    """
    # Timing
    timestamp_ms: int

    # Emissions Data (chunk only, cumulative calculated by summing)
    chunk_emissions_alpha: float = 0.0
    chunk_emissions_tao: float = 0.0
    chunk_emissions_usd: float = 0.0
    avg_alpha_to_tao_rate: Optional[float] = None
    avg_tao_to_usd_rate: Optional[float] = None

    # Performance Data
    portfolio_return: float = 1.0
    pnl_gain: float = 0.0
    pnl_loss: float = 0.0
    spread_fee_loss: float = 0.0
    carry_fee_loss: float = 0.0
    max_drawdown: float = 1.0
    max_portfolio_value: float = 0.0
    open_ms: int = 0
    accum_ms: int = 0
    n_updates: int = 0

    # Penalty Data
    drawdown_penalty: float = 1.0
    risk_profile_penalty: float = 1.0
    min_collateral_penalty: float = 1.0
    risk_adjusted_performance_penalty: float = 1.0
    total_penalty: float = 1.0

    def __post_init__(self):
        """Calculate derived fields after initialization"""
        self.net_pnl = self.pnl_gain + self.pnl_loss
        self.total_fees = self.spread_fee_loss + self.carry_fee_loss
        self.return_after_fees = self.portfolio_return
        self.weighted_score = self.portfolio_return * self.total_penalty

    def __eq__(self, other):
        if not isinstance(other, DebtCheckpoint):
            return False
        return self.timestamp_ms == other.timestamp_ms

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            # Timing
            'timestamp_ms': self.timestamp_ms,
            'timestamp_utc': datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc).isoformat(),

            # Emissions (chunk only)
            'emissions': {
                'chunk_alpha': self.chunk_emissions_alpha,
                'chunk_tao': self.chunk_emissions_tao,
                'chunk_usd': self.chunk_emissions_usd,
                'avg_alpha_to_tao_rate': self.avg_alpha_to_tao_rate,
                'avg_tao_to_usd_rate': self.avg_tao_to_usd_rate,
            },

            # Performance
            'performance': {
                'portfolio_return': self.portfolio_return,
                'pnl_gain': self.pnl_gain,
                'pnl_loss': self.pnl_loss,
                'net_pnl': self.net_pnl,
                'spread_fee_loss': self.spread_fee_loss,
                'carry_fee_loss': self.carry_fee_loss,
                'total_fees': self.total_fees,
                'max_drawdown': self.max_drawdown,
                'max_portfolio_value': self.max_portfolio_value,
                'open_ms': self.open_ms,
                'accum_ms': self.accum_ms,
                'n_updates': self.n_updates,
            },

            # Penalties
            'penalties': {
                'drawdown': self.drawdown_penalty,
                'risk_profile': self.risk_profile_penalty,
                'min_collateral': self.min_collateral_penalty,
                'risk_adjusted_performance': self.risk_adjusted_performance_penalty,
                'cumulative': self.total_penalty,
            },

            # Derived
            'derived': {
                'return_after_fees': self.return_after_fees,
                'weighted_score': self.weighted_score,
            }
        }


class DebtLedger:
    """
    Complete debt/earnings ledger for a SINGLE hotkey.

    Combines emissions, penalties, and performance data into a unified view.
    Stores checkpoints in chronological order.
    """

    def __init__(self, hotkey: str, checkpoints: Optional[List[DebtCheckpoint]] = None):
        """
        Initialize debt ledger for a single hotkey.

        Args:
            hotkey: SS58 address of the hotkey
            checkpoints: Optional list of debt checkpoints
        """
        self.hotkey = hotkey
        self.checkpoints: List[DebtCheckpoint] = checkpoints or []

    def add_checkpoint(self, checkpoint: DebtCheckpoint):
        """
        Add a checkpoint to the ledger.

        Validates that checkpoints are added in chronological order.

        Args:
            checkpoint: The checkpoint to add

        Raises:
            AssertionError: If checkpoint timestamp is not after the previous checkpoint
        """
        if self.checkpoints:
            prev_checkpoint = self.checkpoints[-1]
            assert checkpoint.timestamp_ms > prev_checkpoint.timestamp_ms, (
                f"Checkpoint timestamp must be after previous checkpoint. "
                f"Previous: {prev_checkpoint.timestamp_ms}, New: {checkpoint.timestamp_ms}"
            )

        self.checkpoints.append(checkpoint)

    def get_latest_checkpoint(self) -> Optional[DebtCheckpoint]:
        """Get the most recent checkpoint"""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_checkpoint_at_time(self, timestamp_ms: int) -> Optional[DebtCheckpoint]:
        """
        Get the checkpoint at or before a specific time.

        Args:
            timestamp_ms: Timestamp to query

        Returns:
            Latest checkpoint at or before the timestamp, or None
        """
        valid_checkpoints = [
            cp for cp in self.checkpoints
            if cp.timestamp_ms <= timestamp_ms
        ]
        return valid_checkpoints[-1] if valid_checkpoints else None

    def get_cumulative_emissions_alpha(self) -> float:
        """Get total cumulative alpha emissions by summing chunk emissions"""
        return sum(cp.chunk_emissions_alpha for cp in self.checkpoints)

    def get_cumulative_emissions_tao(self) -> float:
        """Get total cumulative TAO emissions by summing chunk emissions"""
        return sum(cp.chunk_emissions_tao for cp in self.checkpoints)

    def get_cumulative_emissions_usd(self) -> float:
        """Get total cumulative USD emissions by summing chunk emissions"""
        return sum(cp.chunk_emissions_usd for cp in self.checkpoints)

    def get_current_portfolio_return(self) -> float:
        """Get current portfolio return"""
        latest = self.get_latest_checkpoint()
        return latest.portfolio_return if latest else 1.0

    def get_current_weighted_score(self) -> float:
        """Get current weighted score (return * penalties)"""
        latest = self.get_latest_checkpoint()
        return latest.weighted_score if latest else 1.0

    def to_dict(self) -> dict:
        """
        Convert ledger to dictionary for serialization.

        Returns:
            Dictionary with hotkey and all checkpoints
        """
        latest = self.get_latest_checkpoint()

        return {
            'hotkey': self.hotkey,
            'total_checkpoints': len(self.checkpoints),

            # Summary statistics (cumulative emissions calculated by summing)
            'summary': {
                'cumulative_emissions_alpha': self.get_cumulative_emissions_alpha(),
                'cumulative_emissions_tao': self.get_cumulative_emissions_tao(),
                'cumulative_emissions_usd': self.get_cumulative_emissions_usd(),
                'portfolio_return': self.get_current_portfolio_return(),
                'weighted_score': self.get_current_weighted_score(),
                'net_pnl': latest.net_pnl if latest else 0.0,
                'total_fees': latest.total_fees if latest else 0.0,
            } if latest else {},

            # All checkpoints
            'checkpoints': [cp.to_dict() for cp in self.checkpoints]
        }

    def print_summary(self):
        """Print a formatted summary of the debt ledger"""
        if not self.checkpoints:
            print(f"\nNo debt ledger data found for {self.hotkey}")
            return

        latest = self.get_latest_checkpoint()

        print(f"\n{'='*80}")
        print(f"Debt Ledger Summary for {self.hotkey}")
        print(f"{'='*80}")
        print(f"Total Checkpoints: {len(self.checkpoints)}")
        print(f"\n--- Emissions ---")
        print(f"Total Alpha: {self.get_cumulative_emissions_alpha():.6f}")
        print(f"Total TAO: {self.get_cumulative_emissions_tao():.6f}")
        print(f"Total USD: ${self.get_cumulative_emissions_usd():,.2f}")
        print(f"\n--- Performance ---")
        print(f"Portfolio Return: {latest.portfolio_return:.4f} ({(latest.portfolio_return - 1) * 100:+.2f}%)")
        print(f"Net PnL: ${latest.net_pnl:,.2f}")
        print(f"Total Fees: ${latest.total_fees:,.2f}")
        print(f"Max Drawdown: {latest.max_drawdown:.4f}")
        print(f"\n--- Penalties ---")
        print(f"Drawdown Penalty: {latest.drawdown_penalty:.4f}")
        print(f"Risk Profile Penalty: {latest.risk_profile_penalty:.4f}")
        print(f"Min Collateral Penalty: {latest.min_collateral_penalty:.4f}")
        print(f"Risk Adjusted Performance Penalty: {latest.risk_adjusted_performance_penalty:.4f}")
        print(f"Cumulative Penalty: {latest.total_penalty:.4f}")
        print(f"\n--- Final Score ---")
        print(f"Weighted Score: {latest.weighted_score:.4f}")
        print(f"{'='*80}\n")


# TODO: Implement DebtLedgerManager class
class DebtLedgerManager:
    #     """
    #     Manages debt ledgers for multiple hotkeys.
    #
    #     Responsibilities:
    #     - Combine data from EmissionsLedgerManager, PerfLedgerManager, and PenaltyLedger
    #     - Build unified DebtCheckpoints by merging data from all three sources
    #     - Handle serialization/deserialization
    #     - Provide query methods for UI consumption
    #     """
    #     pass
    def __init__(self, perf_ledger_manager, position_manager, contract_manager, slack_webhook_url=None, start_daemon=True, ipc_manager=None):
        self.penalty_ledger_manager = PenaltyLedgerManager(position_manager=position_manager, perf_ledger_manager=perf_ledger_manager,
           contract_manager=contract_manager, slack_webhook_url=slack_webhook_url)
        self.perf_ledger_manager = perf_ledger_manager
        self.emissions_ledger_manager = EmissionsLedgerManager(slack_webhook_url=slack_webhook_url, start_daemon=start_daemon,
                                                               ipc_manager=ipc_manager)
        #self.penalty_ledger_manager =PenaltyLedgerManager
