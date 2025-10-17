"""
Debt Ledger - Tracks penalty checkpoints aligned with performance ledger checkpoints

This module builds penalty ledgers for miners based on their performance ledgers and positions.
Penalties include drawdown threshold, risk profile, and minimum collateral penalties.

Standalone Usage:
    Use runnable/local_debt_ledger.py for standalone execution with hard-coded configuration.
    Edit the configuration variables at the top of that file to customize behavior.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum, auto
from vali_objects.position import Position
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint, TP_ID_PORTFOLIO
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.position_filter import PositionFilter
import bittensor as bt


class PenaltyInputType(Enum):
    LEDGER = auto()
    POSITIONS = auto()
    PSEUDO_POSITIONS = auto()
    COLLATERAL = auto()


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
        cumulative_penalty: float = 1.0
    ):
        self.last_processed_ms = int(last_processed_ms)
        self.drawdown_penalty = float(drawdown_penalty)
        self.risk_profile_penalty = float(risk_profile_penalty)
        self.min_collateral_penalty = float(min_collateral_penalty)
        self.cumulative_penalty = float(cumulative_penalty)

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


class DebtLedger:
    """
    Manages penalty ledgers aligned with performance checkpoints.
    Reads positions and perf ledgers to build penalty checkpoints.
    """

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
        )
    }

    def __init__(
        self,
        position_manager: PositionManager,
        perf_ledger_manager: PerfLedgerManager,
        contract_manager: ValidatorContractManager
    ):
        """
        Initialize DebtLedger with managers for positions, performance ledgers, and collateral.

        Args:
            position_manager: Manager for reading miner positions
            perf_ledger_manager: Manager for reading performance ledgers
            contract_manager: Manager for reading miner collateral/account sizes
        """
        self.position_manager = position_manager
        self.perf_ledger_manager = perf_ledger_manager
        self.contract_manager = contract_manager

        # Read all positions from position manager
        self.all_positions: Dict[str, List[Position]] = self.position_manager.get_positions_for_all_miners()
        bt.logging.info(f"DebtLedger loaded positions for {len(self.all_positions)} miners")

        # Read all perf ledgers from perf ledger manager
        self.all_perf_ledgers: Dict[str, Dict[str, PerfLedger]] = self.perf_ledger_manager.get_perf_ledgers(
            portfolio_only=False
        )
        bt.logging.info(f"DebtLedger loaded perf ledgers for {len(self.all_perf_ledgers)} miners")

        # Storage for penalty checkpoints per miner
        self.penalty_ledgers: Dict[str, List[PenaltyCheckpoint]] = {}

    def get_positions_at_date(self, cutoff_date_ms: int) -> Dict[str, List[Position]]:
        """
        Get all positions that are open at a given date (in ms).

        This utility function is critical for build_penalty_ledger to work.
        Uses PositionFilter.filter_single_position_simple for filtering.

        Args:
            cutoff_date_ms: Timestamp in milliseconds to filter positions

        Returns:
            Dict mapping miner hotkeys to lists of their positions filtered by date
        """
        filtered_positions = {}

        for miner_hotkey, positions in self.all_positions.items():
            miner_filtered_positions = []

            for position in positions:
                filtered_position = PositionFilter.filter_single_position_simple(position, cutoff_date_ms)
                if filtered_position:
                    miner_filtered_positions.append(filtered_position)

            if miner_filtered_positions:
                filtered_positions[miner_hotkey] = miner_filtered_positions

        return filtered_positions

    def build_penalty_ledger(self, verbose: bool = False):
        """
        Build penalty ledgers for all checkpoints in all performance ledgers.

        This function iterates through all checkpoints in each miner's portfolio perf ledger
        and computes the penalties at each checkpoint time using the positions at that time.

        Args:
            verbose: Enable detailed logging
        """
        self.penalty_ledgers.clear()

        for miner_hotkey, ledger_dict in self.all_perf_ledgers.items():
            # Get portfolio ledger for this miner
            portfolio_ledger = ledger_dict.get(TP_ID_PORTFOLIO)

            if not portfolio_ledger or not portfolio_ledger.cps:
                if verbose:
                    bt.logging.debug(f"Skipping miner {miner_hotkey}: no portfolio ledger or checkpoints")
                continue

            miner_penalty_checkpoints = []

            # Get miner's collateral/account size
            miner_account_size = 0
            if self.contract_manager and hasattr(self.contract_manager, 'miner_account_sizes'):
                miner_account_size = self.contract_manager.miner_account_sizes.get(miner_hotkey, 0)
                if miner_account_size is None:
                    miner_account_size = 0

            # Iterate through all checkpoints in the portfolio ledger
            for checkpoint in portfolio_ledger.cps:
                checkpoint_ms = checkpoint.last_update_ms

                # Get positions at this checkpoint time
                positions_at_checkpoint = self.get_positions_at_date(checkpoint_ms)
                miner_positions_at_checkpoint = positions_at_checkpoint.get(miner_hotkey, [])

                # Calculate each penalty
                penalties = {}
                cumulative_penalty = 1.0

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

                    except Exception as e:
                        if verbose:
                            bt.logging.warning(
                                f"Error computing {penalty_name} for miner {miner_hotkey} at {checkpoint_ms}: {e}"
                            )
                        penalty_value = 1.0

                    penalties[penalty_name] = penalty_value
                    cumulative_penalty *= penalty_value

                # Create penalty checkpoint
                penalty_checkpoint = PenaltyCheckpoint(
                    last_processed_ms=checkpoint_ms,
                    drawdown_penalty=penalties.get('drawdown_threshold', 1.0),
                    risk_profile_penalty=penalties.get('risk_profile', 1.0),
                    min_collateral_penalty=penalties.get('min_collateral', 1.0),
                    cumulative_penalty=cumulative_penalty
                )

                miner_penalty_checkpoints.append(penalty_checkpoint)

            if miner_penalty_checkpoints:
                self.penalty_ledgers[miner_hotkey] = miner_penalty_checkpoints
                if verbose:
                    bt.logging.info(
                        f"Built {len(miner_penalty_checkpoints)} penalty checkpoints for miner {miner_hotkey}"
                    )

        bt.logging.info(f"Built penalty ledgers for {len(self.penalty_ledgers)} miners")

    def get_penalty_ledger(self, miner_hotkey: str) -> List[PenaltyCheckpoint]:
        """
        Get the penalty ledger for a specific miner.

        Args:
            miner_hotkey: The miner's hotkey

        Returns:
            List of PenaltyCheckpoints for the miner, or empty list if not found
        """
        return self.penalty_ledgers.get(miner_hotkey, [])

    def get_penalty_at_time(self, miner_hotkey: str, timestamp_ms: int) -> Optional[PenaltyCheckpoint]:
        """
        Get the penalty checkpoint at or before a specific time.

        Args:
            miner_hotkey: The miner's hotkey
            timestamp_ms: The timestamp to query

        Returns:
            PenaltyCheckpoint at or before the timestamp, or None if not found
        """
        penalty_checkpoints = self.penalty_ledgers.get(miner_hotkey, [])

        if not penalty_checkpoints:
            return None

        # Find the latest checkpoint at or before the timestamp
        valid_checkpoints = [
            cp for cp in penalty_checkpoints
            if cp.last_processed_ms <= timestamp_ms
        ]

        if not valid_checkpoints:
            return None

        return max(valid_checkpoints, key=lambda cp: cp.last_processed_ms)

