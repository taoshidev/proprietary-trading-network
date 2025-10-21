#!/usr/bin/env python3
"""
Local Debt Ledger Builder

Builds debt ledgers for miners based on their performance ledgers and positions.
Penalties include drawdown threshold, risk profile, and minimum collateral penalties.

This script loads positions from the database and builds penalty checkpoints.
In single hotkey mode, it can also generate matplotlib plots of penalties over time.
"""

import bittensor as bt
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from time_util.time_util import TimeUtil
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.position_source import PositionSourceManager, PositionSource
from shared_objects.cache_controller import CacheController
from shared_objects.mock_metagraph import MockMetagraph
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_dataclasses.penelty_ledger import DebtLedger


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Set to a specific hotkey to process single miner, or None for all miners
TEST_SINGLE_HOTKEY = '5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY'

# End time in milliseconds for position loading (None = all positions)
END_TIME_MS = None  # Example: 1736035200000

# Whether to generate matplotlib plots (only works in single hotkey mode)
SHOULD_PLOT = True

# Enable verbose/debug logging
VERBOSE = False

# Whether to use database for positions (True recommended)
USE_DATABASE_POSITIONS = True

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Enable logging
    if VERBOSE:
        bt.logging.enable_debug()
    else:
        bt.logging.enable_info()

    # Validate configuration
    if SHOULD_PLOT and not TEST_SINGLE_HOTKEY:
        bt.logging.error("SHOULD_PLOT requires TEST_SINGLE_HOTKEY to be specified")
        exit(1)

    # Initialize components
    all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    all_hotkeys_on_disk = CacheController.get_directory_names(all_miners_dir)

    # Determine which hotkeys to process
    if TEST_SINGLE_HOTKEY:
        hotkeys_to_process = [TEST_SINGLE_HOTKEY]
        bt.logging.info(f"Processing single hotkey: {TEST_SINGLE_HOTKEY}")
    else:
        hotkeys_to_process = all_hotkeys_on_disk
        bt.logging.info(f"Processing all {len(hotkeys_to_process)} hotkeys")

    # Load positions from database
    hk_to_positions = {}
    if USE_DATABASE_POSITIONS:
        source_type = PositionSource.DATABASE
        bt.logging.info("Using database as position source")

        position_source_manager = PositionSourceManager(source_type)
        hk_to_positions = position_source_manager.load_positions(
            end_time_ms=END_TIME_MS,
            hotkeys=hotkeys_to_process
        )

        if hk_to_positions:
            hotkeys_to_process = list(hk_to_positions.keys())
            bt.logging.info(f"Loaded positions for {len(hotkeys_to_process)} miners from database")

    # Initialize metagraph and managers
    mmg = MockMetagraph(hotkeys=hotkeys_to_process)
    elimination_manager = EliminationManager(mmg, None, None)
    position_manager = PositionManager(
        metagraph=mmg,
        running_unit_tests=False,
        elimination_manager=elimination_manager,
        is_backtesting=True
    )

    # Save loaded positions to position manager
    if hk_to_positions:
        position_count = 0
        for hk, positions in hk_to_positions.items():
            for pos in positions:
                position_manager.save_miner_position(pos)
                position_count += 1
        bt.logging.info(f"Saved {position_count} positions to position manager")

    # Create PerfLedgerManager
    bt.logging.info("Creating PerfLedgerManager...")
    perf_ledger_manager = PerfLedgerManager(
        mmg,
        position_manager=position_manager,
        running_unit_tests=False,
        enable_rss=False,
        build_portfolio_ledgers_only=True
    )

    # Build performance ledgers
    bt.logging.info("Building performance ledgers...")
    if TEST_SINGLE_HOTKEY:
        bt.logging.info(f"Building perf ledger for: {TEST_SINGLE_HOTKEY}")
        perf_ledger_manager.update(testing_one_hotkey=TEST_SINGLE_HOTKEY, t_ms=TimeUtil.now_in_millis())
    else:
        bt.logging.info("Building perf ledgers for all hotkeys")
        perf_ledger_manager.update()

    # Create ValidatorContractManager (with mock data for standalone mode)
    bt.logging.info("Creating ValidatorContractManager...")
    contract_manager = ValidatorContractManager(
        config=None,
        position_manager=position_manager,
        ipc_manager=None,
        metagraph=mmg,
        running_unit_tests=False
    )

    # Create DebtLedger
    bt.logging.info("Creating DebtLedger...")
    debt_ledger = DebtLedger(
        position_manager=position_manager,
        perf_ledger_manager=perf_ledger_manager,
        contract_manager=contract_manager
    )

    # Build penalty ledgers
    bt.logging.info("Building penalty ledgers...")
    debt_ledger.build_penalty_ledger(verbose=VERBOSE)

    # Print summary
    bt.logging.info("\n" + "="*60)
    bt.logging.info("Debt Ledger Summary")
    bt.logging.info("="*60)
    for hotkey, penalty_checkpoints in debt_ledger.penalty_ledgers.items():
        bt.logging.info(f"Miner {hotkey[:12]}...: {len(penalty_checkpoints)} penalty checkpoints")

    # Plot if requested and single hotkey mode
    if SHOULD_PLOT and TEST_SINGLE_HOTKEY:
        penalty_checkpoints = debt_ledger.get_penalty_ledger(TEST_SINGLE_HOTKEY)

        if not penalty_checkpoints:
            bt.logging.warning(f"No penalty checkpoints found for {TEST_SINGLE_HOTKEY}")
        else:
            bt.logging.info(f"Plotting penalties for {TEST_SINGLE_HOTKEY}")

            # Extract data for plotting
            timestamps = [datetime.fromtimestamp(cp.last_processed_ms / 1000, tz=timezone.utc)
                         for cp in penalty_checkpoints]
            drawdown_penalties = [cp.drawdown_penalty for cp in penalty_checkpoints]
            risk_profile_penalties = [cp.risk_profile_penalty for cp in penalty_checkpoints]
            min_collateral_penalties = [cp.min_collateral_penalty for cp in penalty_checkpoints]
            cumulative_penalties = [cp.cumulative_penalty for cp in penalty_checkpoints]

            # Get time range for title
            start_date = timestamps[0].strftime('%Y-%m-%d')
            end_date = timestamps[-1].strftime('%Y-%m-%d')

            # Create single plot with all penalties overlaid
            fig, ax = plt.subplots(figsize=(16, 8))

            # Calculate min/max for legend labels
            dd_min, dd_max = min(drawdown_penalties), max(drawdown_penalties)
            rp_min, rp_max = min(risk_profile_penalties), max(risk_profile_penalties)
            mc_min, mc_max = min(min_collateral_penalties), max(min_collateral_penalties)
            cum_min, cum_max = min(cumulative_penalties), max(cumulative_penalties)

            # Plot all penalties on the same axis with min/max in labels
            ax.plot(timestamps, drawdown_penalties, 'b-', linewidth=2,
                   label=f'Drawdown Threshold (min: {dd_min:.4f}, max: {dd_max:.4f})')
            ax.plot(timestamps, risk_profile_penalties, 'r-', linewidth=2,
                   label=f'Risk Profile (min: {rp_min:.4f}, max: {rp_max:.4f})')
            ax.plot(timestamps, min_collateral_penalties, 'g-', linewidth=2,
                   label=f'Min Collateral (min: {mc_min:.4f}, max: {mc_max:.4f})')
            ax.plot(timestamps, cumulative_penalties, color='purple', linewidth=2.5,
                   label=f'Cumulative (min: {cum_min:.4f}, max: {cum_max:.4f})')

            # Set title with full hotkey and time range
            ax.set_title(f'Penalty Analysis for {TEST_SINGLE_HOTKEY}\n({start_date} to {end_date})',
                        fontsize=14, pad=15)

            # Labels and grid
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Penalty Value', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add legend
            ax.legend(loc='best', fontsize=11, framealpha=0.9)

            plt.tight_layout()

            # Save plot
            plot_filename = f'debt_ledger_{TEST_SINGLE_HOTKEY}.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            bt.logging.info(f"Plot saved to: {plot_filename}")

            # Show plot
            plt.show()

    bt.logging.info("\nDebtLedger processing complete!")
