"""
Example usage of debt-based scoring

This example demonstrates how to use the debt-based scoring system to compute
miner weights based on their performance in the previous month.
"""

from datetime import datetime, timezone
from vali_objects.vali_dataclasses.debt_ledger import DebtLedger, DebtCheckpoint
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring


def main():
    """Demonstrate debt-based scoring usage"""

    # Example: December 2025 (after activation date)
    current_time = datetime(2025, 12, 15, 12, 0, 0, tzinfo=timezone.utc)
    current_time_ms = int(current_time.timestamp() * 1000)

    # Create some example debt ledgers for miners
    ledgers = {}

    # Miner 1: Strong performer
    # Previous month (Nov): Made 10,000 PnL with no penalties
    # Current month (Dec): Received 200 ALPHA so far
    prev_month_cp = datetime(2025, 11, 30, 12, 0, 0, tzinfo=timezone.utc)
    prev_month_cp_ms = int(prev_month_cp.timestamp() * 1000)

    current_month_cp = datetime(2025, 12, 1, 12, 0, 0, tzinfo=timezone.utc)
    current_month_cp_ms = int(current_month_cp.timestamp() * 1000)

    miner1 = DebtLedger(hotkey="5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw", checkpoints=[])
    miner1.checkpoints.append(DebtCheckpoint(
        timestamp_ms=prev_month_cp_ms,
        pnl_gain=15000.0,
        pnl_loss=-5000.0,  # Net PnL: 10,000
        total_penalty=1.0,  # No penalties
    ))
    miner1.checkpoints.append(DebtCheckpoint(
        timestamp_ms=current_month_cp_ms,
        chunk_emissions_alpha=200.0,  # Received 200 ALPHA
    ))
    ledgers["5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"] = miner1

    # Miner 2: Moderate performer with some penalties
    # Previous month: Made 6,000 PnL but with 80% penalty
    # Current month: Received 100 ALPHA so far
    miner2 = DebtLedger(hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", checkpoints=[])
    miner2.checkpoints.append(DebtCheckpoint(
        timestamp_ms=prev_month_cp_ms,
        pnl_gain=10000.0,
        pnl_loss=-4000.0,  # Net PnL: 6,000
        total_penalty=0.8,  # 20% penalty applied
    ))
    miner2.checkpoints.append(DebtCheckpoint(
        timestamp_ms=current_month_cp_ms,
        chunk_emissions_alpha=100.0,  # Received 100 ALPHA
    ))
    ledgers["5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"] = miner2

    # Miner 3: Weak performer with high penalties
    # Previous month: Made 2,000 PnL but with 50% penalty
    # Current month: Received 50 ALPHA so far
    miner3 = DebtLedger(hotkey="5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy", checkpoints=[])
    miner3.checkpoints.append(DebtCheckpoint(
        timestamp_ms=prev_month_cp_ms,
        pnl_gain=5000.0,
        pnl_loss=-3000.0,  # Net PnL: 2,000
        total_penalty=0.5,  # 50% penalty applied
    ))
    miner3.checkpoints.append(DebtCheckpoint(
        timestamp_ms=current_month_cp_ms,
        chunk_emissions_alpha=50.0,  # Received 50 ALPHA
    ))
    ledgers["5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"] = miner3

    # Compute weights using debt-based scoring
    print("\n" + "="*80)
    print("DEBT-BASED SCORING EXAMPLE")
    print("="*80)
    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Number of miners: {len(ledgers)}")
    print("\n")

    # Show debt analysis for each miner
    print("MINER DEBT ANALYSIS:")
    print("-" * 80)
    for hotkey, ledger in ledgers.items():
        prev_cp = [cp for cp in ledger.checkpoints if cp.timestamp_ms == prev_month_cp_ms][0]
        curr_cp = [cp for cp in ledger.checkpoints if cp.timestamp_ms == current_month_cp_ms][0]

        needed_payout = prev_cp.net_pnl * prev_cp.total_penalty
        actual_payout = curr_cp.chunk_emissions_alpha
        remaining_payout = max(0, needed_payout - actual_payout)

        print(f"\nMiner: ...{hotkey[-8:]}")
        print(f"  Previous Month (Nov 2025):")
        print(f"    Net PnL: ${prev_cp.net_pnl:,.2f}")
        print(f"    Total Penalty: {prev_cp.total_penalty:.2%}")
        print(f"    Needed Payout: {needed_payout:,.2f} ALPHA")
        print(f"  Current Month (Dec 2025):")
        print(f"    Actual Payout: {actual_payout:,.2f} ALPHA")
        print(f"    Remaining Payout: {remaining_payout:,.2f} ALPHA")

    # Compute weights
    weights = DebtBasedScoring.compute_results(
        ledger_dict=ledgers,
        current_time_ms=current_time_ms,
        verbose=True
    )

    print("\n" + "="*80)
    print("COMPUTED WEIGHTS:")
    print("-" * 80)
    for hotkey, weight in weights:
        print(f"Miner ...{hotkey[-8:]}: {weight:.6f} ({weight*100:.4f}%)")

    print("\n" + "="*80)
    print("SUMMARY:")
    print("-" * 80)
    total_weight = sum(w for _, w in weights)
    print(f"Total weight (should be 1.0): {total_weight:.10f}")
    print(f"Top miner: ...{weights[0][0][-8:]} with {weights[0][1]*100:.4f}% weight")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
