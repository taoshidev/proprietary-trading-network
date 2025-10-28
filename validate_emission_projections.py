#!/usr/bin/env python3
"""
Standalone script to validate emission projection accuracy against historical data.

This compares our projection method against actual emissions received over past periods.

Usage:
    python validate_emission_projections.py
"""

import sys
from datetime import datetime, timezone
from unittest.mock import Mock
import bittensor as bt

from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_dataclasses.emissions_ledger import EmissionsLedgerManager
from time_util.time_util import TimeUtil


def validate_projection_accuracy(days_back: int = 7):
    """
    Validate projection accuracy by comparing to actual emissions.

    Args:
        days_back: Number of days back to test (default: 7)
    """
    print("=" * 80)
    print(f"EMISSION PROJECTION VALIDATION TEST")
    print("=" * 80)
    print(f"Validating {days_back}-day projection against actual emissions data\n")

    # Initialize emissions ledger manager
    netuid = 8
    archive_endpoint = "wss://archive.chain.opentensor.ai:443"

    print("Loading emissions ledger data...")
    # Create mock perf_ledger_manager (not needed for validation, just for initialization)
    mock_perf_ledger = Mock()

    emissions_mgr = EmissionsLedgerManager(
        perf_ledger_manager=mock_perf_ledger,
        archive_endpoint=archive_endpoint,
        netuid=netuid
    )

    try:
        emissions_mgr.load_from_disk()
    except Exception as e:
        print(f"❌ Could not load emissions data: {e}")
        return False

    if not emissions_mgr.emissions_ledgers:
        print("❌ No emissions ledger data available")
        return False

    print(f"✓ Loaded emissions data for {len(emissions_mgr.emissions_ledgers)} hotkeys\n")

    # Calculate time boundaries
    now_ms = TimeUtil.now_in_millis()
    past_time_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)

    now_dt = TimeUtil.millis_to_datetime(now_ms)
    past_dt = TimeUtil.millis_to_datetime(past_time_ms)

    print(f"📅 Period: {past_dt.strftime('%Y-%m-%d %H:%M')} to {now_dt.strftime('%Y-%m-%d %H:%M')}")

    # Calculate actual emissions received
    print(f"\nCalculating actual ALPHA received over past {days_back} days...")
    actual_alpha = 0.0
    checkpoint_count = 0

    for hotkey, ledger in emissions_mgr.emissions_ledgers.items():
        for checkpoint in ledger.checkpoints:
            if past_time_ms <= checkpoint.timestamp_ms <= now_ms:
                actual_alpha += checkpoint.chunk_emissions
                checkpoint_count += 1

    print(f"✓ Found {checkpoint_count} checkpoints in time period")
    print(f"✓ Actual ALPHA received: {actual_alpha:,.2f}\n")

    if actual_alpha == 0:
        print("❌ No emissions data in specified period")
        return False

    # Make projection using current rates
    print(f"Querying current emission rates from subtensor...")
    try:
        subtensor = bt.subtensor(network="finney")
        print("✓ Connected to subtensor\n")
    except Exception as e:
        print(f"❌ Could not connect to subtensor: {e}")
        return False

    try:
        projected_alpha = DebtBasedScoring._estimate_alpha_emissions_until_target(
            subtensor=subtensor,
            netuid=netuid,
            emissions_ledger_manager=emissions_mgr,
            days_until_target=days_back,
            verbose=True
        )
        print(f"\n✓ Projection completed\n")
    except Exception as e:
        print(f"❌ Projection failed: {e}")
        return False

    # Compare results
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"Projected ALPHA (using current rates): {projected_alpha:,.2f}")
    print(f"Actual ALPHA received:                  {actual_alpha:,.2f}")
    print(f"Difference:                             {abs(projected_alpha - actual_alpha):,.2f}")

    if projected_alpha > 0:
        error_pct = abs(projected_alpha - actual_alpha) / actual_alpha * 100
        ratio = projected_alpha / actual_alpha

        print(f"Error percentage:                       {error_pct:.1f}%")
        print(f"Projection/Actual ratio:                {ratio:.2f}x")

        print("\nINTERPRETATION:")
        if error_pct < 10:
            print("✅ EXCELLENT: Projection very accurate (< 10% error)")
            status = True
        elif error_pct < 25:
            print("✓ GOOD: Projection reasonably accurate (< 25% error)")
            status = True
        elif error_pct < 50:
            print("⚠️  MODERATE: Projection has notable error (< 50%)")
            print("   This may be due to emission rate or conversion rate changes")
            status = True
        else:
            print("❌ HIGH ERROR: Projection significantly off (> 50%)")
            print("   Consider investigating emission rate volatility")
            status = False

        # Provide context
        print("\nCONTEXT:")
        print("• This test uses CURRENT emission rates to project backwards")
        print("• High error may indicate emission rate or conversion rate changes")
        print("• For forward projections, rates are expected to be more stable")

    else:
        print("❌ Projection returned zero")
        status = False

    print("=" * 80 + "\n")
    return status


def check_blocks_per_day_assumption():
    """Validate the 7200 blocks/day assumption."""
    print("=" * 80)
    print("BLOCKS PER DAY ASSUMPTION VALIDATION")
    print("=" * 80 + "\n")

    print("Loading emissions ledger data...")
    mock_perf_ledger = Mock()
    emissions_mgr = EmissionsLedgerManager(
        perf_ledger_manager=mock_perf_ledger,
        archive_endpoint="wss://archive.chain.opentensor.ai:443",
        netuid=8
    )

    try:
        emissions_mgr.load_from_disk()
    except Exception as e:
        print(f"❌ Could not load emissions data: {e}")
        return False

    # Sample checkpoints
    print("Sampling recent checkpoints...\n")
    checkpoints = []

    for hotkey, ledger in emissions_mgr.emissions_ledgers.items():
        for cp in ledger.checkpoints[-20:]:  # Last 20 checkpoints
            if hasattr(cp, 'block_end') and hasattr(cp, 'timestamp_ms'):
                checkpoints.append({
                    'timestamp_ms': cp.timestamp_ms,
                    'block_end': cp.block_end
                })
        if len(checkpoints) >= 10:
            break

    if len(checkpoints) < 2:
        print("❌ Not enough checkpoint data")
        return False

    checkpoints.sort(key=lambda x: x['timestamp_ms'])

    # Calculate blocks/day from intervals
    blocks_per_day_samples = []

    for i in range(len(checkpoints) - 1):
        cp1 = checkpoints[i]
        cp2 = checkpoints[i + 1]

        time_diff_ms = cp2['timestamp_ms'] - cp1['timestamp_ms']
        block_diff = cp2['block_end'] - cp1['block_end']

        if time_diff_ms > 0 and block_diff > 0:
            blocks_per_day = (block_diff / time_diff_ms) * (24 * 60 * 60 * 1000)
            blocks_per_day_samples.append(blocks_per_day)

    if not blocks_per_day_samples:
        print("❌ Could not calculate blocks/day")
        return False

    avg_blocks_per_day = sum(blocks_per_day_samples) / len(blocks_per_day_samples)
    min_blocks = min(blocks_per_day_samples)
    max_blocks = max(blocks_per_day_samples)

    print(f"Sample intervals analyzed: {len(blocks_per_day_samples)}")
    print(f"Assumed blocks/day: {DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK:,.0f}")
    print(f"Actual blocks/day:  {avg_blocks_per_day:,.0f}")
    print(f"Range: {min_blocks:,.0f} - {max_blocks:,.0f}")

    error_pct = abs(avg_blocks_per_day - DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK) / DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK * 100
    print(f"Error: {error_pct:.1f}%")

    if error_pct < 5:
        print("\n✅ EXCELLENT: Blocks/day assumption very accurate")
        status = True
    elif error_pct < 15:
        print("\n✓ GOOD: Blocks/day assumption reasonably accurate")
        status = True
    else:
        print(f"\n⚠️  WARNING: Blocks/day assumption off by {error_pct:.1f}%")
        print(f"   Consider updating BLOCKS_PER_DAY_FALLBACK to {avg_blocks_per_day:.0f}")
        status = True  # Still usable, just suboptimal

    print("=" * 80 + "\n")
    return status


def check_conversion_rate_volatility():
    """Check alpha-to-TAO conversion rate stability."""
    print("=" * 80)
    print("CONVERSION RATE VOLATILITY CHECK")
    print("=" * 80 + "\n")

    print("Loading emissions ledger data...")
    mock_perf_ledger = Mock()
    emissions_mgr = EmissionsLedgerManager(
        perf_ledger_manager=mock_perf_ledger,
        archive_endpoint="wss://archive.chain.opentensor.ai:443",
        netuid=8
    )

    try:
        emissions_mgr.load_from_disk()
    except Exception as e:
        print(f"❌ Could not load emissions data: {e}")
        return False

    # Sample conversion rates
    print("Sampling recent conversion rates...\n")
    rates = []

    for hotkey, ledger in emissions_mgr.emissions_ledgers.items():
        for cp in ledger.checkpoints[-30:]:  # Last 30 checkpoints
            if hasattr(cp, 'avg_alpha_to_tao_rate') and cp.avg_alpha_to_tao_rate > 0:
                rates.append(cp.avg_alpha_to_tao_rate)
        if len(rates) >= 20:
            break

    if len(rates) < 2:
        print("❌ Not enough conversion rate data")
        return False

    avg_rate = sum(rates) / len(rates)
    min_rate = min(rates)
    max_rate = max(rates)

    # Calculate coefficient of variation
    variance = sum((r - avg_rate) ** 2 for r in rates) / len(rates)
    std_dev = variance ** 0.5
    cv = (std_dev / avg_rate) * 100 if avg_rate > 0 else 0

    print(f"Sample size: {len(rates)} checkpoints")
    print(f"Average rate: {avg_rate:.6f} TAO per ALPHA")
    print(f"Min: {min_rate:.6f}, Max: {max_rate:.6f}")
    print(f"Std deviation: {std_dev:.6f}")
    print(f"Coefficient of variation: {cv:.1f}%")

    print("\nVOLATILITY ASSESSMENT:")
    if cv < 5:
        print("✅ STABLE: Very low volatility, projections should be accurate")
        status = True
    elif cv < 15:
        print("✓ MODERATE: Some volatility, projections reasonably reliable")
        status = True
    elif cv < 30:
        print("⚠️  HIGH: Significant volatility, projections may vary")
        status = True
    else:
        print("❌ VERY HIGH: Extreme volatility, projections less reliable")
        print("   Consider shorter projection windows or rate averaging")
        status = False

    print("\nIMPLICATION:")
    print(f"• Current method uses instantaneous conversion rate")
    print(f"• With {cv:.1f}% volatility, projections have inherent uncertainty")
    print(f"• Higher volatility = wider confidence intervals on projections")

    print("=" * 80 + "\n")
    return status


def main():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("EMISSION PROJECTION VALIDATION SUITE")
    print("=" * 80 + "\n")

    results = {}

    # Test 1: 7-day projection accuracy
    print("TEST 1: 7-Day Projection Accuracy")
    print("-" * 80)
    try:
        results['projection_7day'] = validate_projection_accuracy(days_back=7)
    except Exception as e:
        print(f"❌ Test failed with error: {e}\n")
        results['projection_7day'] = False

    # Test 2: Blocks/day assumption
    print("\nTEST 2: Blocks Per Day Validation")
    print("-" * 80)
    try:
        results['blocks_per_day'] = check_blocks_per_day_assumption()
    except Exception as e:
        print(f"❌ Test failed with error: {e}\n")
        results['blocks_per_day'] = False

    # Test 3: Conversion rate volatility
    print("\nTEST 3: Conversion Rate Volatility")
    print("-" * 80)
    try:
        results['conversion_volatility'] = check_conversion_rate_volatility()
    except Exception as e:
        print(f"❌ Test failed with error: {e}\n")
        results['conversion_volatility'] = False

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:30s} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All validation tests passed!")
        print("   Emission projection method appears accurate and well-calibrated.")
        return 0
    elif passed >= total / 2:
        print("\n⚠️  Some tests failed, but projection method is usable.")
        print("   Review failed tests for potential improvements.")
        return 0
    else:
        print("\n❌ Multiple validation failures detected.")
        print("   Consider investigating emission projection methodology.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
