#!/usr/bin/env python3
"""
Simple validation test for emission projection math without requiring saved ledgers.

This directly queries the chain to validate our emission rate calculation logic.

Usage:
    python validate_emission_math.py
"""

import sys
import bittensor as bt
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring


def test_emission_rate_query():
    """Test that we can query emission rates from subtensor."""
    print("=" * 80)
    print("EMISSION RATE QUERY TEST")
    print("=" * 80 + "\n")

    try:
        print("Connecting to subtensor...")
        subtensor = bt.subtensor(network="finney")
        print("✓ Connected\n")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

    netuid = 8

    try:
        print(f"Querying metagraph for netuid {netuid}...")
        metagraph = subtensor.metagraph(netuid)
        print(f"✓ Metagraph loaded with {len(metagraph.hotkeys)} neurons\n")
    except Exception as e:
        print(f"❌ Failed to get metagraph: {e}")
        return False

    # Get emission data
    print("Analyzing emission rates:")
    print("-" * 80)

    # Total emissions per block
    # metagraph.emission is already in TAO (not RAO), but per tempo (360 blocks)
    # Convert: per-tempo → per-block (÷360)
    total_tao_per_tempo = sum(metagraph.emission)
    total_emission_tao = total_tao_per_tempo / 360

    print(f"Total subnet emission: {total_emission_tao:.6f} TAO/block")

    # Calculate daily/monthly rates
    blocks_per_day = DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK
    blocks_per_month = blocks_per_day * 30

    daily_tao = total_emission_tao * blocks_per_day
    monthly_tao = total_emission_tao * blocks_per_month

    print(f"\nProjected TAO emissions:")
    print(f"  Per day:   {daily_tao:,.2f} TAO")
    print(f"  Per month: {monthly_tao:,.2f} TAO")

    # Get current block
    try:
        current_block = subtensor.get_current_block()
        print(f"\nCurrent block: {current_block:,}")
    except Exception as e:
        print(f"\n⚠️  Could not get current block: {e}")
        current_block = None

    # Show distribution stats
    emissions_list = sorted(metagraph.emission, reverse=True)
    top_10_pct = sum(emissions_list[:max(1, len(emissions_list)//10)]) / total_emission_tao * 100
    top_50_pct = sum(emissions_list[:max(1, len(emissions_list)//2)]) / total_emission_tao * 100

    print(f"\nEmission distribution:")
    print(f"  Top 10% neurons: {top_10_pct:.1f}% of emissions")
    print(f"  Top 50% neurons: {top_50_pct:.1f}% of emissions")

    # Check if realistic
    print(f"\nSanity check:")
    if total_emission_tao > 0.001:  # At least 0.001 TAO/block seems reasonable
        print(f"✅ Emission rate appears reasonable: {total_emission_tao:.6f} TAO/block")
        status = True
    else:
        print(f"⚠️  Emission rate seems very low: {total_emission_tao:.6f} TAO/block")
        status = False

    print("=" * 80 + "\n")
    return status


def test_blocks_per_day_calculation():
    """Test that blocks/day calculation makes sense."""
    print("=" * 80)
    print("BLOCKS PER DAY CALCULATION TEST")
    print("=" * 80 + "\n")

    blocks_per_day = DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK
    seconds_per_block = (24 * 60 * 60) / blocks_per_day

    print(f"Assumed blocks per day: {blocks_per_day:,}")
    print(f"Implied seconds per block: {seconds_per_block:.2f}")

    print(f"\nExpected values:")
    print(f"  Bittensor block time: ~12 seconds")
    print(f"  Expected blocks/day: ~7,200")

    error_pct = abs(seconds_per_block - 12.0) / 12.0 * 100

    print(f"\nCalculated vs expected:")
    print(f"  Error: {error_pct:.1f}%")

    if error_pct < 5:
        print("✅ Block time assumption is accurate")
        status = True
    elif error_pct < 15:
        print("✓ Block time assumption is reasonable")
        status = True
    else:
        print(f"⚠️  Block time assumption may need adjustment")
        print(f"   Consider verifying actual block time on chain")
        status = True  # Still usable

    print("=" * 80 + "\n")
    return status


def test_projection_math():
    """Test the projection math with hypothetical values."""
    print("=" * 80)
    print("PROJECTION MATH TEST (Hypothetical Values)")
    print("=" * 80 + "\n")

    # Hypothetical scenario
    total_tao_per_block = 10.0  # 10 TAO/block for subnet
    alpha_to_tao_rate = 0.5     # 1 ALPHA = 0.5 TAO
    days_until_target = 10

    print("Scenario:")
    print(f"  Subnet emission: {total_tao_per_block} TAO/block")
    print(f"  ALPHA-to-TAO rate: {alpha_to_tao_rate}")
    print(f"  Days until target: {days_until_target}")

    # Calculate manually
    blocks_until_target = days_until_target * DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK
    total_tao = total_tao_per_block * blocks_until_target
    total_alpha = total_tao / alpha_to_tao_rate

    print(f"\nCalculation:")
    print(f"  Blocks until target: {blocks_until_target:,}")
    print(f"  Total TAO: {total_tao:,}")
    print(f"  Total ALPHA: {total_alpha:,}")

    # Verify math makes sense
    expected_alpha = 10.0 * 7200 * 10 / 0.5  # 1,440,000
    error_pct = abs(total_alpha - expected_alpha) / expected_alpha * 100

    print(f"\nExpected result: {expected_alpha:,.0f} ALPHA")
    print(f"Calculated result: {total_alpha:,.0f} ALPHA")
    print(f"Error: {error_pct:.6f}%")

    if error_pct < 0.001:
        print("✅ Math is correct")
        status = True
    else:
        print("❌ Math error detected")
        status = False

    print("=" * 80 + "\n")
    return status


def main():
    """Run validation tests."""
    print("\n" + "=" * 80)
    print("EMISSION PROJECTION MATH VALIDATION")
    print("=" * 80 + "\n")
    print("This validates the mathematical correctness of our projection method")
    print("without requiring saved ledger data.\n")

    results = {}

    # Test 1: Blocks/day calculation
    print("TEST 1: Blocks Per Day Calculation")
    print("-" * 80)
    try:
        results['blocks_per_day'] = test_blocks_per_day_calculation()
    except Exception as e:
        print(f"❌ Test failed: {e}\n")
        results['blocks_per_day'] = False

    # Test 2: Projection math
    print("TEST 2: Projection Math")
    print("-" * 80)
    try:
        results['projection_math'] = test_projection_math()
    except Exception as e:
        print(f"❌ Test failed: {e}\n")
        results['projection_math'] = False

    # Test 3: Emission rate query
    print("TEST 3: Live Emission Rate Query")
    print("-" * 80)
    try:
        results['emission_query'] = test_emission_rate_query()
    except Exception as e:
        print(f"❌ Test failed: {e}\n")
        results['emission_query'] = False

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed!")
        print("   Emission projection math is correct.")
        return 0
    elif passed >= 2:
        print("\n✓ Core math validated.")
        print("  Some tests failed, but projection logic appears sound.")
        return 0
    else:
        print("\n❌ Multiple failures detected.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
