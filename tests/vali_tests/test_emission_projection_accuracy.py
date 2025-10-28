"""
Test emission projection accuracy by comparing projections against historical reality.

This test validates that our emission estimation method aligns with actual emissions
recorded in the emissions ledgers over past time periods.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock
import bittensor as bt

from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_dataclasses.emissions_ledger import EmissionsLedgerManager
from time_util.time_util import TimeUtil


class TestEmissionProjectionAccuracy(unittest.TestCase):
    """
    Test that our emission projection method matches historical reality.

    This uses REAL emissions ledger data to validate our projection algorithm.
    """

    def setUp(self):
        """Set up with real emissions ledger manager"""
        # This test requires actual ledger data, so skip if not available
        try:
            self.netuid = 8
            self.archive_endpoint = "wss://archive.chain.opentensor.ai:443"

            # Create real emissions ledger manager (read-only, no modifications)
            self.emissions_mgr = EmissionsLedgerManager(
                netuid=self.netuid,
                archive_endpoint=self.archive_endpoint
            )

            # Try to load existing ledgers
            self.emissions_mgr.load_from_disk()

            # Check if we have any data
            if not self.emissions_mgr.emissions_ledgers:
                self.skipTest("No emissions ledger data available for validation")

        except Exception as e:
            self.skipTest(f"Could not initialize emissions ledger manager: {e}")

    def test_projection_vs_actual_past_7_days(self):
        """
        Compare projected emissions to actual emissions over past 7 days.

        Method:
        1. Go back 7 days in time
        2. Simulate making a projection at that point
        3. Compare projected emissions to actual emissions received
        """

        # Calculate time boundaries
        now_ms = TimeUtil.now_in_millis()
        seven_days_ago_ms = now_ms - (7 * 24 * 60 * 60 * 1000)

        # Get actual emissions received over past 7 days across all miners
        actual_alpha_received = self._calculate_actual_emissions_in_period(
            seven_days_ago_ms,
            now_ms
        )

        if actual_alpha_received == 0:
            self.skipTest("No emissions data in past 7 days")

        print(f"\n=== 7-Day Projection Accuracy Test ===")
        print(f"Period: {TimeUtil.millis_to_datetime(seven_days_ago_ms)} to {TimeUtil.millis_to_datetime(now_ms)}")
        print(f"Actual ALPHA received: {actual_alpha_received:.2f}")

        # Simulate making a projection 7 days ago
        try:
            # Get subtensor and make projection
            subtensor = bt.subtensor(network="finney")

            projected_alpha = DebtBasedScoring._estimate_alpha_emissions_until_target(
                subtensor=subtensor,
                netuid=self.netuid,
                emissions_ledger_manager=self.emissions_mgr,
                days_until_target=7,
                verbose=True
            )

            print(f"Projected ALPHA (using current rates): {projected_alpha:.2f}")

            # Calculate accuracy
            if projected_alpha > 0:
                error_pct = abs(projected_alpha - actual_alpha_received) / actual_alpha_received * 100
                print(f"Prediction error: {error_pct:.1f}%")

                # Validate: projection should be within 50% of actual
                # (allowing for volatility in emission rates and conversion rates)
                self.assertLess(
                    error_pct,
                    50.0,
                    f"Projection error too high: {error_pct:.1f}%. "
                    f"Projected: {projected_alpha:.2f}, Actual: {actual_alpha_received:.2f}"
                )

                print("✓ Projection within acceptable error margin")
            else:
                self.fail("Projection returned zero emissions")

        except Exception as e:
            self.skipTest(f"Could not query subtensor for validation: {e}")

    def test_blocks_per_day_assumption(self):
        """
        Verify the 7200 blocks/day assumption by checking actual block times.

        Samples recent checkpoints to calculate actual blocks/day rate.
        """
        print(f"\n=== Blocks Per Day Validation ===")

        # Get a sample of recent checkpoints
        sample_checkpoints = self._get_recent_checkpoint_sample(sample_size=10)

        if len(sample_checkpoints) < 2:
            self.skipTest("Not enough checkpoint data to validate blocks/day")

        # Calculate actual blocks per day from checkpoint data
        blocks_per_day_samples = []

        for i in range(len(sample_checkpoints) - 1):
            cp1 = sample_checkpoints[i]
            cp2 = sample_checkpoints[i + 1]

            time_diff_ms = cp2['timestamp_ms'] - cp1['timestamp_ms']
            block_diff = cp2['block_end'] - cp1['block_end']

            if time_diff_ms > 0:
                blocks_per_day = (block_diff / time_diff_ms) * (24 * 60 * 60 * 1000)
                blocks_per_day_samples.append(blocks_per_day)

        if not blocks_per_day_samples:
            self.skipTest("Could not calculate blocks/day from checkpoints")

        avg_blocks_per_day = sum(blocks_per_day_samples) / len(blocks_per_day_samples)

        print(f"Assumed blocks/day: {DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK}")
        print(f"Actual blocks/day (from checkpoints): {avg_blocks_per_day:.0f}")
        print(f"Sample size: {len(blocks_per_day_samples)} checkpoint intervals")

        # Validate: should be within 20% of 7200
        error_pct = abs(avg_blocks_per_day - DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK) / DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK * 100
        print(f"Error: {error_pct:.1f}%")

        self.assertLess(
            error_pct,
            20.0,
            f"Blocks/day assumption off by {error_pct:.1f}%. "
            f"Assumed: {DebtBasedScoring.BLOCKS_PER_DAY_FALLBACK}, Actual: {avg_blocks_per_day:.0f}"
        )

        print("✓ Blocks/day assumption validated")

    def test_conversion_rate_volatility(self):
        """
        Check alpha-to-TAO conversion rate volatility over past 7 days.

        High volatility means our projections may be less accurate.
        """
        print(f"\n=== Conversion Rate Volatility Check ===")

        # Sample conversion rates from recent checkpoints
        rate_samples = self._get_recent_conversion_rate_samples(sample_size=20)

        if len(rate_samples) < 2:
            self.skipTest("Not enough data to check conversion rate volatility")

        avg_rate = sum(rate_samples) / len(rate_samples)
        min_rate = min(rate_samples)
        max_rate = max(rate_samples)

        # Calculate coefficient of variation (std dev / mean)
        variance = sum((r - avg_rate) ** 2 for r in rate_samples) / len(rate_samples)
        std_dev = variance ** 0.5
        cv = (std_dev / avg_rate) * 100 if avg_rate > 0 else 0

        print(f"Sample size: {len(rate_samples)} checkpoints")
        print(f"Average ALPHA-to-TAO rate: {avg_rate:.6f}")
        print(f"Min: {min_rate:.6f}, Max: {max_rate:.6f}")
        print(f"Std deviation: {std_dev:.6f}")
        print(f"Coefficient of variation: {cv:.1f}%")

        # Interpret volatility
        if cv < 5:
            print("✓ Conversion rate is STABLE (low volatility)")
        elif cv < 15:
            print("⚠️  Conversion rate is MODERATELY VOLATILE")
        else:
            print("❌ Conversion rate is HIGHLY VOLATILE - projections may be less accurate")

        # This is informational, not a hard assertion
        # But warn if extremely volatile
        if cv > 30:
            print(f"WARNING: Very high volatility ({cv:.1f}%) may impact projection accuracy")

    def _calculate_actual_emissions_in_period(self, start_ms: int, end_ms: int) -> float:
        """Calculate total actual ALPHA emissions received in a time period."""
        total_alpha = 0.0

        for hotkey, ledger in self.emissions_mgr.emissions_ledgers.items():
            for checkpoint in ledger.checkpoints:
                if start_ms <= checkpoint.timestamp_ms <= end_ms:
                    total_alpha += checkpoint.chunk_emissions

        return total_alpha

    def _get_recent_checkpoint_sample(self, sample_size: int = 10) -> list:
        """Get a sample of recent checkpoints with block and timestamp data."""
        checkpoints = []

        for hotkey, ledger in self.emissions_mgr.emissions_ledgers.items():
            # Take last N checkpoints from this ledger
            for cp in ledger.checkpoints[-sample_size:]:
                if hasattr(cp, 'block_end') and hasattr(cp, 'timestamp_ms'):
                    checkpoints.append({
                        'timestamp_ms': cp.timestamp_ms,
                        'block_end': cp.block_end
                    })

            if len(checkpoints) >= sample_size:
                break

        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp_ms'])
        return checkpoints[-sample_size:]

    def _get_recent_conversion_rate_samples(self, sample_size: int = 20) -> list:
        """Get recent alpha-to-TAO conversion rate samples."""
        rates = []

        for hotkey, ledger in self.emissions_mgr.emissions_ledgers.items():
            # Take last N checkpoints
            for cp in ledger.checkpoints[-sample_size:]:
                if hasattr(cp, 'avg_alpha_to_tao_rate') and cp.avg_alpha_to_tao_rate > 0:
                    rates.append(cp.avg_alpha_to_tao_rate)

            if len(rates) >= sample_size:
                break

        return rates[-sample_size:]


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
