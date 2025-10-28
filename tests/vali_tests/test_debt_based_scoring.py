"""
Unit tests for debt-based scoring algorithm
"""

import unittest
from datetime import datetime, timezone
from vali_objects.vali_dataclasses.debt_ledger import DebtLedger, DebtCheckpoint
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring


class TestDebtBasedScoring(unittest.TestCase):
    """Test debt-based scoring functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.target_cp_duration_ms = 43200000  # 12 hours

    def test_empty_ledgers(self):
        """Test with no ledgers"""
        result = DebtBasedScoring.compute_results({})
        self.assertEqual(result, [])

    def test_single_miner(self):
        """Test with single miner returns weight 1.0"""
        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])
        result = DebtBasedScoring.compute_results({"test_hotkey": ledger})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("test_hotkey", 1.0))

    def test_before_activation_date(self):
        """Test that all weights are zero before November 2025"""
        # Use October 2025 as current time (previous month is September 2025, before November)
        current_time = datetime(2025, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create some ledgers
        ledger1 = DebtLedger(hotkey="hotkey1", checkpoints=[])
        ledger2 = DebtLedger(hotkey="hotkey2", checkpoints=[])

        ledgers = {"hotkey1": ledger1, "hotkey2": ledger2}

        result = DebtBasedScoring.compute_results(ledgers, current_time_ms=current_time_ms)

        # All miners should have zero weight
        self.assertEqual(len(result), 2)
        for hotkey, weight in result:
            self.assertEqual(weight, 0.0)

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0"""
        # Use December 2025 as current time (previous month is November 2025, after activation)
        current_time = datetime(2025, 12, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create ledgers with some performance data
        # Previous month (November 2025) data
        prev_month_checkpoint = datetime(2025, 11, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Current month (December 2025) data
        current_month_checkpoint = datetime(2025, 12, 1, 12, 0, 0, tzinfo=timezone.utc)
        current_month_checkpoint_ms = int(current_month_checkpoint.timestamp() * 1000)

        # Miner 1: Good performance, needs payout
        ledger1 = DebtLedger(hotkey="hotkey1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,
        ))
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=100.0,  # Received 100 ALPHA so far
        ))

        # Miner 2: Better performance, needs more payout
        ledger2 = DebtLedger(hotkey="hotkey2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=10000.0,
            pnl_loss=-2000.0,  # net_pnl = 8000
            total_penalty=1.0,
        ))
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=200.0,  # Received 200 ALPHA so far
        ))

        ledgers = {"hotkey1": ledger1, "hotkey2": ledger2}

        result = DebtBasedScoring.compute_results(ledgers, current_time_ms=current_time_ms)

        # Check that weights sum to 1.0
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Check that miner2 has higher weight (better performance)
        weights_dict = dict(result)
        self.assertGreater(weights_dict["hotkey2"], weights_dict["hotkey1"])

    def test_negative_performance_gets_zero_weight(self):
        """Test that miners with negative performance get zero weight"""
        # Use December 2025 as current time
        current_time = datetime(2025, 12, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 11, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Miner with negative performance
        ledger_negative = DebtLedger(hotkey="negative_miner", checkpoints=[])
        ledger_negative.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=1000.0,
            pnl_loss=-5000.0,  # net_pnl = -4000 (negative)
            total_penalty=1.0,
        ))

        # Miner with positive performance
        ledger_positive = DebtLedger(hotkey="positive_miner", checkpoints=[])
        ledger_positive.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000 (positive)
            total_penalty=1.0,
        ))

        ledgers = {"negative_miner": ledger_negative, "positive_miner": ledger_positive}

        result = DebtBasedScoring.compute_results(ledgers, current_time_ms=current_time_ms, verbose=True)

        # Negative miner should get minimal weight (remaining payout clamped to 0)
        weights_dict = dict(result)
        self.assertEqual(weights_dict["negative_miner"], 0.0)
        self.assertEqual(weights_dict["positive_miner"], 1.0)

    def test_penalty_reduces_needed_payout(self):
        """Test that penalties reduce the needed payout"""
        current_time = datetime(2025, 12, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 11, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Miner 1: No penalty
        ledger1 = DebtLedger(hotkey="no_penalty", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,  # No penalty
        ))

        # Miner 2: 50% penalty (same PnL but lower needed payout)
        ledger2 = DebtLedger(hotkey="with_penalty", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000
            total_penalty=0.5,  # 50% penalty
        ))

        ledgers = {"no_penalty": ledger1, "with_penalty": ledger2}

        result = DebtBasedScoring.compute_results(ledgers, current_time_ms=current_time_ms)

        # Miner with no penalty should get higher weight
        weights_dict = dict(result)
        self.assertGreater(weights_dict["no_penalty"], weights_dict["with_penalty"])

        # Ratio should be approximately 2:1 (4000 vs 2000 needed payout)
        ratio = weights_dict["no_penalty"] / weights_dict["with_penalty"]
        self.assertAlmostEqual(ratio, 2.0, places=1)


if __name__ == '__main__':
    unittest.main()
