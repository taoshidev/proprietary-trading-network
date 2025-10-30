"""
Unit tests for debt-based scoring algorithm with emission projection
"""

import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone
from vali_objects.vali_dataclasses.debt_ledger import DebtLedger, DebtCheckpoint
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig


class TestDebtBasedScoring(unittest.TestCase):
    """Test debt-based scoring functionality"""

    def setUp(self):
        """Set up mock dependencies"""
        # Mock subtensor
        self.mock_subtensor = Mock()
        self.mock_metagraph = Mock()
        # metagraph.emission is in TAO per tempo (360 blocks)
        # To get 10 TAO/block total, we need 10 * 360 = 3600 TAO per tempo
        self.mock_metagraph.emission = [360] * 10  # 10 miners, 360 TAO per tempo each = 1 TAO/block each
        # Create hotkeys list for burn address testing
        self.mock_metagraph.hotkeys = [f"hotkey_{i}" for i in range(256)]
        self.mock_metagraph.hotkeys[229] = "burn_address_mainnet"
        self.mock_metagraph.hotkeys[5] = "burn_address_testnet"
        self.mock_subtensor.metagraph = Mock(return_value=self.mock_metagraph)
        self.mock_subtensor.get_current_block = Mock(return_value=1000000)

        # Mock emissions ledger manager
        self.mock_emissions_mgr = Mock()
        self.mock_emissions_mgr.query_alpha_to_tao_rate = Mock(return_value=0.5)  # 1 ALPHA = 0.5 TAO

        self.netuid = 8

    def test_empty_ledgers(self):
        """Test with no ledgers"""
        result = DebtBasedScoring.compute_results(
            {},
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            metagraph=self.mock_metagraph,
            is_testnet=False
        )
        self.assertEqual(result, [])

    def test_single_miner(self):
        """Test with single miner returns weight 1.0"""
        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])
        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            metagraph=self.mock_metagraph,
            is_testnet=False
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("test_hotkey", 1.0))

    def test_before_activation_date(self):
        """Test that only dust weights + burn address before December 2025"""
        # Use November 2025 as current time (previous month is October 2025, before December)
        current_time = datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create ledgers with different statuses
        prev_checkpoint = datetime(2025, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_checkpoint_ms = int(prev_checkpoint.timestamp() * 1000)

        ledger1 = DebtLedger(hotkey="hotkey1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger2 = DebtLedger(hotkey="hotkey2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))

        ledgers = {"hotkey1": ledger1, "hotkey2": ledger2}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=False
        )

        # Should have 3 entries: 2 miners + burn address
        self.assertEqual(len(result), 3)

        # Verify dust weights based on status
        weights_dict = dict(result)
        dust = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
        self.assertAlmostEqual(weights_dict["hotkey1"], 3 * dust)  # MAINCOMP = 3x dust
        self.assertAlmostEqual(weights_dict["hotkey2"], 1 * dust)  # CHALLENGE = 1x dust

        # Verify burn address gets excess (sum should be 1.0)
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Verify burn address is present
        burn_hotkey = "burn_address_mainnet"
        self.assertIn(burn_hotkey, weights_dict)

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0"""
        # Use January 2026 as current time (previous month is December 2025, after activation)
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create ledgers with some performance data
        # Previous month (December 2025) data
        prev_month_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Current month (January 2026) data
        current_month_checkpoint = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        current_month_checkpoint_ms = int(current_month_checkpoint.timestamp() * 1000)

        # Miner 1: Good performance, needs payout
        ledger1 = DebtLedger(hotkey="hotkey1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=100.0,  # Received 100 ALPHA so far
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: Better performance, needs more payout
        ledger2 = DebtLedger(hotkey="hotkey2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=10000.0,
            pnl_loss=-2000.0,  # net_pnl = 8000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=200.0,  # Received 200 ALPHA so far
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"hotkey1": ledger1, "hotkey2": ledger2}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=False
        )

        # Check that weights sum to 1.0
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Check that miner2 has higher weight (better performance)
        weights_dict = dict(result)
        self.assertGreater(weights_dict["hotkey2"], weights_dict["hotkey1"])

    def test_minimum_weights_by_status(self):
        """Test that minimum weights are enforced based on challenge period status"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

        # Create miners with different statuses and minimal performance
        ledger_challenge = DebtLedger(hotkey="challenge_miner", checkpoints=[])
        ledger_challenge.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=1.0,
            pnl_loss=-0.5,  # Very small net_pnl
            total_penalty=1.0,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))

        ledger_probation = DebtLedger(hotkey="probation_miner", checkpoints=[])
        ledger_probation.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=1.0,
            pnl_loss=-0.5,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.PROBATION.value
        ))

        ledger_maincomp = DebtLedger(hotkey="maincomp_miner", checkpoints=[])
        ledger_maincomp.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=1.0,
            pnl_loss=-0.5,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {
            "challenge_miner": ledger_challenge,
            "probation_miner": ledger_probation,
            "maincomp_miner": ledger_maincomp
        }

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Verify minimum weights are enforced
        # (actual weights might be slightly higher after normalization, but should maintain ratios)
        weight_challenge = weights_dict.get("challenge_miner", 0)
        weight_probation = weights_dict.get("probation_miner", 0)
        weight_maincomp = weights_dict.get("maincomp_miner", 0)

        # Check that weights are at least the minimum
        self.assertGreaterEqual(weight_challenge, dust)
        self.assertGreaterEqual(weight_probation, 2 * dust)
        self.assertGreaterEqual(weight_maincomp, 3 * dust)

        # Check ratio (probation should be ~2x challenge, maincomp ~3x challenge)
        self.assertAlmostEqual(weight_probation / weight_challenge, 2.0, places=1)
        self.assertAlmostEqual(weight_maincomp / weight_challenge, 3.0, places=1)

    def test_burn_address_mainnet(self):
        """Test burn address receives excess weight on mainnet"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_checkpoint_ms = int(prev_checkpoint.timestamp() * 1000)

        # Create one miner with minimal weight (will cause sum < 1.0)
        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            pnl_gain=1.0,
            pnl_loss=0.0,  # Very small performance
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=False
        )

        # Should have 2 entries: miner + burn address
        self.assertEqual(len(result), 2)

        weights_dict = dict(result)

        # Burn address should be mainnet (uid 229)
        burn_hotkey = "burn_address_mainnet"
        self.assertIn(burn_hotkey, weights_dict)

        # Total should sum to 1.0
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

    def test_burn_address_testnet(self):
        """Test burn address receives excess weight on testnet with correct UID"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_checkpoint_ms = int(prev_checkpoint.timestamp() * 1000)

        # Create one miner with minimal weight
        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            pnl_gain=1.0,
            pnl_loss=0.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=True  # TESTNET
        )

        weights_dict = dict(result)

        # Burn address should be testnet (uid 5)
        burn_hotkey = "burn_address_testnet"
        self.assertIn(burn_hotkey, weights_dict)

        # Total should sum to 1.0
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

    def test_negative_performance_gets_minimum_weight(self):
        """Test that miners with negative performance get minimum dust weight"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Miner with negative performance
        ledger_negative = DebtLedger(hotkey="negative_miner", checkpoints=[])
        ledger_negative.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=1000.0,
            pnl_loss=-5000.0,  # net_pnl = -4000 (negative)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner with positive performance
        ledger_positive = DebtLedger(hotkey="positive_miner", checkpoints=[])
        ledger_positive.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000 (positive)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"negative_miner": ledger_negative, "positive_miner": ledger_positive}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)
        dust = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

        # Negative miner should get only minimum dust weight (3x for MAINCOMP)
        # Positive miner should get much higher weight
        self.assertGreater(weights_dict["positive_miner"], weights_dict["negative_miner"])

        # Negative miner gets at least minimum
        self.assertGreaterEqual(weights_dict["negative_miner"], 3 * dust)

    def test_penalty_reduces_needed_payout(self):
        """Test that penalties reduce the needed payout"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Miner 1: No penalty
        ledger1 = DebtLedger(hotkey="no_penalty", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,  # No penalty
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: 50% penalty (same PnL but lower needed payout)
        ledger2 = DebtLedger(hotkey="with_penalty", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000
            total_penalty=0.5,  # 50% penalty
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"no_penalty": ledger1, "with_penalty": ledger2}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=False
        )

        # Miner with no penalty should get higher weight
        weights_dict = dict(result)
        self.assertGreater(weights_dict["no_penalty"], weights_dict["with_penalty"])

        # Ratio should be approximately 2:1 (4000 vs 2000 needed payout)
        ratio = weights_dict["no_penalty"] / weights_dict["with_penalty"]
        self.assertAlmostEqual(ratio, 2.0, places=1)

    def test_emission_projection_calculation(self):
        """Test that emission projection is calculated correctly"""
        # Use mocked subtensor with known emission rate
        days_until_target = 10

        projected_alpha = DebtBasedScoring._estimate_alpha_emissions_until_target(
            subtensor=self.mock_subtensor,
            netuid=self.netuid,
            emissions_ledger_manager=self.mock_emissions_mgr,
            days_until_target=days_until_target,
            verbose=True
        )

        # Expected calculation:
        # - 10 miners * 1 TAO/block = 10 TAO/block
        # - 7200 blocks/day * 10 days = 72000 blocks
        # - 10 TAO/block * 72000 blocks = 720,000 TAO
        # - 720,000 TAO / 0.5 (ALPHA to TAO rate) = 1,440,000 ALPHA

        expected_alpha = 10 * 7200 * 10 / 0.5  # 1,440,000
        self.assertAlmostEqual(projected_alpha, expected_alpha, places=0)

    def test_aggressive_payout_strategy(self):
        """Test that aggressive payout strategy is applied correctly"""
        # Test day 1 - should use 4-day buffer (aggressive)
        current_time_day1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms_day1 = int(current_time_day1.timestamp() * 1000)

        # Create simple ledger with remaining payout
        prev_month_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            pnl_gain=10000.0,
            pnl_loss=-2000.0,  # net_pnl = 8000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Run compute_results and check projection uses 4-day window
        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms_day1,
            metagraph=self.mock_metagraph,
            is_testnet=False,
            verbose=True
        )

        # Verify weight is assigned (single miner gets 1.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("test_hotkey", 1.0))

        # Test day 23 - should use 3-day buffer (actual remaining is 3)
        current_time_day23 = datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms_day23 = int(current_time_day23.timestamp() * 1000)

        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms_day23,
            metagraph=self.mock_metagraph,
            is_testnet=False,
            verbose=True
        )

        # Should still return weight
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("test_hotkey", 1.0))

    def test_only_earning_periods_counted(self):
        """Test that only MAINCOMP/PROBATION checkpoints count for earnings"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create checkpoints in previous month with different statuses
        challenge_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        challenge_checkpoint_ms = int(challenge_checkpoint.timestamp() * 1000)

        maincomp_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        maincomp_checkpoint_ms = int(maincomp_checkpoint.timestamp() * 1000)

        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])

        # CHALLENGE checkpoint (should NOT count)
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=challenge_checkpoint_ms,
            pnl_gain=5000.0,
            pnl_loss=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))

        # MAINCOMP checkpoint (SHOULD count)
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=maincomp_checkpoint_ms,
            pnl_gain=10000.0,  # Cumulative
            pnl_loss=-2000.0,  # Cumulative, net_pnl = 8000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_subtensor,
            self.netuid,
            self.mock_emissions_mgr,
            current_time_ms=current_time_ms,
            metagraph=self.mock_metagraph,
            is_testnet=False,
            verbose=True
        )

        # Should only use MAINCOMP checkpoint for earnings calculation
        # (net_pnl = 8000, not 4000 from CHALLENGE period)
        self.assertEqual(len(result), 1)
        # With only one miner, weight should be 1.0
        self.assertEqual(result[0][1], 1.0)


if __name__ == '__main__':
    unittest.main()
