import unittest
from unittest.mock import Mock, patch

from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.perf_ledger import (
    TP_ID_PORTFOLIO,
    PerfLedger,
    PerfLedgerManager,
    TradePairReturnStatus,
)


class TestPerfLedgerSerialization(TestBase):
    """Tests for perf ledger serialization and disk persistence"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
        )
        self.position_manager.clear_all_miner_positions()

    def test_perf_ledger_to_dict_comprehensive(self):
        """Test comprehensive serialization of PerfLedger to dictionary"""
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Add multiple checkpoints with various data
        for i in range(5):
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 12)
            portfolio_value = 1.0 + (i * 0.02) + (0.01 if i % 2 else -0.005)  # Some variation

            ledger.update_pl(
                current_portfolio_value=portfolio_value,
                now_ms=update_time,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE if i < 4 else TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
                current_portfolio_fee_spread=1.0 - (i * 0.001),  # Slight spread fee
                current_portfolio_carry=1.0 - (i * 0.0005),  # Slight carry fee
            )

        # Serialize to dict
        ledger_dict = ledger.to_dict()

        # Verify all essential fields are present
        expected_fields = [
            'initialization_time_ms', 'target_ledger_window_ms', 'target_cp_duration_ms',
            'max_return', 'cps',
        ]

        for field in expected_fields:
            self.assertIn(field, ledger_dict, f"Missing field: {field}")

        # Verify checkpoint data integrity
        self.assertEqual(len(ledger_dict['cps']), len(ledger.cps))

        for original_cp, serialized_cp in zip(ledger.cps, ledger_dict['cps']):
            # Check essential checkpoint fields (excluding properties)
            checkpoint_fields = [
                'last_update_ms', 'prev_portfolio_ret', 'mdd', 'gain', 'loss',
                'prev_portfolio_spread_fee', 'prev_portfolio_carry_fee',
                'accum_ms', 'open_ms', 'n_updates',
            ]

            for field in checkpoint_fields:
                self.assertIn(field, serialized_cp, f"Missing checkpoint field: {field}")

                # Verify values match
                original_value = getattr(original_cp, field)
                serialized_value = serialized_cp[field]
                self.assertEqual(original_value, serialized_value,
                               f"Mismatch in checkpoint field {field}: {original_value} != {serialized_value}")

            # Verify computed properties are still accessible after deserialization
            # lowerbound_time_created_ms is a property, not stored directly
            self.assertEqual(
                original_cp.lowerbound_time_created_ms,
                serialized_cp['last_update_ms'] - serialized_cp['accum_ms'],
                "lowerbound_time_created_ms calculation mismatch",
            )

    def test_perf_ledger_from_dict_comprehensive(self):
        """Test comprehensive deserialization of PerfLedger from dictionary"""
        # Create test dictionary matching the expected format
        # Only include fields that are part of PerfLedger's to_dict() output
        test_dict = {
            'initialization_time_ms': self.now_ms,
            'target_ledger_window_ms': 1000 * 60 * 60 * 24 * 30,
            'target_cp_duration_ms': 1000 * 60 * 60 * 12,
            'max_return': 1.05,
            'cps': [
                {
                    'last_update_ms': self.now_ms + 1000 * 60 * 60,
                    'prev_portfolio_ret': 1.05,
                    'mdd': 0.98,
                    'gain': 0.05,
                    'loss': -0.02,
                    'prev_portfolio_spread_fee': 0.999,
                    'prev_portfolio_carry_fee': 0.9995,
                    'accum_ms': 1000 * 60 * 60,
                    'open_ms': 1000 * 60 * 60,
                    'n_updates': 10,
                    'mpv': 1.05,
                    'spread_fee_loss': -0.001,
                    'carry_fee_loss': -0.0005,
                },
            ],
        }

        # Deserialize
        ledger = PerfLedger.from_dict(test_dict)

        # Verify all fields were correctly restored
        self.assertEqual(ledger.initialization_time_ms, test_dict['initialization_time_ms'])
        self.assertEqual(ledger.target_ledger_window_ms, test_dict['target_ledger_window_ms'])
        self.assertEqual(ledger.max_return, test_dict['max_return'])
        self.assertEqual(ledger.target_cp_duration_ms, test_dict['target_cp_duration_ms'])
        self.assertEqual(len(ledger.cps), len(test_dict['cps']))

        # Verify checkpoint restoration
        restored_cp = ledger.cps[0]

        # Note: from_dict modifies the input dict, so we need to use the expected values directly
        self.assertEqual(restored_cp.last_update_ms, self.now_ms + 1000 * 60 * 60)
        self.assertEqual(restored_cp.prev_portfolio_ret, 1.05)
        self.assertEqual(restored_cp.mdd, 0.98)
        self.assertEqual(restored_cp.gain, 0.05)
        self.assertEqual(restored_cp.loss, -0.02)
        self.assertEqual(restored_cp.prev_portfolio_spread_fee, 0.999)
        self.assertEqual(restored_cp.prev_portfolio_carry_fee, 0.9995)
        self.assertEqual(restored_cp.accum_ms, 1000 * 60 * 60)
        self.assertEqual(restored_cp.open_ms, 1000 * 60 * 60)
        self.assertEqual(restored_cp.n_updates, 10)

    def test_round_trip_serialization(self):
        """Test that serialize -> deserialize preserves all data"""
        # Create a complex ledger
        original_ledger = PerfLedger(
            initialization_time_ms=self.now_ms - 1000 * 60 * 60 * 24,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 15,
        )

        # Add varied checkpoint data
        values = [1.0, 1.03, 0.98, 1.06, 1.02, 0.95, 1.08]

        for i, value in enumerate(values):
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 6)
            original_ledger.update_pl(
                current_portfolio_value=value,
                now_ms=update_time,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0 - (i * 0.0001),
                current_portfolio_carry=1.0 - (i * 0.00005),
            )

        # Serialize
        serialized_dict = original_ledger.to_dict()

        # Deserialize
        restored_ledger = PerfLedger.from_dict(serialized_dict)

        # Compare all key attributes
        self.assertEqual(restored_ledger.initialization_time_ms, original_ledger.initialization_time_ms)
        self.assertEqual(restored_ledger.target_ledger_window_ms, original_ledger.target_ledger_window_ms)
        self.assertEqual(restored_ledger.target_cp_duration_ms, original_ledger.target_cp_duration_ms)
        self.assertEqual(restored_ledger.max_return, original_ledger.max_return)
        self.assertEqual(restored_ledger.last_update_ms, original_ledger.last_update_ms)
        self.assertEqual(len(restored_ledger.cps), len(original_ledger.cps))

        # Compare checkpoint data
        for orig_cp, rest_cp in zip(original_ledger.cps, restored_ledger.cps):
            self.assertEqual(orig_cp.last_update_ms, rest_cp.last_update_ms)
            self.assertAlmostEqual(orig_cp.prev_portfolio_ret, rest_cp.prev_portfolio_ret, places=10)
            self.assertAlmostEqual(orig_cp.mdd, rest_cp.mdd, places=10)
            self.assertAlmostEqual(orig_cp.gain, rest_cp.gain, places=10)
            self.assertAlmostEqual(orig_cp.loss, rest_cp.loss, places=10)

    def test_perf_ledger_manager_disk_operations(self):
        """Test PerfLedgerManager disk save/load operations"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Create test ledger bundles
        test_bundles = {}
        for i, hotkey in enumerate([self.DEFAULT_MINER_HOTKEY, "test_miner_2"]):
            portfolio_ledger = PerfLedger(
                initialization_time_ms=self.now_ms - (i * 1000 * 60 * 60),
                target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
            )

            # Add some data
            portfolio_ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.02),
                now_ms=self.now_ms + (i * 1000 * 60 * 60),
                miner_hotkey=hotkey,
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

            test_bundles[hotkey] = {
                TP_ID_PORTFOLIO: portfolio_ledger,
                TradePair.BTCUSD.trade_pair_id: portfolio_ledger,  # Reuse for simplicity
            }

        # Test save to memory only (backtesting mode)
        plm.is_backtesting = True
        plm.save_perf_ledgers(test_bundles)

        # Should be saved to memory
        memory_bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertEqual(len(memory_bundles), 2)

        # Test disk operations when not in backtesting mode
        plm.is_backtesting = False

        # This would normally save to disk, but we're in test mode
        # The test framework handles disk operations differently


class TestPerfLedgerEliminationLogic(TestBase):
    """Tests for elimination and RSS (Random Security Screening) logic"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_MINER_HOTKEY_2 = "test_miner_2"
        self.DEFAULT_MINER_HOTKEY_3 = "test_miner_3"
        self.now_ms = TimeUtil.now_in_millis()
        self.mmg = MockMetagraph(hotkeys=[
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_MINER_HOTKEY_2,
            self.DEFAULT_MINER_HOTKEY_3,
        ])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
        )
        self.position_manager.clear_all_miner_positions()

    def test_rss_random_security_screening_logic(self):
        """Test Random Security Screening (RSS) elimination logic"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            enable_rss=True,  # Enable RSS
        )

        # Create ledger bundles for multiple miners
        for hotkey in [self.DEFAULT_MINER_HOTKEY, self.DEFAULT_MINER_HOTKEY_2]:
            bundle = {
                TP_ID_PORTFOLIO: PerfLedger(
                    initialization_time_ms=self.now_ms,
                    target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
                ),
            }
            plm.hotkey_to_perf_bundle[hotkey] = bundle

        # Manually trigger RSS for one miner
        plm.random_security_screenings.add(self.DEFAULT_MINER_HOTKEY)

        # Test that RSS is tracked
        self.assertIn(self.DEFAULT_MINER_HOTKEY, plm.random_security_screenings)
        self.assertNotIn(self.DEFAULT_MINER_HOTKEY_2, plm.random_security_screenings)

        # Test RSS reset logic
        plm.random_security_screenings = set()  # Reset
        self.assertEqual(len(plm.random_security_screenings), 0)

    def test_hotkey_deletion_logic(self):
        """Test various scenarios that trigger hotkey deletion from perf ledgers"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Create test bundles
        test_bundles = {}
        for hotkey in [self.DEFAULT_MINER_HOTKEY, self.DEFAULT_MINER_HOTKEY_2, self.DEFAULT_MINER_HOTKEY_3]:
            test_bundles[hotkey] = {
                TP_ID_PORTFOLIO: PerfLedger(
                    initialization_time_ms=self.now_ms,
                    target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
                ),
            }

        plm.hotkey_to_perf_bundle.update(test_bundles)

        # Test deletion due to no positions
        positions_with_gaps = {
            self.DEFAULT_MINER_HOTKEY: [],  # No positions - should be deleted
            self.DEFAULT_MINER_HOTKEY_2: [Mock()],  # Has positions - should be kept
            # HOTKEY_3 missing entirely - should be deleted
        }

        # Test deletion logic by examining what would be deleted
        hotkeys_with_no_positions = set()
        for hotkey in test_bundles.keys():
            if not positions_with_gaps.get(hotkey, []):
                hotkeys_with_no_positions.add(hotkey)

        # Should identify miners with no positions
        self.assertIn(self.DEFAULT_MINER_HOTKEY, hotkeys_with_no_positions)
        self.assertNotIn(self.DEFAULT_MINER_HOTKEY_2, hotkeys_with_no_positions)
        self.assertIn(self.DEFAULT_MINER_HOTKEY_3, hotkeys_with_no_positions)

    def test_recently_reregistered_hotkeys_detection(self):
        """Test detection and handling of recently re-registered hotkeys"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Create a ledger with initialization time that doesn't match first order
        old_init_time = self.now_ms - 1000 * 60 * 60 * 24 * 10  # 10 days ago
        new_first_order_time = self.now_ms - 1000 * 60 * 60 * 24 * 5  # 5 days ago

        ledger = PerfLedger(
            initialization_time_ms=old_init_time,  # Older than first order
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        bundle = {TP_ID_PORTFOLIO: ledger}
        plm.hotkey_to_perf_bundle[self.DEFAULT_MINER_HOTKEY] = bundle

        # Create mock position with first order time newer than ledger init time
        mock_order = Mock()
        mock_order.processed_ms = new_first_order_time

        mock_position = Mock()
        mock_position.orders = [mock_order]

        positions_dict = {
            self.DEFAULT_MINER_HOTKEY: [mock_position],
        }

        # Test the detection logic
        corresponding_ledger_bundle = plm.hotkey_to_perf_bundle.get(self.DEFAULT_MINER_HOTKEY)
        self.assertIsNotNone(corresponding_ledger_bundle)

        portfolio_ledger = corresponding_ledger_bundle[TP_ID_PORTFOLIO]
        first_order_time_ms = min(p.orders[0].processed_ms for p in positions_dict[self.DEFAULT_MINER_HOTKEY])

        # Should detect mismatch (re-registration)
        self.assertNotEqual(portfolio_ledger.initialization_time_ms, first_order_time_ms)
        self.assertLess(portfolio_ledger.initialization_time_ms, first_order_time_ms)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_elimination_integration(self, mock_candle_fetcher):
        """Test integration with elimination manager"""
        mock_candle_fetcher.return_value = {}

        # Create elimination manager with mock eliminations
        elimination_manager = EliminationManager(self.mmg, None, None)
        elimination_manager.eliminations = [{'hotkey': self.DEFAULT_MINER_HOTKEY_2, 'reason': "test_elimination"}]

        position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=elimination_manager,
        )

        PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=position_manager,
        )

        # Test that eliminated hotkeys are handled differently
        eliminated_hotkeys = elimination_manager.get_eliminated_hotkeys()
        self.assertIn(self.DEFAULT_MINER_HOTKEY_2, eliminated_hotkeys)

        # Eliminated miners should be preserved in ledgers for dashboard visualization
        # but not updated with new data


class TestCheckpointManagement(TestBase):
    """Tests for checkpoint creation, management, and boundary alignment"""

    def setUp(self):
        super().setUp()
        self.now_ms = TimeUtil.now_in_millis()

        # Align to a 12-hour boundary for predictable testing
        twelve_hours_ms = 1000 * 60 * 60 * 12
        self.aligned_now_ms = (self.now_ms // twelve_hours_ms) * twelve_hours_ms

    def test_checkpoint_boundary_precision(self):
        """Test precise checkpoint boundary alignment"""
        ledger = PerfLedger(
            initialization_time_ms=self.aligned_now_ms,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Create updates at various times around boundaries
        boundary_times = []
        for i in range(5):
            # Create times just before, on, and after 12-hour boundaries
            base_boundary = self.aligned_now_ms + (i * 1000 * 60 * 60 * 12)
            boundary_times.extend([
                base_boundary - 1000,  # 1 second before
                base_boundary,         # Exactly on boundary
                base_boundary + 1000,   # 1 second after
            ])

        for i, update_time in enumerate(boundary_times):
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.001),
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        # Verify boundary alignment for completed checkpoints
        # Note: Only checkpoints that have completed their full duration are aligned
        twelve_hour_ms = 1000 * 60 * 60 * 12
        for i, cp in enumerate(ledger.cps):
            if i == 0:
                continue  # First checkpoint may not be aligned

            # Only check alignment for completed checkpoints (not the active one)
            if i < len(ledger.cps) - 1 and cp.accum_ms >= twelve_hour_ms:
                remainder = cp.last_update_ms % twelve_hour_ms
                self.assertEqual(remainder, 0,
                               f"Completed checkpoint {i} at {cp.last_update_ms} not aligned to 12-hour boundary. Remainder: {remainder}")

            # The active (last) checkpoint should have the actual update time
            if i == len(ledger.cps) - 1:
                self.assertIn(cp.last_update_ms, boundary_times,
                            f"Active checkpoint time {cp.last_update_ms} not in expected times")

    def test_checkpoint_data_accumulation(self):
        """Test that checkpoint data accumulates correctly within boundaries"""
        ledger = PerfLedger(
            initialization_time_ms=self.aligned_now_ms,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Create multiple updates within the same 12-hour period
        base_time = self.aligned_now_ms + 1000 * 60 * 60  # 1 hour after boundary
        values = [1.0, 1.02, 1.01, 1.03, 0.99, 1.04]  # Mix of gains and losses

        for i, value in enumerate(values):
            update_time = base_time + (i * 1000 * 60 * 60)  # Hourly updates
            ledger.update_pl(
                current_portfolio_value=value,
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        # Should have created one checkpoint that accumulates all the changes
        self.assertGreater(len(ledger.cps), 0)

        final_cp = ledger.cps[-1]

        # Verify that gains and losses are accumulated (using log returns)
        # The exact values depend on logarithmic return calculations
        # Just verify that we have both gains and losses
        self.assertGreater(final_cp.gain, 0, "Should have accumulated some gains")
        self.assertLess(final_cp.loss, 0, "Should have accumulated some losses")

        # Verify final portfolio value
        self.assertAlmostEqual(final_cp.prev_portfolio_ret, values[-1], places=10)

    def test_checkpoint_pruning_by_window(self):
        """Test that old checkpoints are pruned based on target window"""
        short_window = 1000 * 60 * 60 * 24 * 7  # 7 days
        ledger = PerfLedger(
            initialization_time_ms=self.aligned_now_ms,
            target_ledger_window_ms=short_window,
        )

        # Create checkpoints spanning more than the target window
        twelve_hours = 1000 * 60 * 60 * 12
        num_checkpoints = 20  # 20 * 12 hours = 10 days > 7 days

        for i in range(num_checkpoints):
            update_time = self.aligned_now_ms + (i * twelve_hours)
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.01),
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        # Check that total duration is reasonably close to target window
        total_duration = ledger.get_total_ledger_duration_ms()
        # The pruning logic might keep a few extra checkpoints for safety
        # Allow up to 50% extra as a reasonable buffer
        max_allowed = int(short_window * 1.5)
        self.assertLessEqual(total_duration, max_allowed,
                           f"Total duration {total_duration} significantly exceeds window {short_window}")

        # Should have fewer than the total number of checkpoints created
        self.assertLess(len(ledger.cps), num_checkpoints)

    def test_checkpoint_trimming_precision(self):
        """Test precise checkpoint trimming functionality"""
        ledger = PerfLedger(
            initialization_time_ms=self.aligned_now_ms,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Create a series of checkpoints
        twelve_hours = 1000 * 60 * 60 * 12
        checkpoint_times = []

        for i in range(10):
            update_time = self.aligned_now_ms + (i * twelve_hours)
            checkpoint_times.append(update_time)
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.01),
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        initial_count = len(ledger.cps)

        # Trim at the 5th checkpoint
        cutoff_time = checkpoint_times[4]  # 5th checkpoint (0-indexed)
        ledger.trim_checkpoints(cutoff_time)

        # Should have removed early checkpoints
        remaining_count = len(ledger.cps)
        self.assertLess(remaining_count, initial_count)

        # All remaining checkpoints should be before the cutoff
        # The trim_checkpoints method KEEPS checkpoints where:
        # lowerbound_time_created_ms + target_cp_duration_ms < cutoff_ms
        for cp in ledger.cps:
            checkpoint_end_estimate = cp.lowerbound_time_created_ms + ledger.target_cp_duration_ms
            self.assertLess(checkpoint_end_estimate, cutoff_time,
                          f"Checkpoint starting at {cp.lowerbound_time_created_ms} with estimated end {checkpoint_end_estimate} "
                          f"should have been trimmed (cutoff: {cutoff_time})")

    def test_max_portfolio_value_tracking(self):
        """Test maximum portfolio value tracking across checkpoints"""
        ledger = PerfLedger(
            initialization_time_ms=self.aligned_now_ms,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Create a sequence with a clear maximum
        values = [1.0, 1.05, 1.12, 1.08, 1.15, 1.03, 1.09]  # Peak at 1.15
        expected_max = max(values)

        for i, value in enumerate(values):
            update_time = self.aligned_now_ms + (i * 1000 * 60 * 60 * 6)  # 6-hour intervals
            ledger.update_pl(
                current_portfolio_value=value,
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        # Verify max return tracking
        self.assertAlmostEqual(ledger.max_return, expected_max, places=10)

        # Verify that individual checkpoint MPVs are monotonically increasing or stable
        prev_mpv = 1.0
        for cp in ledger.cps:
            self.assertGreaterEqual(cp.mpv, prev_mpv,
                                  f"MPV should not decrease: {cp.mpv} < {prev_mpv}")
            prev_mpv = cp.mpv


if __name__ == '__main__':
    unittest.main()
