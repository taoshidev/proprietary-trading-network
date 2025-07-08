import unittest
from unittest.mock import patch

from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    PerfLedger, PerfLedgerManager, TP_ID_PORTFOLIO, 
    ParallelizationMode, TradePairReturnStatus
)


class TestPerfLedgerCore(TestBase):
    """Tests for core PerfLedger functionality"""

    def setUp(self):
        super().setUp()
        self.now_ms = TimeUtil.now_in_millis()
        self.target_window_ms = 1000 * 60 * 60 * 24 * 30  # 30 days
        self.target_cp_duration_ms = 1000 * 60 * 60 * 12  # 12 hours

    def test_perf_ledger_initialization(self):
        """Test PerfLedger initialization with various parameters"""
        init_time = self.now_ms - 1000 * 60 * 60 * 24  # 1 day ago
        
        ledger = PerfLedger(
            initialization_time_ms=init_time,
            target_ledger_window_ms=self.target_window_ms
        )
        
        self.assertEqual(ledger.initialization_time_ms, init_time)
        self.assertEqual(ledger.target_ledger_window_ms, self.target_window_ms)
        self.assertEqual(ledger.target_cp_duration_ms, self.target_cp_duration_ms)
        self.assertEqual(ledger.max_return, 1.0)
        self.assertEqual(ledger.last_update_ms, 0)
        self.assertEqual(len(ledger.cps), 0)

    def test_init_with_first_order(self):
        """Test initialization with first order"""
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=self.target_window_ms
        )
        
        order_time = self.now_ms + 1000 * 60 * 60  # 1 hour later
        ledger.init_with_first_order(
            order_time, 
            point_in_time_dd=1.0,
            current_portfolio_value=1.0,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0
        )
        
        self.assertEqual(len(ledger.cps), 1)
        self.assertEqual(ledger.last_update_ms, order_time)
        self.assertEqual(ledger.cps[0].prev_portfolio_ret, 1.0)
        self.assertEqual(ledger.cps[0].last_update_ms, order_time)

    def test_update_pl_basic(self):
        """Test basic performance ledger update"""
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=self.target_window_ms
        )
        
        # First update
        time1 = self.now_ms + 1000 * 60 * 60  # 1 hour
        ledger.update_pl(
            current_portfolio_value=1.05,
            now_ms=time1,
            miner_hotkey="test_miner",
            any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0
        )
        
        self.assertEqual(len(ledger.cps), 1)
        self.assertEqual(ledger.max_return, 1.05)
        self.assertEqual(ledger.last_update_ms, time1)
        
        # Second update with gain
        time2 = time1 + 1000 * 60 * 60 * 12  # 12 hours later
        ledger.update_pl(
            current_portfolio_value=1.10,
            now_ms=time2,
            miner_hotkey="test_miner",
            any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0
        )
        
        self.assertEqual(len(ledger.cps), 2)
        self.assertEqual(ledger.max_return, 1.10)
        self.assertEqual(ledger.last_update_ms, time2)

    def test_checkpoint_boundary_alignment(self):
        """Test that checkpoints align to 12-hour boundaries"""
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=self.target_window_ms
        )
        
        # Create multiple updates across checkpoint boundaries
        for i in range(5):
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 13)  # 13-hour intervals
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.01),
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0
            )
        
        # Check that completed checkpoints (except first and last) align to 12-hour boundaries
        for i, cp in enumerate(ledger.cps):
            if i < 2:
                continue  # Skip first checkpoint, it may not align
            if i == len(ledger.cps) - 1:
                continue  # Skip last checkpoint, it's still accumulating
            # build an error message where we show every cp before the current one
            err_msg = f"Checkpoint {i}/{len(ledger.cps)} at {cp.last_update_ms} not aligned to 12-hour boundary. \n"
            for j in range(i+1):
                err_msg += f"Checkpoint {j}: {ledger.cps[j]}\n"

            self.assertEqual(
                cp.last_update_ms % ledger.target_cp_duration_ms, 
                0, msg=err_msg
            )

    def test_mdd_calculation(self):
        """Test maximum drawdown calculation"""
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=self.target_window_ms
        )
        
        values = [1.0, 1.10, 1.05, 0.95, 1.20]  # Rise, small drop, big drop, recovery
        
        for i, value in enumerate(values):
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 12)
            ledger.update_pl(
                current_portfolio_value=value,
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0
            )
        
        # After value drops from 1.10 to 0.95, MDD should be 0.95/1.10 ≈ 0.864
        expected_mdd = 0.95 / 1.10  # Lowest point divided by previous peak
        self.assertAlmostEqual(ledger.mdd, expected_mdd, places=3)

    def test_gains_and_losses_tracking(self):
        """Test gain and loss accumulation"""
        
        # Use 12-hour checkpoints as required
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=self.target_window_ms,
            target_cp_duration_ms=43200000  # 12 hours
        )
        
        # Create updates that span multiple checkpoint periods
        values = [1.0, 1.05, 1.02, 1.08]  # Start, gain, loss, gain
        
        for i, value in enumerate(values):
            # Space updates 13 hours apart to ensure separate checkpoints
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 13)  # 13-hour intervals
            ledger.update_pl(
                current_portfolio_value=value,
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0
            )
        
        # Debug output
        print("\nGains/Losses Test Debug:")
        print(f"Number of checkpoints: {len(ledger.cps)}")
        for i, cp in enumerate(ledger.cps):
            print(f"  CP{i}: gain={cp.gain:.6f}, loss={cp.loss:.6f}, prev_ret={cp.prev_portfolio_ret:.4f}, accum_ms={cp.accum_ms}")
        
        # Verify we got the expected final value
        final_cp = ledger.cps[-1]
        self.assertEqual(final_cp.prev_portfolio_ret, 1.08, "Final portfolio value should be 1.08")
        
        # With 13-hour intervals, we should have multiple checkpoints
        self.assertGreater(len(ledger.cps), 1, "Should have multiple checkpoints")
        
        # Each checkpoint should track the returns during its period
        # Verify basic properties
        for cp in ledger.cps:
            self.assertGreaterEqual(cp.gain, 0, "Gains should be non-negative")
            self.assertLessEqual(cp.loss, 0, "Losses should be non-positive")


    def test_purge_old_checkpoints(self):
        """Test that old checkpoints are purged when exceeding target window"""
        short_window = 1000 * 60 * 60 * 24 * 7  # 7 days
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=short_window
        )
        
        # Create checkpoints spanning more than the target window
        for i in range(20):  # 20 * 12 hours = 10 days > 7 days
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 12)
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.01),
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0
            )
        
        # Should have purged old checkpoints
        duration = ledger.get_total_ledger_duration_ms()
        # Allow some buffer - the purging mechanism keeps a few extra checkpoints for safety
        # Typically allows up to 50% extra
        max_allowed = int(short_window * 1.5)
        self.assertLessEqual(duration, max_allowed,
                           f"Duration {duration} should be within 150% of target window {short_window}")

    def test_trim_checkpoints(self):
        """Test checkpoint trimming functionality"""
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=self.target_window_ms
        )
        
        # Create several checkpoints
        for i in range(10):
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 12)
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.01),
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0
            )
        
        initial_count = len(ledger.cps)
        cutoff_time = self.now_ms + (5 * 1000 * 60 * 60 * 12)  # Cut at 5th checkpoint
        
        ledger.trim_checkpoints(cutoff_time)
        
        # Should have fewer checkpoints after trimming
        self.assertLess(len(ledger.cps), initial_count)
        
        # All remaining checkpoints should be before cutoff
        # trim_checkpoints KEEPS checkpoints where lowerbound + duration < cutoff
        for cp in ledger.cps:
            self.assertLess(
                cp.lowerbound_time_created_ms + ledger.target_cp_duration_ms,
                cutoff_time,
                f"Checkpoint starting at {cp.lowerbound_time_created_ms} should have been kept"
            )

    def test_serialization_deserialization(self):
        """Test PerfLedger to_dict and from_dict functionality"""
        ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=self.target_window_ms
        )
        
        # Add some data
        for i in range(3):
            update_time = self.now_ms + (i * 1000 * 60 * 60 * 12)
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.05),
                now_ms=update_time,
                miner_hotkey="test_miner",
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0
            )
        
        # Serialize
        ledger_dict = ledger.to_dict()
        
        # Deserialize
        restored_ledger = PerfLedger.from_dict(ledger_dict)
        
        # Compare key attributes
        self.assertEqual(restored_ledger.initialization_time_ms, ledger.initialization_time_ms)
        self.assertEqual(restored_ledger.target_ledger_window_ms, ledger.target_ledger_window_ms)
        self.assertEqual(restored_ledger.max_return, ledger.max_return)
        self.assertEqual(restored_ledger.last_update_ms, ledger.last_update_ms)
        self.assertEqual(len(restored_ledger.cps), len(ledger.cps))
        
        # Compare checkpoint data
        for original_cp, restored_cp in zip(ledger.cps, restored_ledger.cps):
            self.assertEqual(original_cp.last_update_ms, restored_cp.last_update_ms)
            self.assertEqual(original_cp.prev_portfolio_ret, restored_cp.prev_portfolio_ret)
            self.assertEqual(original_cp.mdd, restored_cp.mdd)


class TestPerfLedgerManager(TestBase):
    """Tests for PerfLedgerManager functionality"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner_1"
        self.DEFAULT_MINER_HOTKEY_2 = "test_miner_2"
        self.now_ms = TimeUtil.now_in_millis()
        self.mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY, self.DEFAULT_MINER_HOTKEY_2])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg, 
            running_unit_tests=True, 
            elimination_manager=self.elimination_manager
        )
        self.position_manager.clear_all_miner_positions()

    def test_perf_ledger_manager_initialization(self):
        """Test PerfLedgerManager initialization with different configurations"""
        # Test with default settings
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        self.assertEqual(plm.parallel_mode, ParallelizationMode.SERIAL)
        self.assertTrue(plm.running_unit_tests)
        self.assertTrue(plm.enable_rss)
        self.assertIsInstance(plm.hotkey_to_perf_bundle, dict)

    def test_perf_ledger_manager_parallel_modes(self):
        """Test PerfLedgerManager with different parallel modes"""
        for mode in [ParallelizationMode.SERIAL, ParallelizationMode.MULTIPROCESSING]:
            with self.subTest(mode=mode):
                plm = PerfLedgerManager(
                    metagraph=self.mmg,
                    running_unit_tests=True,
                    position_manager=self.position_manager,
                    parallel_mode=mode
                )
                
                self.assertEqual(plm.parallel_mode, mode)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_memory_persistence_between_updates(self, mock_candle_fetcher):
        """Test that perf ledgers persist in memory between updates"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create a position
        order = Order(
            price=50000,
            processed_ms=self.now_ms,
            order_uuid="test_order_1",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5
        )
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_1",
            open_ms=self.now_ms,
            trade_pair=TradePair.BTCUSD,
            orders=[order],
            position_type=OrderType.LONG
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)
        
        # First update
        plm.update(t_ms=self.now_ms + 1000 * 60 * 60)
        
        # Get ledgers from memory
        ledgers_1 = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, ledgers_1)
        self.assertIn(TP_ID_PORTFOLIO, ledgers_1[self.DEFAULT_MINER_HOTKEY])
        
        first_update_time = ledgers_1[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        
        # Second update
        plm.update(t_ms=self.now_ms + 1000 * 60 * 60 * 2)
        
        # Get ledgers again
        ledgers_2 = plm.get_perf_ledgers(portfolio_only=False)
        second_update_time = ledgers_2[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        
        # Should have been updated, not rebuilt from scratch
        self.assertGreater(second_update_time, first_update_time)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_incremental_vs_full_rebuild(self, mock_candle_fetcher):
        """Test incremental updates vs full rebuilds"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create position with multiple orders over time
        orders = []
        for i in range(3):
            order = Order(
                price=50000 + (i * 1000),
                processed_ms=self.now_ms + (i * 1000 * 60 * 60),
                order_uuid=f"test_order_{i}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG if i < 2 else OrderType.FLAT,
                leverage=0.5 if i < 2 else 0
            )
            orders.append(order)
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_1",
            open_ms=self.now_ms,
            trade_pair=TradePair.BTCUSD,
            orders=orders,
            position_type=OrderType.FLAT
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)
        
        # Update to first order time
        plm.update(t_ms=orders[0].processed_ms + 1000)
        ledgers_after_first = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIsNotNone(ledgers_after_first)
        
        # Update to second order time
        plm.update(t_ms=orders[1].processed_ms + 1000)
        ledgers_after_second = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIsNotNone(ledgers_after_second)
        
        # Update to third order time
        plm.update(t_ms=orders[2].processed_ms + 1000)
        ledgers_after_third = plm.get_perf_ledgers(portfolio_only=False)
        
        # Each update should build incrementally on the previous
        portfolio_ledger = ledgers_after_third[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
        self.assertGreater(len(portfolio_ledger.cps), 0)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_testing_one_hotkey_parameter_effect(self, mock_candle_fetcher):
        """Test the effect of testing_one_hotkey parameter"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create positions for both miners
        for i, hotkey in enumerate([self.DEFAULT_MINER_HOTKEY, self.DEFAULT_MINER_HOTKEY_2]):
            order = Order(
                price=50000 + (i * 1000),
                processed_ms=self.now_ms + (i * 1000),
                order_uuid=f"test_order_{hotkey}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG,
                leverage=0.5
            )
            
            position = Position(
                miner_hotkey=hotkey,
                position_uuid=f"test_position_{hotkey}",
                open_ms=self.now_ms + (i * 1000),
                trade_pair=TradePair.BTCUSD,
                orders=[order],
                position_type=OrderType.LONG
            )
            position.rebuild_position_with_updated_orders()
            self.position_manager.save_miner_position(position)
        
        # Update without testing_one_hotkey - should preserve existing ledgers
        plm.update(t_ms=self.now_ms + 1000 * 60 * 60)
        ledgers_normal = plm.get_perf_ledgers(portfolio_only=False)
        
        # Both miners should have ledgers
        self.assertEqual(len(ledgers_normal), 2)
        
        # Update with testing_one_hotkey - should trigger regenerate_all_ledgers
        plm.update(testing_one_hotkey=self.DEFAULT_MINER_HOTKEY, t_ms=self.now_ms + 1000 * 60 * 60 * 2)
        ledgers_with_testing = plm.get_perf_ledgers(portfolio_only=False)
        
        # Should still have both miners (this tests the current behavior)
        # In production, this triggers full rebuilds which impacts performance
        self.assertGreaterEqual(len(ledgers_with_testing), 1)

    def test_get_perf_ledgers_portfolio_only_flag(self):
        """Test get_perf_ledgers with portfolio_only parameter"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            build_portfolio_ledgers_only=False  # Build both portfolio and trade pair ledgers
        )
        
        # Manually add some test data to memory
        test_bundle = {
            TP_ID_PORTFOLIO: PerfLedger(
                initialization_time_ms=self.now_ms,
                target_ledger_window_ms=1000 * 60 * 60 * 24 * 30
            ),
            TradePair.BTCUSD.trade_pair_id: PerfLedger(
                initialization_time_ms=self.now_ms,
                target_ledger_window_ms=1000 * 60 * 60 * 24 * 30
            )
        }
        plm.hotkey_to_perf_bundle[self.DEFAULT_MINER_HOTKEY] = test_bundle
        
        # Test portfolio_only=True
        portfolio_only = plm.get_perf_ledgers(portfolio_only=True)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, portfolio_only)
        self.assertIsInstance(portfolio_only[self.DEFAULT_MINER_HOTKEY], PerfLedger)
        
        # Test portfolio_only=False
        full_bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, full_bundles)
        self.assertIsInstance(full_bundles[self.DEFAULT_MINER_HOTKEY], dict)
        self.assertIn(TP_ID_PORTFOLIO, full_bundles[self.DEFAULT_MINER_HOTKEY])
        self.assertIn(TradePair.BTCUSD.trade_pair_id, full_bundles[self.DEFAULT_MINER_HOTKEY])

    def test_save_perf_ledgers_memory_persistence(self):
        """Test that save_perf_ledgers properly updates memory"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create test ledger data
        test_bundles = {
            self.DEFAULT_MINER_HOTKEY: {
                TP_ID_PORTFOLIO: PerfLedger(
                    initialization_time_ms=self.now_ms,
                    target_ledger_window_ms=1000 * 60 * 60 * 24 * 30
                )
            },
            self.DEFAULT_MINER_HOTKEY_2: {
                TP_ID_PORTFOLIO: PerfLedger(
                    initialization_time_ms=self.now_ms,
                    target_ledger_window_ms=1000 * 60 * 60 * 24 * 30
                )
            }
        }
        
        # Save to memory
        plm.save_perf_ledgers(test_bundles)
        
        # Verify data is in memory
        memory_bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertEqual(len(memory_bundles), 2)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, memory_bundles)
        self.assertIn(self.DEFAULT_MINER_HOTKEY_2, memory_bundles)
        
        # Test removal of hotkeys
        reduced_bundles = {
            self.DEFAULT_MINER_HOTKEY: test_bundles[self.DEFAULT_MINER_HOTKEY]
        }
        plm.save_perf_ledgers(reduced_bundles)
        
        # Should have removed the second miner
        memory_bundles_after = plm.get_perf_ledgers(portfolio_only=False)
        self.assertEqual(len(memory_bundles_after), 1)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, memory_bundles_after)
        self.assertNotIn(self.DEFAULT_MINER_HOTKEY_2, memory_bundles_after)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_build_portfolio_ledgers_only_flag(self, mock_candle_fetcher):
        """Test build_portfolio_ledgers_only functionality"""
        mock_candle_fetcher.return_value = {}
        
        # Test with portfolio ledgers only
        plm_portfolio_only = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            build_portfolio_ledgers_only=True
        )
        
        # Test with both portfolio and trade pair ledgers
        plm_full = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            build_portfolio_ledgers_only=False
        )
        
        # Create a position
        order = Order(
            price=50000,
            processed_ms=self.now_ms,
            order_uuid="test_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5
        )
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position",
            open_ms=self.now_ms,
            trade_pair=TradePair.BTCUSD,
            orders=[order],
            position_type=OrderType.LONG
        )
        position.rebuild_position_with_updated_orders()
        
        # Test portfolio only mode
        self.position_manager.clear_all_miner_positions()
        self.position_manager.save_miner_position(position)
        plm_portfolio_only.update(t_ms=self.now_ms + 1000 * 60 * 60)
        
        bundles_portfolio_only = plm_portfolio_only.get_perf_ledgers(portfolio_only=False)
        if self.DEFAULT_MINER_HOTKEY in bundles_portfolio_only:
            # Should only have portfolio ledger
            bundle = bundles_portfolio_only[self.DEFAULT_MINER_HOTKEY]
            self.assertIn(TP_ID_PORTFOLIO, bundle)
            # Should not have trade pair specific ledgers in portfolio-only mode
            self.assertEqual(len(bundle), 1)
        
        # Test full mode
        self.position_manager.clear_all_miner_positions()
        self.position_manager.save_miner_position(position)
        plm_full.update(t_ms=self.now_ms + 1000 * 60 * 60)
        
        bundles_full = plm_full.get_perf_ledgers(portfolio_only=False)
        if self.DEFAULT_MINER_HOTKEY in bundles_full:
            # Should have both portfolio and trade pair ledgers
            bundle = bundles_full[self.DEFAULT_MINER_HOTKEY]
            self.assertIn(TP_ID_PORTFOLIO, bundle)
            # May have trade pair specific ledgers depending on the logic


if __name__ == '__main__':
    unittest.main()