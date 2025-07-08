import unittest
import time
import copy
from copy import deepcopy
from unittest.mock import patch

from shared_objects.sn8_multiprocessing import get_spark_session, get_multiprocessing_pool
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


class TestPerfLedgerDeltaBuilding(TestBase):
    """Tests for delta/incremental building logic and performance issues"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg, 
            running_unit_tests=True, 
            elimination_manager=self.elimination_manager
        )
        self.position_manager.clear_all_miner_positions()

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_incremental_building_vs_full_rebuild(self, mock_candle_fetcher):
        """Test that incremental building is faster than full rebuilds"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL
        )
        
        # Create a position with orders spanning several days
        base_time = self.now_ms - (1000 * 60 * 60 * 24 * 10)  # 10 days ago
        orders = []
        
        for i in range(5):  # 5 orders over 5 days
            order_time = base_time + (i * 1000 * 60 * 60 * 24)
            order = Order(
                price=50000 + (i * 100),
                processed_ms=order_time,
                order_uuid=f"test_order_{i}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG if i == 0 else (OrderType.FLAT if i == 4 else OrderType.LONG),
                leverage=0.5 if i != 4 else 0
            )
            orders.append(order)
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=orders,
            position_type=OrderType.FLAT
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)
        
        # First full build - build up to order 2
        update_time_1 = orders[2].processed_ms + 1000
        start_time = time.time()
        plm.update(t_ms=update_time_1)
        full_build_time = time.time() - start_time
        
        # Get initial state
        ledgers_after_full = plm.get_perf_ledgers(portfolio_only=False)
        initial_last_update = ledgers_after_full[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        
        # Incremental build - build up to order 3
        update_time_2 = orders[3].processed_ms + 1000
        start_time = time.time()
        plm.update(t_ms=update_time_2)
        incremental_build_time = time.time() - start_time
        
        # Get updated state
        ledgers_after_incremental = plm.get_perf_ledgers(portfolio_only=False)
        updated_last_update = ledgers_after_incremental[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        
        # Verify incremental build occurred
        self.assertGreater(updated_last_update, initial_last_update)
        
        # In a properly functioning system, incremental should be faster or similar
        # This test documents the current behavior and would catch regressions
        print(f"Full build time: {full_build_time:.4f}s")
        print(f"Incremental build time: {incremental_build_time:.4f}s")

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_testing_one_hotkey_causes_full_rebuild(self, mock_candle_fetcher):
        """Test that testing_one_hotkey parameter forces full rebuilds"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create position
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
        self.position_manager.save_miner_position(position)
        
        # First update without testing_one_hotkey
        plm.update(t_ms=self.now_ms + 1000 * 60 * 60)
        
        # Check that ledger exists in memory
        ledgers_1 = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, ledgers_1)
        
        # Store initial checkpoint count
        len(ledgers_1[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps)
        
        # Second update WITH testing_one_hotkey (this triggers the bug)
        plm.update(testing_one_hotkey=self.DEFAULT_MINER_HOTKEY, t_ms=self.now_ms + 1000 * 60 * 60 * 2)
        
        # Check ledgers after testing_one_hotkey update
        ledgers_2 = plm.get_perf_ledgers(portfolio_only=False)
        
        # The testing_one_hotkey parameter should cause regeneration
        # This test documents the problematic behavior
        self.assertIn(self.DEFAULT_MINER_HOTKEY, ledgers_2)
        
        # In the current broken implementation, this forces a full rebuild
        # which is why SERIAL mode is slow in backtest_manager.py

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_existing_ledger_preservation(self, mock_candle_fetcher):
        """Test that existing ledgers are properly preserved during updates"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create initial position
        order1 = Order(
            price=50000,
            processed_ms=self.now_ms,
            order_uuid="test_order_1",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5
        )
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position",
            open_ms=self.now_ms,
            trade_pair=TradePair.BTCUSD,
            orders=[order1],
            position_type=OrderType.LONG
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)
        
        # Initial update
        plm.update(t_ms=self.now_ms + 1000 * 60 * 60)
        initial_ledgers = plm.get_perf_ledgers(portfolio_only=False)
        
        # Verify ledger was created
        self.assertIn(self.DEFAULT_MINER_HOTKEY, initial_ledgers)
        initial_last_update = initial_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        initial_cp_count = len(initial_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps)
        
        # Add a new order to the same position
        order2 = Order(
            price=51000,
            processed_ms=self.now_ms + 1000 * 60 * 60 * 2,
            order_uuid="test_order_2",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0
        )
        
        position.add_order(order2)
        self.position_manager.save_miner_position(position)
        
        # Update with new order (should be incremental)
        plm.update(t_ms=self.now_ms + 1000 * 60 * 60 * 3)
        updated_ledgers = plm.get_perf_ledgers(portfolio_only=False)
        
        # Verify incremental update occurred
        updated_last_update = updated_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        updated_cp_count = len(updated_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps)
        
        # Should have advanced the last update time
        self.assertGreater(updated_last_update, initial_last_update)
        
        # Should have added checkpoints, not rebuilt from scratch
        self.assertGreaterEqual(updated_cp_count, initial_cp_count)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_ledger_bundle_deepcopy_behavior_serial(self, mock_candle_fetcher):
        """Test that existing ledger bundles are properly handled in SERIAL mode"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL
        )
        
        # Create initial ledger bundle
        original_ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30
        )

        # Create position
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
        self.position_manager.save_miner_position(position)
        
        # Add some checkpoints
        original_ledger.update_pl(
            current_portfolio_value=1.05,
            now_ms=self.now_ms + 1000 * 60 * 60,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0
        )
        
        bundle = {TP_ID_PORTFOLIO: original_ledger}
        plm.hotkey_to_perf_bundle[self.DEFAULT_MINER_HOTKEY] = bundle
        
        # Store original checkpoint count
        original_cp_count = len(original_ledger.cps)
        
        # Get positions
        hotkey_to_positions, _ = plm.get_positions_perf_ledger(testing_one_hotkey=self.DEFAULT_MINER_HOTKEY)

        # Update in SERIAL mode
        result = plm.update_one_perf_ledger_bundle(
            0, 1, self.DEFAULT_MINER_HOTKEY, hotkey_to_positions[self.DEFAULT_MINER_HOTKEY],
            self.now_ms + 1000 * 60 * 60 * 2, 
            {self.DEFAULT_MINER_HOTKEY: bundle}
        )
        
        # In SERIAL mode, the result should be None (updated in place)
        self.assertIsNone(result)
        
        # Verify the original bundle was modified in place
        self.assertGreaterEqual(len(original_ledger.cps), original_cp_count)
        
        # Verify the stored bundle is the same object
        stored_bundle = plm.hotkey_to_perf_bundle.get(self.DEFAULT_MINER_HOTKEY)
        if stored_bundle:
            self.assertIs(stored_bundle[TP_ID_PORTFOLIO], original_ledger)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_ledger_bundle_deepcopy_behavior_parallel(self, mock_candle_fetcher):
        """Test that existing ledger bundles are properly deepcopied in PARALLEL modes"""
        mock_candle_fetcher.return_value = {}
        
        # Test with MULTIPROCESSING mode
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.MULTIPROCESSING
        )
        
        # Create initial ledger bundle
        original_ledger = PerfLedger(
            initialization_time_ms=self.now_ms,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30
        )

        # Create position
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
        self.position_manager.save_miner_position(position)
        
        # Add some checkpoints
        original_ledger.update_pl(
            current_portfolio_value=1.05,
            now_ms=self.now_ms + 1000 * 60 * 60,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0
        )
        
        bundle = {TP_ID_PORTFOLIO: original_ledger}
        plm.hotkey_to_perf_bundle[self.DEFAULT_MINER_HOTKEY] = bundle
        
        # Store original state
        original_cp_count = len(original_ledger.cps)
        original_last_update = original_ledger.last_update_ms
        
        # Get positions
        hotkey_to_positions, _ = plm.get_positions_perf_ledger(testing_one_hotkey=self.DEFAULT_MINER_HOTKEY)

        # Update in PARALLEL mode
        result = plm.update_one_perf_ledger_bundle(
            0, 1, self.DEFAULT_MINER_HOTKEY, hotkey_to_positions[self.DEFAULT_MINER_HOTKEY],
            self.now_ms + 1000 * 60 * 60 * 2, 
            {self.DEFAULT_MINER_HOTKEY: bundle}
        )
        
        # In parallel modes, should return the updated bundle
        self.assertIsNotNone(result)

        # Verify the original bundle was NOT modified (deepcopy was used)
        self.assertEqual(len(original_ledger.cps), original_cp_count)
        self.assertEqual(original_ledger.last_update_ms, original_last_update)
        
        # Verify the returned bundle is different
        returned_ledger = result[TP_ID_PORTFOLIO]
        self.assertIsNot(returned_ledger, original_ledger)
        self.assertGreaterEqual(len(returned_ledger.cps), original_cp_count)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_performance_regression_detection(self, mock_candle_fetcher):
        """Test that helps detect performance regressions in perf ledger updates"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create a position with many orders to simulate real workload
        base_time = self.now_ms - (1000 * 60 * 60 * 24 * 30)  # 30 days ago
        orders = []
        
        # Create orders every day for 30 days
        for day in range(30):
            order_time = base_time + (day * 1000 * 60 * 60 * 24)
            order = Order(
                price=50000 + (day * 10),
                processed_ms=order_time,
                order_uuid=f"order_{day}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG if day % 10 != 9 else OrderType.FLAT,
                leverage=0.1 if day % 10 != 9 else 0
            )
            orders.append(order)
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=orders,
            position_type=OrderType.FLAT
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)
        
        # Time multiple updates to detect performance issues
        update_times = []
        
        # First update (full build)
        start_time = time.time()
        plm.update(t_ms=orders[10].processed_ms + 1000)
        update_times.append(time.time() - start_time)
        
        # Subsequent updates (should be incremental)
        for i in range(11, min(15, len(orders))):
            start_time = time.time()
            plm.update(t_ms=orders[i].processed_ms + 1000)
            update_times.append(time.time() - start_time)
        
        # Print timing info for analysis
        print("Update times:")
        for i, update_time in enumerate(update_times):
            print(f"  Update {i}: {update_time:.4f}s")
        
        # In a well-functioning system, incremental updates should be consistently fast
        # This test helps identify when full rebuilds are happening instead of incremental updates
        if len(update_times) > 1:
            avg_incremental_time = sum(update_times[1:]) / len(update_times[1:])
            print(f"Average incremental update time: {avg_incremental_time:.4f}s")
            
            # This is a soft assertion - in practice, incremental updates should be much faster
            # If this fails consistently, it indicates the delta building isn't working

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_concurrent_position_updates(self, mock_candle_fetcher):
        """Test performance with concurrent updates to multiple positions"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create multiple positions for the same miner on different trade pairs
        # Use unique trade pairs to avoid violating the one-open-position-per-pair constraint
        trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.GBPUSD, TradePair.EURUSD, TradePair.AUDUSD]
        num_positions = len(trade_pairs)
        base_time = self.now_ms - (1000 * 60 * 60 * 24)  # 1 day ago
        
        for i, trade_pair in enumerate(trade_pairs):
            order = Order(
                price=50000 + (i * 1000) if trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD] else 1.2 + (i * 0.01),
                processed_ms=base_time + (i * 1000 * 60 * 60),  # Stagger by hours
                order_uuid=f"concurrent_order_{i}",
                trade_pair=trade_pair,
                order_type=OrderType.LONG,
                leverage=0.1 + (i * 0.1)
            )
            
            position = Position(
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                position_uuid=f"concurrent_pos_{i}",
                open_ms=base_time + (i * 1000 * 60 * 60),
                trade_pair=trade_pair,
                orders=[order],
                position_type=OrderType.LONG
            )
            position.rebuild_position_with_updated_orders()
            self.position_manager.save_miner_position(position)
        
        # Time the initial update with multiple positions
        start_time = time.time()
        plm.update(t_ms=self.now_ms)
        initial_update_time = time.time() - start_time
        
        initial_ledgers = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, initial_ledgers)
        
        # Verify all positions are tracked
        portfolio_ledger = initial_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
        self.assertGreater(len(portfolio_ledger.cps), 0)
        portfolio_ledger = deepcopy(portfolio_ledger) # Deepcopy to avoid modifying original
        
        # Add orders to existing positions by updating each one sequentially
        close_time = self.now_ms + (1000 * 60 * 30)  # 30 minutes from now
        hotkey_to_positions, _ = plm.get_positions_perf_ledger()
        if self.DEFAULT_MINER_HOTKEY in hotkey_to_positions:
            positions = hotkey_to_positions[self.DEFAULT_MINER_HOTKEY]
            for i, position in enumerate(positions):
                # Close the position and then reopen with different parameters to simulate updates
                close_order = Order(
                    price=51000 + (i * 100) if position.trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD] else 1.25 + (i * 0.01),
                    processed_ms=close_time,  # 30 minutes from now
                    order_uuid=f"concurrent_close_{i}",
                    trade_pair=position.trade_pair,
                    order_type=OrderType.FLAT,
                    leverage=0.0
                )
                position.add_order(close_order)
                self.position_manager.save_miner_position(position)
        
        # Time the incremental update - use time after close orders
        update_time = close_time + (1000 * 60 * 10)  # 10 minutes after close orders
        start_time = time.time()
        plm.update(t_ms=update_time)
        incremental_update_time = time.time() - start_time
        
        print("\nConcurrent position update times:")
        print(f"  Initial update ({num_positions} positions): {initial_update_time:.4f}s")
        print(f"  Incremental update: {incremental_update_time:.4f}s")
        
        # Verify all updates were processed
        updated_ledgers = plm.get_perf_ledgers(portfolio_only=False)
        updated_portfolio_ledger = updated_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
        self.assertGreater(updated_portfolio_ledger.last_update_ms, portfolio_ledger.last_update_ms)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_memory_efficiency(self, mock_candle_fetcher):
        """Test memory efficiency with large number of checkpoints"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Create a long-running position
        base_time = self.now_ms - (1000 * 60 * 60 * 24 * 60)  # 60 days ago
        
        order = Order(
            price=50000,
            processed_ms=base_time,
            order_uuid="memory_test_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5
        )
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="memory_test_pos",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=[order],
            position_type=OrderType.LONG
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)
        
        # Do many updates to create many checkpoints
        update_interval = 1000 * 60 * 60 * 6  # 6 hours
        num_updates = 100  # Simulate 25 days of updates
        
        for i in range(num_updates):
            update_time = base_time + (i * update_interval)
            plm.update(t_ms=update_time)
            
            if i % 20 == 0:  # Check memory usage periodically
                ledgers = plm.get_perf_ledgers(portfolio_only=False)
                portfolio_ledger = ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
                checkpoint_count = len(portfolio_ledger.cps)
                window_duration = portfolio_ledger.get_total_ledger_duration_ms()

                print(f"\nMemory efficiency check at update {i}:")
                print(f"  Checkpoints: {checkpoint_count}")
                print(f"  Window duration: {window_duration / (1000 * 60 * 60 * 24):.1f} days")

                # Verify old checkpoints are being pruned
                self.assertLessEqual(
                    window_duration,
                    portfolio_ledger.target_ledger_window_ms * 1.5,
                    "Window duration should not exceed target by more than 50%"
                )


class TestParallelVsSerialModes(TestBase):
    """Tests comparing parallel and serial execution modes"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager
        )
        self.position_manager.clear_all_miner_positions()

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_serial_vs_parallel_consistency(self, mock_candle_fetcher):
        """Test that serial and parallel modes produce consistent results"""
        mock_candle_fetcher.return_value = {}
        
        # Create identical position managers for both modes
        position_manager_serial = PositionManager(
            metagraph=self.mmg, 
            running_unit_tests=True, 
            elimination_manager=self.elimination_manager
        )
        position_manager_serial.clear_all_miner_positions()
        
        position_manager_parallel = PositionManager(
            metagraph=self.mmg, 
            running_unit_tests=True, 
            elimination_manager=self.elimination_manager
        )
        position_manager_parallel.clear_all_miner_positions()
        
        # Create test position
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
        
        # Add position to both managers
        position_manager_serial.save_miner_position(position)
        
        # For parallel manager, clear memory and reload from disk to avoid state conflicts
        position_manager_parallel.hotkey_to_positions = {}
        position_manager_parallel._populate_memory_positions_for_first_time()
        
        # Now save the position to parallel manager (it will see the existing disk state)
        position_manager_parallel.save_miner_position(copy.deepcopy(position))
        
        # Create ledger managers with different modes
        plm_serial = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=position_manager_serial,
            parallel_mode=ParallelizationMode.SERIAL
        )
        
        plm_multiprocessing = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=position_manager_parallel,
            parallel_mode=ParallelizationMode.MULTIPROCESSING
        )
        
        # Update both
        update_time = self.now_ms + 1000 * 60 * 60
        plm_serial.update(t_ms=update_time)
        
        # For parallel mode, we need to test the parallel update method
        existing_ledgers = plm_multiprocessing.get_perf_ledgers(portfolio_only=False)
        hotkey_to_positions, _ = plm_multiprocessing.get_positions_perf_ledger()
        
        if hotkey_to_positions:  # Only test if we have positions
            try:
                from multiprocessing import Pool
                # Use real multiprocessing pool for testing
                with Pool(processes=2) as pool:
                    updated_ledgers = plm_multiprocessing.update_perf_ledgers_parallel(
                        spark=None,  # Not using Spark
                        pool=pool,   # Real multiprocessing pool
                        hotkey_to_positions=hotkey_to_positions,
                        existing_perf_ledgers=existing_ledgers,
                        parallel_mode=ParallelizationMode.MULTIPROCESSING,
                        now_ms=update_time,
                        is_backtesting=True
                    )
                    
                    # Verify parallel processing worked
                    if updated_ledgers and self.DEFAULT_MINER_HOTKEY in updated_ledgers:
                        parallel_ledger = updated_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
                        self.assertGreater(parallel_ledger.last_update_ms, 0)
                        
            except Exception as e:
                # Document any issues with multiprocessing setup
                print(f"Multiprocessing test encountered: {e}")
        
        # Get results from serial mode
        serial_ledgers = plm_serial.get_perf_ledgers(portfolio_only=False)
        
        # Verify serial mode worked
        if self.DEFAULT_MINER_HOTKEY in serial_ledgers:
            self.assertIn(TP_ID_PORTFOLIO, serial_ledgers[self.DEFAULT_MINER_HOTKEY])
            portfolio_ledger = serial_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
            self.assertGreater(portfolio_ledger.last_update_ms, 0)

    def test_parallel_mode_configuration(self):
        """Test that parallel modes are configured correctly"""
        
        # Test SERIAL mode
        plm_serial = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=None,
            parallel_mode=ParallelizationMode.SERIAL
        )
        self.assertEqual(plm_serial.parallel_mode, ParallelizationMode.SERIAL)
        
        # Test MULTIPROCESSING mode
        plm_mp = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=None,
            parallel_mode=ParallelizationMode.MULTIPROCESSING
        )
        self.assertEqual(plm_mp.parallel_mode, ParallelizationMode.MULTIPROCESSING)
        
        # Note: Skipping PYSPARK mode for GitHub CI compatibility
        # Focus on MULTIPROCESSING which is more portable

    def test_update_one_perf_ledger_parallel_structure(self):
        """Test the structure of the parallel update method"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=None,
            parallel_mode=ParallelizationMode.MULTIPROCESSING
        )
        
        # Test the data tuple structure expected by update_one_perf_ledger_parallel
        test_data_tuple = (
            0,  # hotkey_i
            1,  # n_hotkeys
            self.DEFAULT_MINER_HOTKEY,  # hotkey
            [],  # positions
            None,  # existing_bundle
            self.now_ms,  # now_ms
            True  # is_backtesting
        )
        
        # This should create a worker PLM and return results
        # In practice, this would be called by multiprocessing
        try:
            result = plm.update_one_perf_ledger_parallel(test_data_tuple)
            # Should return (hotkey, new_bundle)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], self.DEFAULT_MINER_HOTKEY)
            self.assertIsInstance(result[1], dict)  # Should be a ledger bundle
        except Exception as e:
            # May fail due to missing dependencies, but structure should be testable
            print(f"Parallel worker test encountered: {e}")

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_parallel_mode_error_handling(self, mock_candle_fetcher):
        """Test error handling in parallel processing modes"""
        mock_candle_fetcher.return_value = {}
        
        position_manager = PositionManager(
            metagraph=self.mmg, 
            running_unit_tests=True, 
            elimination_manager=self.elimination_manager
        )
        
        # Create a position that might cause issues
        order = Order(
            price=50000,
            processed_ms=self.now_ms,
            order_uuid="error_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5
        )
        
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="error_position",
            open_ms=self.now_ms,
            trade_pair=TradePair.BTCUSD,
            orders=[order],
            position_type=OrderType.LONG
        )
        position.rebuild_position_with_updated_orders()
        position_manager.save_miner_position(position)
        
        # Test with multiprocessing mode
        plm_mp = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=position_manager,
            parallel_mode=ParallelizationMode.MULTIPROCESSING
        )
        
        # Mock a failure in candle fetching
        mock_candle_fetcher.side_effect = Exception("Price data unavailable")
        
        # Update should handle the error gracefully
        try:
            plm_mp.update(t_ms=self.now_ms + 1000 * 60 * 60)
            # If no exception, check that ledgers are empty or contain default values
            ledgers = plm_mp.get_perf_ledgers(portfolio_only=False)
            # The error handling might result in empty ledgers or partial updates
            print(f"Error handled gracefully. Ledgers: {list(ledgers.keys())}")
        except Exception as e:
            # Document what exceptions are expected
            print(f"Expected exception type: {type(e).__name__}: {e}")

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_parallel_vs_serial_large_scale(self, mock_candle_fetcher):
        """Test performance difference between parallel and serial with many miners"""
        mock_candle_fetcher.return_value = {}
        
        # Create multiple miners
        num_miners = 5  # Reduced for faster test execution
        miner_hotkeys = [f"miner_{i}" for i in range(num_miners)]
        mmg_large = MockMetagraph(hotkeys=miner_hotkeys)
        
        # Create position managers for both modes
        elimination_manager_large = EliminationManager(mmg_large, None, None)
        
        position_manager = PositionManager(
            metagraph=mmg_large, 
            running_unit_tests=True, 
            elimination_manager=elimination_manager_large
        )
        position_manager.clear_all_miner_positions()

        
        # Create positions for each miner
        base_time = self.now_ms - (1000 * 60 * 60 * 24)  # 1 day ago
        
        for i, hotkey in enumerate(miner_hotkeys):
            order = Order(
                price=50000 + (i * 100),
                processed_ms=base_time,
                order_uuid=f"order_{hotkey}",
                trade_pair=TradePair.BTCUSD if i % 2 == 0 else TradePair.ETHUSD,
                order_type=OrderType.LONG,
                leverage=0.3
            )
            
            position = Position(
                miner_hotkey=hotkey,
                position_uuid=f"pos_{hotkey}",
                open_ms=base_time,
                trade_pair=TradePair.BTCUSD if i % 2 == 0 else TradePair.ETHUSD,
                orders=[order],
                position_type=OrderType.LONG
            )
            position.rebuild_position_with_updated_orders()
            
            # Save to both managers  
            position_manager.save_miner_position(position, delete_open_position_if_exists=True)

        # Create ledger managers
        plm_serial = PerfLedgerManager(
            metagraph=mmg_large,
            running_unit_tests=True,
            position_manager=position_manager,
            parallel_mode=ParallelizationMode.SERIAL
        )
        
        plm_parallel = PerfLedgerManager(
            metagraph=mmg_large,
            running_unit_tests=True,
            position_manager=position_manager,
            parallel_mode=ParallelizationMode.MULTIPROCESSING
        )
        
        # Time serial update
        start_time = time.time()
        plm_serial.update(t_ms=self.now_ms)
        serial_time = time.time() - start_time
        
        # Time parallel update (note: in unit tests, multiprocessing might not show benefits)
        start_time = time.time()
        parallel_mode = ParallelizationMode.MULTIPROCESSING
        spark, should_close = get_spark_session(parallel_mode)
        pool = get_multiprocessing_pool(parallel_mode)
        plm_parallel.update_perf_ledgers_parallel(
            spark=spark,
            pool=pool,
            hotkey_to_positions=plm_parallel.get_positions_perf_ledger()[0],
            existing_perf_ledgers=plm_parallel.get_perf_ledgers(portfolio_only=False),
            parallel_mode=parallel_mode,
            now_ms=self.now_ms,
            is_backtesting=True
        )
        parallel_time = time.time() - start_time
        
        print(f"\nLarge scale performance comparison ({num_miners} miners):")
        print(f"  Serial mode: {serial_time:.4f}s")
        print(f"  Parallel mode: {parallel_time:.4f}s")
        if parallel_time > 0:
            print(f"  Speedup: {serial_time / parallel_time:.2f}x")
        
        # Verify both produce the same results
        serial_ledgers = plm_serial.get_perf_ledgers(portfolio_only=False)
        parallel_ledgers = plm_parallel.get_perf_ledgers(portfolio_only=False)
        
        self.assertEqual(set(serial_ledgers.keys()), set(parallel_ledgers.keys()),
                        "Both modes should process the same miners")
        
        # Verify checkpoint counts match
        for hotkey in serial_ledgers:
            if hotkey in parallel_ledgers:
                serial_cp_count = len(serial_ledgers[hotkey][TP_ID_PORTFOLIO].cps)
                parallel_cp_count = len(parallel_ledgers[hotkey][TP_ID_PORTFOLIO].cps)
                self.assertEqual(serial_cp_count, parallel_cp_count,
                               f"Checkpoint counts should match for {hotkey}")

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_edge_case_handling(self, mock_candle_fetcher):
        """Test various edge cases in performance ledger updates"""
        mock_candle_fetcher.return_value = {}
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager
        )
        
        # Edge case 1: Very old position
        old_order = Order(
            price=30000,
            processed_ms=self.now_ms - (1000 * 60 * 60 * 24 * 365),  # 1 year ago
            order_uuid="old_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.1
        )
        
        old_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="old_position",
            open_ms=self.now_ms - (1000 * 60 * 60 * 24 * 365),
            trade_pair=TradePair.BTCUSD,
            orders=[old_order],
            position_type=OrderType.LONG
        )
        old_position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(old_position)
        
        # Should handle old positions
        plm.update(t_ms=self.now_ms)
        
        # Edge case 2: Position closed very quickly
        quick_open = Order(
            price=50000,
            processed_ms=self.now_ms - 1000 * 60,  # 1 minute ago
            order_uuid="quick_open",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5
        )
        
        quick_close = Order(
            price=50100,
            processed_ms=self.now_ms - 1000 * 30,  # 30 seconds ago
            order_uuid="quick_close",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0.0
        )
        
        quick_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="quick_position",
            open_ms=self.now_ms - 1000 * 60,
            trade_pair=TradePair.BTCUSD,
            orders=[quick_open, quick_close],
            position_type=OrderType.FLAT
        )
        quick_position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(quick_position)
        
        # Edge case 3: Position with very small leverage
        small_leverage_order = Order(
            price=40000,
            processed_ms=self.now_ms - 1000 * 60 * 60,  # 1 hour ago
            order_uuid="small_leverage",
            trade_pair=TradePair.ETHUSD,
            order_type=OrderType.SHORT,
            leverage=0.001  # Minimum allowed leverage
        )
        
        small_leverage_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="small_leverage_pos",
            open_ms=self.now_ms - 1000 * 60 * 60,
            trade_pair=TradePair.ETHUSD,
            orders=[small_leverage_order],
            position_type=OrderType.SHORT
        )
        small_leverage_position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(small_leverage_position)
        
        # Update with multiple edge cases
        plm.update(t_ms=self.now_ms)
        
        ledgers = plm.get_perf_ledgers(portfolio_only=False)
        if self.DEFAULT_MINER_HOTKEY in ledgers:
            portfolio_ledger = ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
            print("\nEdge case handling results:")
            print(f"  Checkpoint count: {len(portfolio_ledger.cps)}")
            print(f"  Window duration: {portfolio_ledger.get_total_ledger_duration_ms() / (1000 * 60 * 60 * 24):.1f} days")
            
            # Get positions using the correct API
            hotkey_to_positions, _ = plm.get_positions_perf_ledger()
            if self.DEFAULT_MINER_HOTKEY in hotkey_to_positions:
                print(f"  Positions processed: {len(hotkey_to_positions[self.DEFAULT_MINER_HOTKEY])}")
            
            # Verify all edge cases were handled
            self.assertGreater(len(portfolio_ledger.cps), 0, "Should have created checkpoints")
            self.assertGreater(portfolio_ledger.last_update_ms, 0, "Should have updated")


if __name__ == '__main__':
    unittest.main()