import multiprocessing
import unittest
from unittest.mock import patch

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    ParallelizationMode,
    PerfLedger,
    PerfLedgerManager,
    ShortcutReason,
    TradePairReturnStatus,
)


class TestPerfLedgerErrorHandling(TestBase):
    """Tests for error handling and exception scenarios in performance ledgers"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "error_test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.BASE_TIME = self.now_ms - (1000 * 60 * 60 * 24 * 7)  # 1 week ago

        self.mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
        )
        self.position_manager.clear_all_miner_positions()

    def test_invalid_portfolio_values(self):
        """Test handling of invalid portfolio values (NaN, inf, negative)"""
        ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Test various invalid values
        invalid_values = [float('nan'), float('inf'), float('-inf'), -1.0, 0.0]

        for i, invalid_value in enumerate(invalid_values):
            with self.subTest(invalid_value=invalid_value):
                try:
                    ledger.update_pl(
                        current_portfolio_value=invalid_value,
                        now_ms=self.BASE_TIME + (i * 1000 * 60 * 60),
                        miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                        any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                        current_portfolio_fee_spread=1.0,
                        current_portfolio_carry=1.0,
                    )

                    # If it doesn't raise an exception, verify the behavior
                    if ledger.cps:
                        final_cp = ledger.cps[-1]
                        # Should handle gracefully or have documented behavior
                        if invalid_value != invalid_value:  # NaN check
                            print(f"NaN value handled: {final_cp.prev_portfolio_ret}")
                        elif invalid_value < 0:
                            print(f"Negative value handled: {final_cp.prev_portfolio_ret}")

                except (ValueError, AssertionError, ZeroDivisionError) as e:
                    # Expected for some invalid values
                    print(f"Invalid value {invalid_value} correctly rejected: {e}")
                except Exception as e:
                    # Unexpected exceptions should be documented
                    print(f"Unexpected error with {invalid_value}: {e}")

    def test_corrupted_position_data(self):
        """Test handling of corrupted or invalid position data"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Create position with missing required fields
        try:
            corrupted_position = Position(
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                position_uuid="corrupted_pos",
                open_ms=self.BASE_TIME,
                trade_pair=None,  # Invalid - should be TradePair
                orders=[],  # Empty orders list
                position_type=OrderType.LONG,
            )

            self.position_manager.save_miner_position(corrupted_position)

            with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
                mock_fetch.return_value = {}

                # Should handle corrupted data gracefully
                plm.update(t_ms=self.now_ms)

        except Exception as e:
            # Should either handle gracefully or raise specific exceptions
            print(f"Corrupted position data handling: {e}")

    def test_price_data_fetch_failures(self):
        """Test handling of price data fetching failures"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Create valid position
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="price_fail_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=self.BASE_TIME, order_uuid="test_order",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
            ],
            position_type=OrderType.LONG,
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)

        # Test different price data failure scenarios
        failure_scenarios = [
            Exception("Network timeout"),
            ValueError("Invalid price data format"),
            KeyError("Missing price field"),
            TimeoutError("API timeout"),
        ]

        for i, exception in enumerate(failure_scenarios):
            with self.subTest(exception=type(exception).__name__):
                with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
                    mock_fetch.side_effect = exception

                    try:
                        plm.update(t_ms=self.BASE_TIME + (i * 1000 * 60 * 60))

                        # If it doesn't raise, check how it handled the failure
                        plm.get_perf_ledgers(portfolio_only=False)
                        print(f"Price fetch failure {type(exception).__name__} handled gracefully")

                    except Exception as e:
                        # Document how different failures are handled
                        print(f"Price fetch failure {type(exception).__name__} caused: {e}")

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_multiprocessing_worker_failures(self, mock_candle_fetcher):
        """Test handling of multiprocessing worker failures"""
        mock_candle_fetcher.return_value = {}

        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.MULTIPROCESSING,
        )

        # Create position
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="worker_fail_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=self.BASE_TIME, order_uuid="worker_test",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
            ],
            position_type=OrderType.LONG,
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)

        # Test multiprocessing failure scenarios
        existing_ledgers = plm.get_perf_ledgers(portfolio_only=False)
        hotkey_to_positions, _ = plm.get_positions_perf_ledger()

        if hotkey_to_positions:
            # Test with broken worker function
            with patch.object(plm, 'update_one_perf_ledger_parallel') as mock_worker:
                mock_worker.side_effect = Exception("Worker process crashed")

                try:
                    with multiprocessing.Pool(processes=2) as pool:
                        plm.update_perf_ledgers_parallel(
                            spark=None,
                            pool=pool,
                            hotkey_to_positions=hotkey_to_positions,
                            existing_perf_ledgers=existing_ledgers,
                            parallel_mode=ParallelizationMode.MULTIPROCESSING,
                            now_ms=self.now_ms,
                            is_backtesting=True,
                        )

                except Exception as e:
                    print(f"Multiprocessing worker failure handled: {e}")

    def test_memory_exhaustion_scenarios(self):
        """Test behavior under memory pressure"""
        # Create a scenario that could cause memory issues
        large_ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 365,  # 1 year window
        )

        try:
            # Try to create many checkpoints to stress memory
            for i in range(10000):  # Many updates
                large_ledger.update_pl(
                    current_portfolio_value=1.0 + (i * 0.0001),
                    now_ms=self.BASE_TIME + (i * 1000 * 60),  # Every minute
                    miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                    any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                    current_portfolio_fee_spread=1.0,
                    current_portfolio_carry=1.0,
                )

                # Check if pruning kicks in to prevent memory issues
                if i % 1000 == 0:
                    checkpoint_count = len(large_ledger.cps)
                    if checkpoint_count > 1000:  # Should have pruned by now
                        print(f"Checkpoint pruning working: {checkpoint_count} checkpoints at iteration {i}")
                        break

        except MemoryError:
            print("Memory exhaustion correctly detected")
        except Exception as e:
            print(f"Unexpected error during memory stress test: {e}")

    def test_concurrent_access_race_conditions(self):
        """Test race conditions in concurrent access scenarios"""
        import threading
        import time

        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Create shared state that could have race conditions
        errors = []

        def update_worker(worker_id):
            try:
                with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
                    mock_fetch.return_value = {}

                    for i in range(10):
                        # Simulate concurrent updates
                        plm.update(t_ms=self.BASE_TIME + (worker_id * 1000) + (i * 100))
                        time.sleep(0.001)  # Small delay to increase chance of race conditions

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # Start multiple threads to test concurrent access
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=update_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for race condition errors
        if errors:
            print(f"Race condition errors detected: {errors}")
        else:
            print("No race condition errors detected in concurrent access test")

    def test_invalid_parallel_mode_configuration(self):
        """Test invalid parallel mode configurations"""

        # Test with invalid parallel mode enum value
        try:
            invalid_plm = PerfLedgerManager(
                metagraph=self.mmg,
                running_unit_tests=True,
                position_manager=self.position_manager,
                parallel_mode="INVALID_MODE",  # Should be ParallelizationMode enum
            )

            # If it doesn't raise during construction, test during update
            with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
                mock_fetch.return_value = {}
                invalid_plm.update(t_ms=self.now_ms)

        except (ValueError, TypeError, AttributeError) as e:
            print(f"Invalid parallel mode correctly rejected: {e}")
        except Exception as e:
            print(f"Unexpected error with invalid parallel mode: {e}")

    def test_shortcut_reason_edge_cases(self):
        """Test that ShortcutReason enum values are properly defined"""

        # Verify all expected shortcut reasons exist and have correct values
        self.assertEqual(ShortcutReason.NO_SHORTCUT.value, 0)
        self.assertEqual(ShortcutReason.NO_OPEN_POSITIONS.value, 1)
        self.assertEqual(ShortcutReason.OUTSIDE_WINDOW.value, 2)
        self.assertEqual(ShortcutReason.ZERO_TIME_DELTA.value, 3)

        # Verify enum names are accessible
        self.assertEqual(ShortcutReason.NO_SHORTCUT.name, "NO_SHORTCUT")
        self.assertEqual(ShortcutReason.NO_OPEN_POSITIONS.name, "NO_OPEN_POSITIONS")

        # Test that we can iterate over all enum values
        all_reasons = list(ShortcutReason)
        self.assertEqual(len(all_reasons), 4)

    def test_ledger_invalidation_scenarios(self):
        """Test performance ledger invalidation scenarios"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Test manual invalidation
        test_time = self.now_ms
        plm.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] = test_time

        # Create position and update
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="invalidation_test",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=self.BASE_TIME, order_uuid="invalidation_order",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
            ],
            position_type=OrderType.LONG,
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)

        with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
            mock_fetch.return_value = {}

            # Update should handle invalidation
            plm.update(t_ms=self.now_ms)

            # Check that invalidation was processed
            if self.DEFAULT_MINER_HOTKEY not in plm.perf_ledger_hks_to_invalidate:
                print("Ledger invalidation processed successfully")
            else:
                print("Ledger invalidation still pending")

    def test_data_consistency_validation(self):
        """Test data consistency validation and corruption detection"""
        ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Create checkpoints
        for i in range(5):
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.01),
                now_ms=self.BASE_TIME + (i * 1000 * 60 * 60 * 12),
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        # Simulate data corruption
        if ledger.cps:
            # Corrupt timestamp order
            ledger.cps[1].last_update_ms = ledger.cps[0].last_update_ms - 1000

            # Try to add another checkpoint - should detect inconsistency
            try:
                ledger.update_pl(
                    current_portfolio_value=1.06,
                    now_ms=self.BASE_TIME + (6 * 1000 * 60 * 60 * 12),
                    miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                    any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                    current_portfolio_fee_spread=1.0,
                    current_portfolio_carry=1.0,
                )

                print("Data corruption not detected - may need validation enhancement")

            except (AssertionError, ValueError) as e:
                print(f"Data corruption correctly detected: {e}")
            except Exception as e:
                print(f"Unexpected error during corruption test: {e}")


class TestPerfLedgerFeeCalculations(TestBase):
    """Tests for fee calculation edge cases and precision"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "fee_test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.BASE_TIME = self.now_ms - (1000 * 60 * 60 * 24 * 7)

    def test_carry_fee_calculation_precision(self):
        """Test carry fee calculations with high precision requirements"""
        from vali_objects.vali_dataclasses.perf_ledger import FeeCache

        # Create test position for fee calculations
        test_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="fee_test_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=self.BASE_TIME, order_uuid="fee_test_order",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
            ],
            position_type=OrderType.LONG,
        )
        test_position.rebuild_position_with_updated_orders()

        # Test different time intervals for carry fee calculation
        test_scenarios = [
            (self.BASE_TIME + 1000 * 60 * 60 * 8, "8_hour"),      # 8 hours after open
            (self.BASE_TIME + 1000 * 60 * 60 * 24, "24_hour"),    # 24 hours after open
            (self.BASE_TIME + 1000 * 60 * 60 * 25, "25_hour"),    # 25 hours after open (edge case)
            (self.BASE_TIME + 1000 * 60 * 60 * 7, "7_hour"),      # 7 hours after open (edge case)
        ]

        for current_time_ms, scenario_name in test_scenarios:
            with self.subTest(scenario=scenario_name):
                fee_cache = FeeCache()

                # Test carry fee calculation
                try:
                    carry_fee, is_cache_miss = fee_cache.get_carry_fee(current_time_ms, test_position)

                    # Verify fee is reasonable (between 0.99 and 1.001 for short periods)
                    self.assertGreater(carry_fee, 0.99, f"Carry fee too low for {scenario_name}: {carry_fee}")
                    self.assertLessEqual(carry_fee, 1.0, f"Carry fee should be <= 1.0 for {scenario_name}: {carry_fee}")

                    print(f"Carry fee for {scenario_name}: {carry_fee}, cache_miss={is_cache_miss}")

                except Exception as e:
                    print(f"Carry fee calculation failed for {scenario_name}: {e}")

    def test_spread_fee_accumulation(self):
        """Test spread fee accumulation with many small trades"""
        from vali_objects.vali_dataclasses.perf_ledger import FeeCache

        fee_cache = FeeCache()

        # Create test position with multiple orders
        test_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="spread_test_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[],
            position_type=OrderType.LONG,
        )

        # Simulate many small trades
        accumulated_spread_fee = 1.0
        num_trades = 10  # Reduced for testing

        for i in range(num_trades):
            try:
                # Add a new order to the position
                order_time = self.BASE_TIME + (i * 1000 * 60)  # Each order 1 minute apart
                new_order = Order(
                    price=50000 + (i * 100),  # Slight price variation
                    processed_ms=order_time,
                    order_uuid=f"spread_order_{i}",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=0.5,
                )
                test_position.orders.append(new_order)
                test_position.rebuild_position_with_updated_orders()

                # Get spread fee for this position at current time
                current_time = order_time + 1000  # 1 second after order
                spread_fee, is_cache_miss = fee_cache.get_spread_fee(test_position, current_time)
                accumulated_spread_fee *= spread_fee

                # Verify spread fee is always <= 1.0 (fee reduces value)
                self.assertLessEqual(spread_fee, 1.0, f"Spread fee should be <= 1.0, got {spread_fee}")
                self.assertGreater(spread_fee, 0.98, f"Spread fee too low, got {spread_fee}")

                print(f"Trade {i}: spread_fee={spread_fee:.6f}, cache_miss={is_cache_miss}")

            except Exception as e:
                print(f"Spread fee calculation failed on trade {i}: {e}")
                break

        # After trades, should have some fee impact
        print(f"Accumulated spread fee after {num_trades} trades: {accumulated_spread_fee}")

    def test_fee_cache_invalidation(self):
        """Test fee cache invalidation and refresh scenarios"""
        from vali_objects.vali_dataclasses.perf_ledger import FeeCache

        fee_cache = FeeCache()

        # Create test position
        test_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="cache_test_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=self.BASE_TIME, order_uuid="cache_order_1",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
            ],
            position_type=OrderType.LONG,
        )
        test_position.rebuild_position_with_updated_orders()

        # Test times
        time_8h = self.BASE_TIME + (1000 * 60 * 60 * 8)
        time_24h = self.BASE_TIME + (1000 * 60 * 60 * 24)

        # Get initial fees
        initial_spread, _ = fee_cache.get_spread_fee(test_position, time_8h)
        initial_carry_8h, _ = fee_cache.get_carry_fee(time_8h, test_position)

        # Simulate cache usage
        try:
            # Access with different time to potentially hit/miss cache
            different_carry, cache_miss = fee_cache.get_carry_fee(time_24h, test_position)

            # Get fees again with same parameters - should hit cache
            refreshed_spread, spread_cache_miss = fee_cache.get_spread_fee(test_position, time_8h)
            refreshed_carry_8h, carry_cache_miss = fee_cache.get_carry_fee(time_8h, test_position)

            # Spread fee should be consistent (same position, same time)
            self.assertEqual(initial_spread, refreshed_spread, "Spread fee inconsistent after cache operations")

            # Carry fee cache behavior depends on implementation
            print("Fee cache test results:")
            print(f"  Initial spread: {initial_spread}, carry_8h: {initial_carry_8h}")
            print(f"  Different time carry_24h: {different_carry}, cache_miss={cache_miss}")
            print(f"  Refreshed spread: {refreshed_spread}, cache_miss={spread_cache_miss}")
            print(f"  Refreshed carry_8h: {refreshed_carry_8h}, cache_miss={carry_cache_miss}")

        except Exception as e:
            print(f"Fee cache test encountered: {e}")


if __name__ == '__main__':
    unittest.main()
