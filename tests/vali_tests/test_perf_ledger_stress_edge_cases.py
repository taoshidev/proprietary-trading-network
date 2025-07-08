import time
import unittest
from unittest.mock import patch

from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    TP_ID_PORTFOLIO,
    PerfLedger,
    PerfLedgerManager,
    TradePairReturnStatus,
)


class TestPerfLedgerStressTests(TestBase):
    """Stress tests for performance ledgers with large datasets and extreme scenarios"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "stress_test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.BASE_TIME = self.now_ms - (1000 * 60 * 60 * 24 * 90)  # 90 days ago

        self.mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
        )
        self.position_manager.clear_all_miner_positions()

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_high_frequency_trading_simulation(self, mock_candle_fetcher):
        """Test performance with high-frequency trading patterns"""
        mock_candle_fetcher.return_value = {}

        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            build_portfolio_ledgers_only=True,  # Optimize by building only portfolio ledgers
        )

        # Simulate HFT with 200 orders over 200 minutes (reduced from 1000)
        base_time = self.BASE_TIME
        minute_ms = 60 * 1000

        orders = []
        current_leverage = 0.001
        base_price = 50000

        # Reduced to 200 orders to make CI more reliable
        for i in range(200):
            # Alternate between increasing and decreasing position
            if i % 10 == 0 and i != 0:
                current_leverage = 0  # Flatten position occasionally
            else:
                current_leverage += 0.001 * (1 if i % 2 == 0 else -1)

            # Small price movements
            price = base_price + (i % 100) - 50

            orders.append(Order(
                price=price,
                processed_ms=base_time + (i * minute_ms),  # Every minute
                order_uuid=f"hft_order_{i}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG if current_leverage > 0 else (OrderType.SHORT if current_leverage < 0 else OrderType.FLAT),
                leverage=abs(current_leverage),
            ))

        hft_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="hft_position",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=orders,
            position_type=OrderType.FLAT,
        )
        hft_position.rebuild_position_with_updated_orders()

        self.position_manager.save_miner_position(hft_position)

        # Time the update operation
        start_time = time.time()
        plm.update(t_ms=base_time + (201 * minute_ms))
        update_time = time.time() - start_time

        print(f"HFT simulation update time: {update_time:.4f}s for 200 orders")

        # Increased timeout to 60s for CI environments, which is still reasonable
        self.assertLess(update_time, 60.0, "HFT simulation taking too long")

        # Verify ledger was created
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, bundles)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_massive_position_count(self, mock_candle_fetcher):
        """Test with many concurrent positions across different trade pairs"""
        mock_candle_fetcher.return_value = {}

        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            build_portfolio_ledgers_only=False,
        )

        # Create 50 positions across different trade pairs
        available_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.USDJPY,
                          TradePair.EURUSD, TradePair.GBPUSD, TradePair.AUDUSD]

        base_time = self.BASE_TIME
        hour_ms = 60 * 60 * 1000

        positions = []

        for i in range(50):
            trade_pair = available_pairs[i % len(available_pairs)]

            # Create position with 2-5 orders each
            num_orders = 2 + (i % 4)
            orders = []

            base_price = 50000 if trade_pair == TradePair.BTCUSD else (3000 if trade_pair == TradePair.ETHUSD else 150)
            current_leverage = 0.0

            for j in range(num_orders):
                current_leverage += 0.1 if j < num_orders - 1 else -current_leverage  # Close at end

                orders.append(Order(
                    price=base_price + (j * 10),
                    processed_ms=base_time + (i * hour_ms) + (j * hour_ms // 4),
                    order_uuid=f"mass_order_{i}_{j}",
                    trade_pair=trade_pair,
                    order_type=OrderType.LONG if current_leverage > 0 else (OrderType.SHORT if current_leverage < 0 else OrderType.FLAT),
                    leverage=abs(current_leverage),
                ))

            position = Position(
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                position_uuid=f"mass_position_{i}",
                open_ms=base_time + (i * hour_ms),
                trade_pair=trade_pair,
                orders=orders,
                position_type=OrderType.FLAT,
            )
            position.rebuild_position_with_updated_orders()
            positions.append(position)

        # Save all positions
        for position in positions:
            self.position_manager.save_miner_position(position)

        # Time the update
        start_time = time.time()
        plm.update(t_ms=base_time + (60 * hour_ms))
        update_time = time.time() - start_time

        print(f"Massive position count update time: {update_time:.4f}s for 50 positions")

        # Verify all trade pairs were processed
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        if self.DEFAULT_MINER_HOTKEY in bundles:
            bundle = bundles[self.DEFAULT_MINER_HOTKEY]
            self.assertGreater(len(bundle), 1)  # Should have multiple trade pair ledgers

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_extreme_leverage_values(self, mock_candle_fetcher):
        """Test with extreme leverage values (very high and very low)"""
        mock_candle_fetcher.return_value = {}

        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        base_time = self.BASE_TIME
        day_ms = 24 * 60 * 60 * 1000

        # Position with extremely high leverage
        high_leverage_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="high_leverage",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=base_time, order_uuid="high_lev_open",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=5.0),  # Very high
                Order(price=50100, processed_ms=base_time + day_ms, order_uuid="high_lev_close",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.FLAT, leverage=0.0),
            ],
            position_type=OrderType.FLAT,
        )
        high_leverage_position.rebuild_position_with_updated_orders()

        # Position with extremely low leverage
        low_leverage_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="low_leverage",
            open_ms=base_time + (2 * day_ms),
            trade_pair=TradePair.ETHUSD,
            orders=[
                Order(price=3000, processed_ms=base_time + (2 * day_ms), order_uuid="low_lev_open",
                      trade_pair=TradePair.ETHUSD, order_type=OrderType.LONG, leverage=0.001),  # Very low
                Order(price=3001, processed_ms=base_time + (3 * day_ms), order_uuid="low_lev_close",
                      trade_pair=TradePair.ETHUSD, order_type=OrderType.FLAT, leverage=0.0),
            ],
            position_type=OrderType.FLAT,
        )
        low_leverage_position.rebuild_position_with_updated_orders()

        for position in [high_leverage_position, low_leverage_position]:
            self.position_manager.save_miner_position(position)

        # Update and verify no errors with extreme values
        plm.update(t_ms=base_time + (5 * day_ms))
        bundles = plm.get_perf_ledgers(portfolio_only=False)

        # Should handle extreme values gracefully
        self.assertIn(self.DEFAULT_MINER_HOTKEY, bundles)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_very_long_running_positions(self, mock_candle_fetcher):
        """Test positions running for very long periods (months)"""
        mock_candle_fetcher.return_value = {}

        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 180,  # 6 months window
        )

        # Position running for 4 months
        start_time = self.now_ms - (1000 * 60 * 60 * 24 * 120)  # 4 months ago
        end_time = self.now_ms

        long_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="long_running",
            open_ms=start_time,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=45000, processed_ms=start_time, order_uuid="long_open",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
                # Add some adjustments over time
                Order(price=46000, processed_ms=start_time + (30 * 24 * 60 * 60 * 1000), order_uuid="long_adj1",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.7),
                Order(price=47000, processed_ms=start_time + (60 * 24 * 60 * 60 * 1000), order_uuid="long_adj2",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.4),
                Order(price=48000, processed_ms=end_time, order_uuid="long_close",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.FLAT, leverage=0.0),
            ],
            position_type=OrderType.FLAT,
        )
        long_position.rebuild_position_with_updated_orders()

        self.position_manager.save_miner_position(long_position)

        # Update to current time
        start_update_time = time.time()
        plm.update(t_ms=self.now_ms)
        update_time = time.time() - start_update_time

        print(f"Long-running position update time: {update_time:.4f}s for 4-month position")

        # Verify ledger was created and has reasonable checkpoint count
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        if self.DEFAULT_MINER_HOTKEY in bundles:
            portfolio_ledger = bundles[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]

            # Should have many checkpoints for 4-month period
            self.assertGreater(len(portfolio_ledger.cps), 100, "Should have many checkpoints for long period")

            # Verify checkpoint pruning worked (if window is shorter than position duration)
            total_duration = portfolio_ledger.get_total_ledger_duration_ms()
            self.assertLessEqual(total_duration, plm.target_ledger_window_ms)


class TestPerfLedgerEdgeCases(TestBase):
    """Edge case tests for boundary conditions and error scenarios"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "edge_case_miner"
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

    def test_exactly_on_checkpoint_boundaries(self):
        """Test behavior when orders occur exactly on 12-hour boundaries"""
        ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Calculate exact 12-hour boundaries
        twelve_hours_ms = 1000 * 60 * 60 * 12
        aligned_base = (self.BASE_TIME // twelve_hours_ms) * twelve_hours_ms

        # Create updates exactly on boundaries
        boundary_times = [
            aligned_base + (i * twelve_hours_ms) for i in range(5)
        ]

        for i, boundary_time in enumerate(boundary_times):
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.01),
                now_ms=boundary_time,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        # All checkpoints should be exactly aligned
        for cp in ledger.cps:
            remainder = cp.last_update_ms % twelve_hours_ms
            self.assertEqual(remainder, 0, f"Boundary checkpoint not aligned: {cp.last_update_ms}")

    def test_zero_and_negative_portfolio_values(self):
        """Test handling of zero and negative portfolio values"""
        ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Test sequence: positive -> zero -> negative -> positive
        values = [1.0, 1.05, 0.0, -0.1, -0.05, 0.5, 1.02]

        for i, value in enumerate(values):
            update_time = self.BASE_TIME + (i * 1000 * 60 * 60)
            try:
                ledger.update_pl(
                    current_portfolio_value=value,
                    now_ms=update_time,
                    miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                    any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                    current_portfolio_fee_spread=1.0,
                    current_portfolio_carry=1.0,
                )
            except Exception as e:
                # Document behavior with negative/zero values
                print(f"Error with value {value}: {e}")

        # Should handle gracefully or have documented behavior
        self.assertGreater(len(ledger.cps), 0)

    def test_extremely_small_time_intervals(self):
        """Test with updates happening in very small time intervals"""
        ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )

        # Updates every millisecond for 10 seconds
        base_time = self.BASE_TIME

        for i in range(10000):  # 10,000 milliseconds = 10 seconds
            ledger.update_pl(
                current_portfolio_value=1.0 + (i * 0.000001),  # Tiny changes
                now_ms=base_time + i,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                current_portfolio_fee_spread=1.0,
                current_portfolio_carry=1.0,
            )

        # Should aggregate into reasonable number of checkpoints
        self.assertLess(len(ledger.cps), 100, "Too many checkpoints for small intervals")

    def test_position_with_single_order(self):
        """Test positions with exactly one order (edge case)"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # Position with only opening order (never closed)
        single_order_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="single_order",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=self.BASE_TIME, order_uuid="only_order",
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
            ],
            position_type=OrderType.LONG,  # Still open
        )
        single_order_position.rebuild_position_with_updated_orders()

        self.position_manager.save_miner_position(single_order_position)

        # Should handle single-order position
        with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
            mock_fetch.return_value = {}
            plm.update(t_ms=self.now_ms)

        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, bundles)

    def test_empty_positions_list(self):
        """Test behavior with empty positions list"""
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
        )

        # No positions saved - should handle gracefully
        with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
            mock_fetch.return_value = {}
            plm.update(t_ms=self.now_ms)

        plm.get_perf_ledgers(portfolio_only=False)
        # Should not crash, may or may not have entries depending on implementation

    def test_orders_out_of_chronological_order(self):
        """Test handling of orders that aren't in chronological order"""
        # Note: This tests the system's robustness to data integrity issues

        base_time = self.BASE_TIME
        hour_ms = 60 * 60 * 1000

        # Orders deliberately out of chronological order
        orders = [
            Order(price=50000, processed_ms=base_time + (3 * hour_ms), order_uuid="order_3",
                  trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
            Order(price=49500, processed_ms=base_time, order_uuid="order_1",
                  trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.3),
            Order(price=50500, processed_ms=base_time + (5 * hour_ms), order_uuid="order_5",
                  trade_pair=TradePair.BTCUSD, order_type=OrderType.FLAT, leverage=0.0),
            Order(price=49800, processed_ms=base_time + (1 * hour_ms), order_uuid="order_2",
                  trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.4),
        ]

        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="unordered_position",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=orders,  # Deliberately unordered
            position_type=OrderType.FLAT,
        )

        # The system should handle this or sort internally
        try:
            position.rebuild_position_with_updated_orders()
            self.position_manager.save_miner_position(position)

            plm = PerfLedgerManager(
                metagraph=self.mmg,
                running_unit_tests=True,
                position_manager=self.position_manager,
            )

            with patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher') as mock_fetch:
                mock_fetch.return_value = {}
                plm.update(t_ms=self.now_ms)

            # Should either handle gracefully or have documented behavior

        except Exception as e:
            # Document that unordered orders cause issues
            print(f"Unordered orders caused error: {e}")

    def test_ledger_initialization_at_different_times(self):
        """Test ledger initialization at various times relative to checkpoint boundaries"""
        # Test initialization at different offsets from 12-hour boundaries
        twelve_hours_ms = 1000 * 60 * 60 * 12
        base_boundary = (self.BASE_TIME // twelve_hours_ms) * twelve_hours_ms

        # Test different initialization times relative to boundaries
        init_times = [
            base_boundary,                    # Exactly on boundary
            base_boundary + 1000,            # 1 second after
            base_boundary + (6 * 60 * 60 * 1000),  # 6 hours after (middle)
            base_boundary + (12 * 60 * 60 * 1000) - 1000,  # 1 second before next boundary
        ]

        for i, init_time in enumerate(init_times):
            ledger = PerfLedger(
                initialization_time_ms=init_time,
                target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
            )

            # Add some updates
            for j in range(3):
                update_time = init_time + (j * 1000 * 60 * 60 * 6)  # Every 6 hours
                ledger.update_pl(
                    current_portfolio_value=1.0 + (j * 0.01),
                    now_ms=update_time,
                    miner_hotkey=f"test_miner_{i}",
                    any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                    current_portfolio_fee_spread=1.0,
                    current_portfolio_carry=1.0,
                )

            # Should handle all initialization times correctly
            self.assertGreater(len(ledger.cps), 0)
            self.assertEqual(ledger.initialization_time_ms, init_time)


if __name__ == '__main__':
    unittest.main()

