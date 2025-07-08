import math
import unittest

from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    PerfCheckpoint,
    TradePairReturnStatus,
)
from vali_objects.vali_dataclasses.perf_ledger_utils import (
    PerfLedgerMath,
    PerfLedgerValidator,
)


class TestPerfLedgerMath(TestBase):
    """Tests for mathematical utility functions in performance ledgers"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "math_test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.BASE_TIME = self.now_ms - (1000 * 60 * 60 * 24 * 7)  # 1 week ago

    def test_compute_return_delta_logarithmic(self):
        """Test logarithmic return delta calculations"""

        # Test normal positive returns
        current_value = 1.05
        previous_value = 1.0

        log_delta = PerfLedgerMath.compute_return_delta(current_value, previous_value, use_log=True)
        expected_log = math.log(1.05)
        self.assertAlmostEqual(log_delta, expected_log, places=6)

        # Test with different values
        test_cases = [
            (1.1, 1.0, math.log(1.1)),
            (0.95, 1.0, math.log(0.95)),
            (2.0, 1.5, math.log(2.0/1.5)),
            (1.001, 1.0, math.log(1.001)),
        ]

        for current, previous, expected in test_cases:
            with self.subTest(current=current, previous=previous):
                result = PerfLedgerMath.compute_return_delta(current, previous, use_log=True)
                self.assertAlmostEqual(result, expected, places=6)

    def test_compute_return_delta_percentage(self):
        """Test percentage return delta calculations"""

        # Test normal positive returns
        current_value = 1.05
        previous_value = 1.0

        pct_delta = PerfLedgerMath.compute_return_delta(current_value, previous_value, use_log=False)
        expected_pct = 0.05  # 5% return
        self.assertAlmostEqual(pct_delta, expected_pct, places=6)

        # Test with different values
        test_cases = [
            (1.1, 1.0, 0.1),      # 10% gain
            (0.9, 1.0, -0.1),     # 10% loss
            (1.5, 1.0, 0.5),      # 50% gain
            (2.0, 1.0, 1.0),      # 100% gain
        ]

        for current, previous, expected in test_cases:
            with self.subTest(current=current, previous=previous):
                result = PerfLedgerMath.compute_return_delta(current, previous, use_log=False)
                self.assertAlmostEqual(result, expected, places=6)

    def test_compute_return_delta_invalid_values(self):
        """Test return delta with invalid input values"""

        # Test negative values
        with self.assertRaises(ValueError):
            PerfLedgerMath.compute_return_delta(-1.0, 1.0)

        with self.assertRaises(ValueError):
            PerfLedgerMath.compute_return_delta(1.0, -1.0)

        # Test zero values
        with self.assertRaises(ValueError):
            PerfLedgerMath.compute_return_delta(0.0, 1.0)

        with self.assertRaises(ValueError):
            PerfLedgerMath.compute_return_delta(1.0, 0.0)

    def test_compute_simple_delta(self):
        """Test simple arithmetic delta calculations"""

        test_cases = [
            (1.05, 1.0, 0.05),
            (0.95, 1.0, -0.05),
            (2.0, 1.5, 0.5),
            (100.0, 90.0, 10.0),
            (-5.0, -10.0, 5.0),
        ]

        for current, previous, expected in test_cases:
            with self.subTest(current=current, previous=previous):
                result = PerfLedgerMath.compute_simple_delta(current, previous)
                self.assertAlmostEqual(result, expected, places=6)

    def test_update_maximum_drawdown(self):
        """Test maximum drawdown calculation updates"""

        # Test initial case - new high
        current_value = 1.1
        max_portfolio_value = 1.0
        current_mdd = 1.0

        new_max, new_mdd = PerfLedgerMath.update_maximum_drawdown(
            current_value, max_portfolio_value, current_mdd,
        )

        self.assertEqual(new_max, 1.1)  # Should update max
        self.assertEqual(new_mdd, 1.0)  # MDD unchanged (no drawdown)

        # Test drawdown case
        current_value = 0.9
        max_portfolio_value = 1.1
        current_mdd = 1.0

        new_max, new_mdd = PerfLedgerMath.update_maximum_drawdown(
            current_value, max_portfolio_value, current_mdd,
        )

        self.assertEqual(new_max, 1.1)  # Max unchanged
        expected_drawdown = 0.9 / 1.1  # ~0.818
        self.assertAlmostEqual(new_mdd, expected_drawdown, places=6)

        # Test sequence of values
        test_sequence = [
            (1.0, 1.0, 1.0),    # Initial
            (1.2, 1.2, 1.0),    # New high
            (1.1, 1.2, 1.1/1.2), # Drawdown
            (0.9, 1.2, 0.9/1.2), # Deeper drawdown
            (1.3, 1.3, 0.9/1.2), # New high, but MDD preserved
        ]

        max_val = 0.0
        mdd = 1.0

        for current, expected_max, expected_mdd in test_sequence:
            max_val, mdd = PerfLedgerMath.update_maximum_drawdown(current, max_val, mdd)
            self.assertAlmostEqual(max_val, expected_max, places=6)
            self.assertAlmostEqual(mdd, expected_mdd, places=6)

    def test_calculate_fee_delta(self):
        """Test fee delta calculations with validation"""

        # Test normal fee progression
        current_fee = 0.995
        previous_fee = 1.0

        fee_delta = PerfLedgerMath.calculate_fee_delta(current_fee, previous_fee)
        self.assertAlmostEqual(fee_delta, -0.005, places=6)  # Fees should decrease value

        # Test various fee scenarios
        test_cases = [
            (0.99, 1.0, -0.01),   # 1% fee loss
            (0.995, 0.99, 0.005), # Fee improvement
            (1.0, 1.0, 0.0),      # No change
            (0.0, 0.5, -0.5),     # Large fee loss
        ]

        for current, previous, expected in test_cases:
            with self.subTest(current=current, previous=previous):
                result = PerfLedgerMath.calculate_fee_delta(current, previous)
                self.assertAlmostEqual(result, expected, places=6)

    def test_calculate_fee_delta_invalid_values(self):
        """Test fee delta with invalid input values"""

        # Test negative fees
        with self.assertRaises(ValueError):
            PerfLedgerMath.calculate_fee_delta(-0.1, 1.0)

        with self.assertRaises(ValueError):
            PerfLedgerMath.calculate_fee_delta(1.0, -0.1)

        # Test fees > 1
        with self.assertRaises(ValueError):
            PerfLedgerMath.calculate_fee_delta(1.1, 1.0)

        with self.assertRaises(ValueError):
            PerfLedgerMath.calculate_fee_delta(1.0, 1.1)

    def test_validate_checkpoint_data(self):
        """Test checkpoint data validation"""

        # Create valid checkpoint
        valid_checkpoint = PerfCheckpoint(
            last_update_ms=self.now_ms,
            prev_portfolio_ret=1.05,
            prev_portfolio_spread_fee=0.999,
            prev_portfolio_carry_fee=0.998,
            accum_ms=1000 * 60 * 60,
            open_ms=self.BASE_TIME,
            n_updates=5,
            gain=0.05,
            loss=0.0,
        )

        # Should pass validation
        self.assertTrue(PerfLedgerMath.validate_checkpoint_data(valid_checkpoint))

        # Test invalid timestamp
        invalid_checkpoint = PerfCheckpoint(
            last_update_ms=-1,  # Invalid negative timestamp
            prev_portfolio_ret=1.05,
            prev_portfolio_spread_fee=0.999,
            prev_portfolio_carry_fee=0.998,
            gain=0.05,
            loss=0.0,
        )

        with self.assertRaises(ValueError):
            PerfLedgerMath.validate_checkpoint_data(invalid_checkpoint)

        # Test invalid portfolio return
        invalid_checkpoint = PerfCheckpoint(
            last_update_ms=self.now_ms,
            prev_portfolio_ret=0.0,  # Invalid zero return
            prev_portfolio_spread_fee=0.999,
            prev_portfolio_carry_fee=0.998,
            gain=0.05,
            loss=0.0,
        )

        with self.assertRaises(ValueError):
            PerfLedgerMath.validate_checkpoint_data(invalid_checkpoint)

    def test_calculate_time_weighted_return(self):
        """Test time-weighted return calculation"""

        # Create test checkpoints
        checkpoints = [
            PerfCheckpoint(
                last_update_ms=self.BASE_TIME,
                prev_portfolio_ret=1.0,
                open_ms=self.BASE_TIME,
            ),
            PerfCheckpoint(
                last_update_ms=self.BASE_TIME + 1000 * 60 * 60 * 24,
                prev_portfolio_ret=1.05,
                open_ms=self.BASE_TIME + 1000 * 60 * 60 * 24,
            ),
            PerfCheckpoint(
                last_update_ms=self.BASE_TIME + 1000 * 60 * 60 * 48,
                prev_portfolio_ret=1.02,
                open_ms=self.BASE_TIME + 1000 * 60 * 60 * 48,
            ),
        ]

        twr = PerfLedgerMath.calculate_time_weighted_return(checkpoints)
        self.assertGreater(twr, 0.0)
        self.assertIsInstance(twr, float)

        # Test empty checkpoints
        empty_twr = PerfLedgerMath.calculate_time_weighted_return([])
        self.assertEqual(empty_twr, 1.0)

        # Test single checkpoint
        single_twr = PerfLedgerMath.calculate_time_weighted_return(checkpoints[:1])
        self.assertEqual(single_twr, 1.0)

    def test_calculate_volatility(self):
        """Test volatility calculation from checkpoints"""

        # Create checkpoints with varying returns
        checkpoints = []
        returns = [1.0, 1.02, 0.98, 1.05, 0.95, 1.03, 0.97, 1.01]

        for i, ret in enumerate(returns):
            checkpoint = PerfCheckpoint(
                last_update_ms=self.BASE_TIME + (i * 1000 * 60 * 60 * 24),
                prev_portfolio_ret=ret,
                open_ms=self.BASE_TIME + (i * 1000 * 60 * 60 * 24),
            )
            checkpoints.append(checkpoint)

        volatility = PerfLedgerMath.calculate_volatility(checkpoints)
        self.assertGreater(volatility, 0.0)
        self.assertIsInstance(volatility, float)

        # Test with insufficient data
        low_vol = PerfLedgerMath.calculate_volatility(checkpoints[:1])
        self.assertEqual(low_vol, 0.0)

        # Test with constant returns (should have low volatility)
        constant_checkpoints = []
        for i in range(5):
            checkpoint = PerfCheckpoint(
                last_update_ms=self.BASE_TIME + (i * 1000 * 60 * 60 * 24),
                prev_portfolio_ret=1.02,  # Constant return
                open_ms=self.BASE_TIME + (i * 1000 * 60 * 60 * 24),
            )
            constant_checkpoints.append(checkpoint)

        constant_vol = PerfLedgerMath.calculate_volatility(constant_checkpoints)
        self.assertAlmostEqual(constant_vol, 0.0, places=10)


class TestPerfLedgerValidator(TestBase):
    """Tests for validation utility functions"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "validator_test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.BASE_TIME = self.now_ms - (1000 * 60 * 60 * 24 * 7)

    def test_validate_position_consistency(self):
        """Test position consistency validation"""

        # Create valid positions
        order1 = Order(
            price=50000,
            processed_ms=self.BASE_TIME,
            order_uuid="order_1",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5,
        )

        order2 = Order(
            price=51000,
            processed_ms=self.BASE_TIME + 1000 * 60 * 60,
            order_uuid="order_2",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0,
        )

        valid_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="valid_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[order1, order2],
            position_type=OrderType.FLAT,
        )

        # Should pass validation
        self.assertTrue(
            PerfLedgerValidator.validate_position_consistency(
                [valid_position], self.DEFAULT_MINER_HOTKEY,
            ),
        )

        # Test wrong miner hotkey
        wrong_miner_position = Position(
            miner_hotkey="wrong_miner",
            position_uuid="wrong_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[order1],
            position_type=OrderType.LONG,
        )

        with self.assertRaises(ValueError):
            PerfLedgerValidator.validate_position_consistency(
                [wrong_miner_position], self.DEFAULT_MINER_HOTKEY,
            )

        # Test empty orders
        empty_orders_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="empty_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[],  # Empty orders
            position_type=OrderType.FLAT,
        )

        with self.assertRaises(ValueError):
            PerfLedgerValidator.validate_position_consistency(
                [empty_orders_position], self.DEFAULT_MINER_HOTKEY,
            )

    def test_validate_position_chronological_order(self):
        """Test that orders are validated for chronological order"""

        # Create orders out of chronological order
        order1 = Order(
            price=50000,
            processed_ms=self.BASE_TIME + 1000 * 60 * 60,  # Later time
            order_uuid="order_1",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5,
        )

        order2 = Order(
            price=51000,
            processed_ms=self.BASE_TIME,  # Earlier time
            order_uuid="order_2",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0,
        )

        invalid_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="invalid_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[order1, order2],  # Wrong chronological order
            position_type=OrderType.FLAT,
        )

        with self.assertRaises(ValueError):
            PerfLedgerValidator.validate_position_consistency(
                [invalid_position], self.DEFAULT_MINER_HOTKEY,
            )

    def test_validate_ledger_integrity(self):
        """Test ledger bundle integrity validation"""
        from vali_objects.vali_dataclasses.perf_ledger import (
            TP_ID_PORTFOLIO,
            PerfLedger,
        )

        # Create valid ledger bundle
        portfolio_ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )
        # Update the ledger to set last_update_ms
        portfolio_ledger.update_pl(
            current_portfolio_value=1.0,
            now_ms=self.now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0,
        )

        trade_pair_ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )
        # Update the ledger to set last_update_ms
        trade_pair_ledger.update_pl(
            current_portfolio_value=1.0,
            now_ms=self.now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0,
        )

        valid_bundle = {
            TP_ID_PORTFOLIO: portfolio_ledger,
            TradePair.BTCUSD.trade_pair_id: trade_pair_ledger,
        }

        # Should pass validation
        self.assertTrue(PerfLedgerValidator.validate_ledger_integrity(valid_bundle))

        # Test invalid bundle type
        with self.assertRaises(ValueError):
            PerfLedgerValidator.validate_ledger_integrity("not_a_dict")

        # Test missing portfolio ledger
        invalid_bundle = {
            TradePair.BTCUSD.trade_pair_id: trade_pair_ledger,
        }

        with self.assertRaises(ValueError):
            PerfLedgerValidator.validate_ledger_integrity(invalid_bundle)

        # Test inconsistent timing
        inconsistent_trade_pair_ledger = PerfLedger(
            initialization_time_ms=self.BASE_TIME,
            target_ledger_window_ms=1000 * 60 * 60 * 24 * 30,
        )
        # Update with different time
        inconsistent_trade_pair_ledger.update_pl(
            current_portfolio_value=1.0,
            now_ms=self.now_ms - 1000,  # Different time
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            any_open=TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            current_portfolio_fee_spread=1.0,
            current_portfolio_carry=1.0,
        )

        inconsistent_bundle = {
            TP_ID_PORTFOLIO: portfolio_ledger,
            TradePair.BTCUSD.trade_pair_id: inconsistent_trade_pair_ledger,
        }

        with self.assertRaises(ValueError):
            PerfLedgerValidator.validate_ledger_integrity(inconsistent_bundle)

    def test_edge_case_validations(self):
        """Test edge cases in validation functions"""

        # Test empty position list
        self.assertTrue(
            PerfLedgerValidator.validate_position_consistency([], self.DEFAULT_MINER_HOTKEY),
        )

        # Test single valid order
        single_order = Order(
            price=50000,
            processed_ms=self.BASE_TIME,
            order_uuid="single_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5,
        )

        single_order_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="single_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=[single_order],
            position_type=OrderType.LONG,
        )

        self.assertTrue(
            PerfLedgerValidator.validate_position_consistency(
                [single_order_position], self.DEFAULT_MINER_HOTKEY,
            ),
        )


if __name__ == '__main__':
    unittest.main()
