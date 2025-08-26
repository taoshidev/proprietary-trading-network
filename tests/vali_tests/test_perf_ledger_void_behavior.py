"""
Performance ledger void behavior tests.

This file consolidates all void-related tests:
- Void filling and bypass logic
- Floating point drift prevention
- Multi-trade pair void scenarios
- Edge cases during void periods
"""

import unittest
from unittest.mock import patch, Mock

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    PerfLedger,
    PerfLedgerManager,
    PerfCheckpoint,
    TP_ID_PORTFOLIO,
    ParallelizationMode,
    TradePairReturnStatus,
)


class TestPerfLedgerVoidBehavior(TestBase):
    """Tests for performance ledger void period behavior."""

    def setUp(self):
        super().setUp()
        self.test_hotkey = "test_miner_void"
        self.now_ms = TimeUtil.now_in_millis()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        self.mmg = MockMetagraph(hotkeys=[self.test_hotkey])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        self.position_manager.clear_all_miner_positions()

    def validate_void_checkpoint(self, cp: PerfCheckpoint, context: str = ""):
        """Validate void checkpoint has expected characteristics."""
        # Void checkpoints should have 0 updates
        self.assertEqual(cp.n_updates, 0, f"{context}: void checkpoint should have 0 updates")
        
        # Void checkpoints should have no new gains/losses
        self.assertEqual(cp.gain, 0.0, f"{context}: void checkpoint should have no gain")
        self.assertEqual(cp.loss, 0.0, f"{context}: void checkpoint should have no loss")
        
        # Portfolio values should be reasonable
        self.assertGreater(cp.prev_portfolio_ret, 0.0, f"{context}: portfolio return should be positive")
        self.assertGreater(cp.prev_portfolio_spread_fee, 0.0, f"{context}: spread fee should be positive")
        self.assertGreater(cp.prev_portfolio_carry_fee, 0.0, f"{context}: carry fee should be positive")
        
        # Risk metrics should be reasonable
        self.assertGreater(cp.mdd, 0.0, f"{context}: MDD should be positive")
        self.assertGreater(cp.mpv, 0.0, f"{context}: MPV should be positive")
        
        # Carry fee loss during void should be 0 (this was the original bug)
        self.assertEqual(cp.carry_fee_loss, 0.0, f"{context}: void checkpoint should have 0 carry_fee_loss")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_void_filling_prevents_drift(self, mock_lpf):
        """
        Test that void filling with bypass logic prevents floating point drift.
        This is the core test for the original bug fix.
        """
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        base_time = (self.now_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS - (30 * MS_IN_24_HOURS)
        
        # Create position that will generate carry fees
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid="drift_test",
            open_ms=base_time,
            close_ms=base_time + (3 * MS_IN_24_HOURS),
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(
                    price=50000.0,
                    processed_ms=base_time,
                    order_uuid="open",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=1.0,
                ),
                Order(
                    price=50000.0,
                    processed_ms=base_time + (3 * MS_IN_24_HOURS),
                    order_uuid="close",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.FLAT,
                    leverage=0.0,
                )
            ],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        self.position_manager.save_miner_position(position)
        
        # Process position
        plm.update(t_ms=base_time + (3 * MS_IN_24_HOURS) + 1000)
        
        # Get checkpoint values at close
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Find last active checkpoint
        close_checkpoint = None
        for cp in reversed(btc_ledger.cps):
            if cp.n_updates > 0:
                close_checkpoint = cp
                break
        
        self.assertIsNotNone(close_checkpoint)
        
        # Perform many void updates
        void_checkpoints = []
        for i in range(50):  # 50 days of void
            plm.update(t_ms=base_time + (4 + i) * MS_IN_24_HOURS)
            
            bundles = plm.get_perf_ledgers(portfolio_only=False)
            btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
            
            for cp in btc_ledger.cps:
                if cp.n_updates == 0 and cp.last_update_ms > base_time + (3 * MS_IN_24_HOURS):
                    if cp not in void_checkpoints:
                        void_checkpoints.append(cp)
        
        # Verify no drift - all void checkpoints should be identical
        self.assertGreater(len(void_checkpoints), 40)
        
        if void_checkpoints:
            ref = void_checkpoints[0]
            for i, cp in enumerate(void_checkpoints):
                # Exact equality - no tolerance
                self.assertEqual(cp.prev_portfolio_ret, ref.prev_portfolio_ret,
                               f"Void checkpoint {i}: return drifted")
                self.assertEqual(cp.prev_portfolio_carry_fee, ref.prev_portfolio_carry_fee,
                               f"Void checkpoint {i}: carry fee drifted")
                self.assertEqual(cp.prev_portfolio_spread_fee, ref.prev_portfolio_spread_fee,
                               f"Void checkpoint {i}: spread fee drifted")
                self.assertEqual(cp.mdd, ref.mdd,
                               f"Void checkpoint {i}: MDD drifted")
                
                # Validate this as a proper void checkpoint
                self.validate_void_checkpoint(cp, f"Void checkpoint {i}")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_multi_tp_staggered_void_periods(self, mock_lpf):
        """Test void behavior with multiple trade pairs having different timings."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        base_time = (self.now_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS - (30 * MS_IN_24_HOURS)
        
        # Create staggered positions
        positions = [
            ("btc", TradePair.BTCUSD, 0, 10),    # Days 0-10
            ("eth", TradePair.ETHUSD, 5, 15),    # Days 5-15 (overlaps BTC)
            ("jpy", TradePair.USDJPY, 12, 18),   # Days 12-18
        ]
        
        for name, tp, start_day, end_day in positions:
            position = self._create_position(
                name, tp,
                base_time + (start_day * MS_IN_24_HOURS),
                base_time + (end_day * MS_IN_24_HOURS),
                1000.0, 1000.0, OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
        
        # Update to day 25
        plm.update(t_ms=base_time + (25 * MS_IN_24_HOURS))
        
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        bundle = bundles[self.test_hotkey]
        
        # Verify each TP has correct void period
        btc_ledger = bundle[TradePair.BTCUSD.trade_pair_id]
        eth_ledger = bundle[TradePair.ETHUSD.trade_pair_id]
        jpy_ledger = bundle[TradePair.USDJPY.trade_pair_id]
        
        # Count void checkpoints for each
        btc_void = sum(1 for cp in btc_ledger.cps if cp.n_updates == 0 and 
                      cp.last_update_ms > base_time + (10 * MS_IN_24_HOURS))
        eth_void = sum(1 for cp in eth_ledger.cps if cp.n_updates == 0 and
                      cp.last_update_ms > base_time + (15 * MS_IN_24_HOURS))
        jpy_void = sum(1 for cp in jpy_ledger.cps if cp.n_updates == 0 and
                      cp.last_update_ms > base_time + (18 * MS_IN_24_HOURS))
        
        # BTC should have most void checkpoints (closed earliest)
        self.assertGreater(btc_void, eth_void)
        self.assertGreater(eth_void, jpy_void)

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_bypass_logic_direct(self, mock_lpf):
        """Test the bypass logic utility function directly."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        mmg = MockMetagraph(hotkeys=["test"])
        plm = PerfLedgerManager(
            metagraph=mmg,
            running_unit_tests=True,
            position_manager=PositionManager(
                metagraph=mmg,
                running_unit_tests=True,
                elimination_manager=EliminationManager(mmg, None, None),
            ),
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        # Create test ledger with checkpoint
        ledger = PerfLedger(initialization_time_ms=self.now_ms)
        prev_cp = PerfCheckpoint(
            last_update_ms=self.now_ms,
            prev_portfolio_ret=0.95,
            prev_portfolio_spread_fee=0.999,
            prev_portfolio_carry_fee=0.998,
            mdd=0.95,
            mpv=1.0
        )
        ledger.cps.append(prev_cp)
        
        # Test case 1: Should use bypass
        ret, spread, carry = plm.get_bypass_values_if_applicable(
            ledger, "BTCUSD", TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
            1.0, 1.0, 1.0, {"BTCUSD": None}
        )
        self.assertEqual(ret, 0.95)
        self.assertEqual(spread, 0.999)
        self.assertEqual(carry, 0.998)
        
        # Test case 2: Should NOT use bypass (position just closed)
        # Create a mock closed position to simulate a position that just closed
        mock_closed_position = Mock()
        mock_closed_position.is_open_position = False
        ret, spread, carry = plm.get_bypass_values_if_applicable(
            ledger, "BTCUSD", TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
            1.0, 1.0, 1.0, {"BTCUSD": mock_closed_position}
        )
        self.assertEqual(ret, 1.0)
        
        # Test case 3: Should NOT use bypass (positions open)
        ret, spread, carry = plm.get_bypass_values_if_applicable(
            ledger, "BTCUSD", TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            1.0, 1.0, 1.0, {"BTCUSD": None}
        )
        self.assertEqual(ret, 1.0)
        
        # Test case 4: Should NOT use bypass (different TP)
        ret, spread, carry = plm.get_bypass_values_if_applicable(
            ledger, "ETHUSD", TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
            1.0, 1.0, 1.0, {"BTCUSD": None}
        )
        self.assertEqual(ret, 1.0)

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_void_checkpoint_characteristics(self, mock_lpf):
        """Test that void checkpoints have expected characteristics."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        base_time = self.now_ms - (10 * MS_IN_24_HOURS)
        
        # Create and close position
        position = self._create_position(
            "char_test", TradePair.BTCUSD,
            base_time, base_time + MS_IN_24_HOURS,
            50000.0, 50000.0, OrderType.LONG
        )
        self.position_manager.save_miner_position(position)
        
        # Update through void period
        plm.update(t_ms=base_time + (5 * MS_IN_24_HOURS))
        
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Check void checkpoint characteristics
        void_checkpoint_count = 0
        for cp in btc_ledger.cps:
            if cp.n_updates == 0 and cp.last_update_ms > base_time + MS_IN_24_HOURS:
                void_checkpoint_count += 1
                # Validate void checkpoint characteristics
                self.validate_void_checkpoint(cp, f"Void checkpoint at {cp.last_update_ms}")
        
        self.assertGreater(void_checkpoint_count, 0, "Should have found at least one void checkpoint")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_position_calculation_efficiency(self, mock_lpf):
        """Test that bypass logic reduces unnecessary calculations."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        # Track method calls
        calculation_count = {'count': 0}
        original_method = Position.calculate_return_with_fees
        
        def tracking_method(self, *args, **kwargs):
            calculation_count['count'] += 1
            return original_method(self, *args, **kwargs)
        
        Position.calculate_return_with_fees = tracking_method
        
        try:
            plm = PerfLedgerManager(
                metagraph=self.mmg,
                running_unit_tests=True,
                position_manager=self.position_manager,
                parallel_mode=ParallelizationMode.SERIAL,
            )
            
            base_time = self.now_ms - (10 * MS_IN_24_HOURS)
            
            # Create position
            position = self._create_position(
                "eff_test", TradePair.BTCUSD,
                base_time, base_time + MS_IN_24_HOURS,
                50000.0, 50000.0, OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
            
            # Update to close
            plm.update(t_ms=base_time + (2 * MS_IN_24_HOURS))
            calls_before_void = calculation_count['count']
            
            # Multiple void updates
            for i in range(10):
                plm.update(t_ms=base_time + (3 + i) * MS_IN_24_HOURS)
            
            calls_during_void = calculation_count['count'] - calls_before_void
            
            # During void, calculations should be minimal due to bypass
            # Allow some calls for portfolio calculations but not many
            self.assertLess(calls_during_void, 25,
                           f"Too many calculations during void: {calls_during_void}")
            
        finally:
            Position.calculate_return_with_fees = original_method

    def _create_position(self, position_id: str, trade_pair: TradePair, 
                        open_ms: int, close_ms: int, open_price: float, 
                        close_price: float, order_type: OrderType) -> Position:
        """Helper to create a position."""
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid=position_id,
            open_ms=open_ms,
            close_ms=close_ms,
            trade_pair=trade_pair,
            orders=[
                Order(
                    price=open_price,
                    processed_ms=open_ms,
                    order_uuid=f"{position_id}_open",
                    trade_pair=trade_pair,
                    order_type=order_type,
                    leverage=1.0 if order_type == OrderType.LONG else -1.0,
                ),
                Order(
                    price=close_price,
                    processed_ms=close_ms,
                    order_uuid=f"{position_id}_close",
                    trade_pair=trade_pair,
                    order_type=OrderType.FLAT,
                    leverage=0.0,
                )
            ],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        return position


if __name__ == '__main__':
    unittest.main()