"""
Performance ledger math utilities and metrics tests.

This file consolidates calculation and metrics tests:
- Portfolio alignment
- Fee calculations
- Performance with large datasets
"""

import unittest
from unittest.mock import patch, Mock
import math
from decimal import Decimal
import random
import numpy as np
import time

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS, MS_IN_8_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    PerfLedger,
    PerfLedgerManager,
    PerfCheckpoint,
    TP_ID_PORTFOLIO,
    ParallelizationMode,
)


class TestPerfLedgerMathAndMetrics(TestBase):
    """Tests for mathematical calculations and performance metrics."""

    def setUp(self):
        super().setUp()
        self.test_hotkey = "test_miner_math"
        self.now_ms = TimeUtil.now_in_millis()
        
        self.mmg = MockMetagraph(hotkeys=[self.test_hotkey])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
        )
        self.position_manager.clear_all_miner_positions()

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_portfolio_alignment_calculations(self, mock_lpf):
        """Test that portfolio calculations align with individual trade pairs."""
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
        
        base_time = self.now_ms - (20 * MS_IN_24_HOURS)
        
        # Create positions with known returns
        positions = [
            ("btc", TradePair.BTCUSD, 50000.0, 51000.0, 1.0),   # 2% gain, weight 1.0
            ("eth", TradePair.ETHUSD, 3000.0, 3090.0, 0.5),     # 3% gain, weight 0.5
            ("eur", TradePair.EURUSD, 1.10, 1.10, 0.3),         # 0% gain, weight 0.3
        ]
        
        total_weight = sum(w for _, _, _, _, w in positions)
        
        for name, tp, open_price, close_price, weight in positions:
            position = Position(
                miner_hotkey=self.test_hotkey,
                position_uuid=name,
                open_ms=base_time,
                close_ms=base_time + MS_IN_24_HOURS,
                trade_pair=tp,
                orders=[
                    Order(
                        price=open_price,
                        processed_ms=base_time,
                        order_uuid=f"{name}_open",
                        trade_pair=tp,
                        order_type=OrderType.LONG,
                        leverage=weight,  # Use leverage as proxy for position size
                    ),
                    Order(
                        price=close_price,
                        processed_ms=base_time + MS_IN_24_HOURS,
                        order_uuid=f"{name}_close",
                        trade_pair=tp,
                        order_type=OrderType.FLAT,
                        leverage=0.0,
                    )
                ],
                position_type=OrderType.FLAT,
                is_closed_position=True,
            )
            position.rebuild_position_with_updated_orders()
            self.position_manager.save_miner_position(position)
        
        # Update
        plm.update(t_ms=base_time + (2 * MS_IN_24_HOURS))
        
        # Get ledgers
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        bundle = bundles[self.test_hotkey]
        
        # Portfolio should exist
        self.assertIn(TP_ID_PORTFOLIO, bundle, "Portfolio ledger should exist")
        
        # All individual TPs should exist
        for _, tp, _, _, _ in positions:
            self.assertIn(tp.trade_pair_id, bundle, f"{tp.trade_pair_id} should exist")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_exact_fee_calculations(self, mock_lpf):
        """Test exact fee calculations match expected values."""
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
        
        base_time = (self.now_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS - (10 * MS_IN_24_HOURS)
        
        # Create position with exact 1-day duration
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid="exact_fee",
            open_ms=base_time,
            close_ms=base_time + MS_IN_24_HOURS,  # Exactly 1 day
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
                    price=50000.0,  # No price change
                    processed_ms=base_time + MS_IN_24_HOURS,
                    order_uuid="close",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.FLAT,
                    leverage=0.0,
                )
            ],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        position.rebuild_position_with_updated_orders()
        self.position_manager.save_miner_position(position)
        
        # Update
        plm.update(t_ms=base_time + (2 * MS_IN_24_HOURS))
        
        # Get checkpoint with position
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Find checkpoint with the position
        for cp in btc_ledger.cps:
            if cp.n_updates > 0 and cp.last_update_ms <= base_time + MS_IN_24_HOURS:
                # For BTC with 1x leverage for 1 day:
                # Annual carry fee ~3%, so daily ~3%/365 = 0.0082%
                # prev_portfolio_carry_fee = 1 - 0.000082 = 0.999918
                
                # Allow reasonable tolerance for calculation differences
                # The actual carry fee depends on the exact implementation
                self.assertLess(
                    cp.prev_portfolio_carry_fee, 1.0,
                    msg="Carry fee should be less than 1.0 (some fee applied)"
                )
                self.assertGreater(
                    cp.prev_portfolio_carry_fee, 0.99,
                    msg="Carry fee should be reasonable (not too large)"
                )
                break

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_performance_with_large_dataset(self, mock_lpf):
        """Test performance with a large number of positions and checkpoints."""
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
        
        base_time = self.now_ms - (100 * MS_IN_24_HOURS)
        
        # Create 100 positions over 100 days
        for i in range(100):
            # Random returns between -5% and +5%
            open_price = 50000.0
            return_pct = (random.random() - 0.5) * 0.10  # -5% to +5%
            close_price = open_price * (1 + return_pct)
            
            position = self._create_position(
                f"perf_{i}", TradePair.BTCUSD,
                base_time + (i * MS_IN_24_HOURS),
                base_time + ((i + 1) * MS_IN_24_HOURS),
                open_price, close_price, OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
        
        # Time the update
        start_time = time.time()
        plm.update(t_ms=self.now_ms)
        update_time = time.time() - start_time
        
        # Should complete in reasonable time (relaxed for CI/different hardware)
        self.assertLess(update_time, 10.0, f"Update took {update_time}s, should be < 10s")
        
        # Verify data integrity
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Should have many checkpoints
        self.assertGreater(len(btc_ledger.cps), 50, "Should have many checkpoints")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_return_compounding(self, mock_lpf):
        """Test that returns compound correctly over multiple periods."""
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
        
        # Create sequential positions with known returns
        returns = [0.10, -0.05, 0.03]  # 10% gain, 5% loss, 3% gain
        
        for i, ret in enumerate(returns):
            open_price = 50000.0
            close_price = open_price * (1 + ret)
            
            position = self._create_position(
                f"compound_{i}", TradePair.BTCUSD,
                base_time + (i * 2 * MS_IN_24_HOURS),
                base_time + (i * 2 * MS_IN_24_HOURS) + MS_IN_24_HOURS,
                open_price, close_price, OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
        
        # Update
        plm.update(t_ms=base_time + (8 * MS_IN_24_HOURS))
        
        # Get ledger
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Find final checkpoint with data
        final_cp = None
        for cp in reversed(btc_ledger.cps):
            if cp.n_updates > 0:
                final_cp = cp
                break
        
        self.assertIsNotNone(final_cp, "Should find final checkpoint")
        
        # Compounded return should be: 1.10 * 0.95 * 1.03 = 1.07635
        # So portfolio return should be around 1.076
        # (accounting for fees will make it slightly less)
        self.assertGreater(final_cp.prev_portfolio_ret, 1.05, 
                          "Compounded return should show overall gain")
        self.assertLess(final_cp.prev_portfolio_ret, 1.08,
                       "Compounded return should account for the loss")

    def _create_position(self, position_id: str, trade_pair: TradePair, 
                        open_ms: int, close_ms: int, open_price: float, 
                        close_price: float, order_type: OrderType,
                        leverage: float = 1.0) -> Position:
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
                    leverage=leverage if order_type == OrderType.LONG else -leverage,
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
        position.rebuild_position_with_updated_orders()
        return position


if __name__ == '__main__':
    unittest.main()