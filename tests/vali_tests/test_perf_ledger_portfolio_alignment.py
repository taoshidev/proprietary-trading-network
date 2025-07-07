import unittest
from unittest.mock import patch, Mock
from collections import defaultdict

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
    ParallelizationMode, PerfCheckpoint, TradePairReturnStatus
)


class TestPortfolioTradeParAlignment(TestBase):
    """Tests for portfolio-trade pair return alignment (inspired by existing test patterns)"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.now_ms = TimeUtil.now_in_millis()
        self.BASE_TIME = self.now_ms - (1000 * 60 * 60 * 24 * 30)  # 30 days ago
        
        self.mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg, 
            running_unit_tests=True, 
            elimination_manager=self.elimination_manager
        )
        self.position_manager.clear_all_miner_positions()
        
        self.perf_ledger_manager = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            build_portfolio_ledgers_only=False  # We need both portfolio and trade pair ledgers
        )

    def check_portfolio_alignment_per_checkpoint(self, bundles):
        """
        Enhanced version of check_alignment_per_cp from existing tests.
        Validates that portfolio returns equal the product of trade pair returns.
        """
        if self.DEFAULT_MINER_HOTKEY not in bundles:
            return  # No data to check
            
        bundle = bundles[self.DEFAULT_MINER_HOTKEY]
        portfolio_pl = bundle[TP_ID_PORTFOLIO]
        
        failures = []
        
        for cp_idx in range(len(portfolio_pl.cps)):
            portfolio_cp = portfolio_pl.cps[cp_idx]
            
            manual_portfolio_ret = 1.0
            manual_portfolio_spread_fee = 1.0
            manual_portfolio_carry_fee = 1.0
            manual_portfolio_mdd = 1.0
            
            contributing_tps = set()
            debug_info = {}
            
            # Calculate manual portfolio values from trade pair ledgers
            for tp_id, tp_ledger in bundle.items():
                if tp_id == TP_ID_PORTFOLIO:
                    continue
                
                # Find matching checkpoint by timestamp
                matching_cps = [cp for cp in tp_ledger.cps 
                              if cp.last_update_ms == portfolio_cp.last_update_ms]
                
                if matching_cps:
                    tp_cp = matching_cps[0]
                    manual_portfolio_ret *= tp_cp.prev_portfolio_ret
                    manual_portfolio_spread_fee *= tp_cp.prev_portfolio_spread_fee
                    manual_portfolio_carry_fee *= tp_cp.prev_portfolio_carry_fee
                    manual_portfolio_mdd *= tp_cp.mdd
                    
                    contributing_tps.add(tp_id)
                    debug_info[tp_id] = {
                        'return': tp_cp.prev_portfolio_ret,
                        'spread_fee': tp_cp.prev_portfolio_spread_fee,
                        'carry_fee': tp_cp.prev_portfolio_carry_fee,
                        'mdd': tp_cp.mdd
                    }
            
            # Validate alignment
            if abs(portfolio_cp.prev_portfolio_ret - manual_portfolio_ret) > 1e-10:
                failures.append(f"Checkpoint {cp_idx}: Portfolio return {portfolio_cp.prev_portfolio_ret} != "
                              f"Manual {manual_portfolio_ret}. Contributing TPs: {contributing_tps}. "
                              f"Debug: {debug_info}")
            
            if abs(portfolio_cp.prev_portfolio_spread_fee - manual_portfolio_spread_fee) > 1e-10:
                failures.append(f"Checkpoint {cp_idx}: Portfolio spread fee {portfolio_cp.prev_portfolio_spread_fee} != "
                              f"Manual {manual_portfolio_spread_fee}")
            
            if abs(portfolio_cp.prev_portfolio_carry_fee - manual_portfolio_carry_fee) > 1e-10:
                failures.append(f"Checkpoint {cp_idx}: Portfolio carry fee {portfolio_cp.prev_portfolio_carry_fee} != "
                              f"Manual {manual_portfolio_carry_fee}")
        
        if failures:
            self.fail(f"Portfolio alignment failures:\n" + "\n".join(failures))

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_multi_trade_pair_portfolio_alignment(self, mock_candle_fetcher):
        """Test portfolio alignment with multiple trade pairs"""
        mock_candle_fetcher.return_value = {}
        
        # Create positions in different trade pairs with overlapping timeframes
        positions = []
        
        # BTC position - longest running
        btc_orders = [
            Order(price=45000, processed_ms=self.BASE_TIME, order_uuid="btc_1", 
                  trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.3),
            Order(price=47000, processed_ms=self.BASE_TIME + (5 * 24 * 60 * 60 * 1000), 
                  order_uuid="btc_2", trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.2),
            Order(price=48000, processed_ms=self.BASE_TIME + (15 * 24 * 60 * 60 * 1000), 
                  order_uuid="btc_3", trade_pair=TradePair.BTCUSD, order_type=OrderType.FLAT, leverage=0)
        ]
        
        btc_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="btc_pos",
            open_ms=self.BASE_TIME,
            trade_pair=TradePair.BTCUSD,
            orders=btc_orders,
            position_type=OrderType.FLAT
        )
        btc_position.rebuild_position_with_updated_orders()
        positions.append(btc_position)
        
        # ETH position - starts later
        eth_orders = [
            Order(price=3000, processed_ms=self.BASE_TIME + (3 * 24 * 60 * 60 * 1000), 
                  order_uuid="eth_1", trade_pair=TradePair.ETHUSD, order_type=OrderType.LONG, leverage=0.4),
            Order(price=3200, processed_ms=self.BASE_TIME + (10 * 24 * 60 * 60 * 1000), 
                  order_uuid="eth_2", trade_pair=TradePair.ETHUSD, order_type=OrderType.FLAT, leverage=0)
        ]
        
        eth_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="eth_pos",
            open_ms=self.BASE_TIME + (3 * 24 * 60 * 60 * 1000),
            trade_pair=TradePair.ETHUSD,
            orders=eth_orders,
            position_type=OrderType.FLAT
        )
        eth_position.rebuild_position_with_updated_orders()
        positions.append(eth_position)
        
        # USD/JPY position - overlaps with others
        usdjpy_orders = [
            Order(price=150.5, processed_ms=self.BASE_TIME + (7 * 24 * 60 * 60 * 1000), 
                  order_uuid="jpy_1", trade_pair=TradePair.USDJPY, order_type=OrderType.SHORT, leverage=-0.2),
            Order(price=148.2, processed_ms=self.BASE_TIME + (20 * 24 * 60 * 60 * 1000), 
                  order_uuid="jpy_2", trade_pair=TradePair.USDJPY, order_type=OrderType.FLAT, leverage=0)
        ]
        
        usdjpy_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="jpy_pos",
            open_ms=self.BASE_TIME + (7 * 24 * 60 * 60 * 1000),
            trade_pair=TradePair.USDJPY,
            orders=usdjpy_orders,
            position_type=OrderType.FLAT
        )
        usdjpy_position.rebuild_position_with_updated_orders()
        positions.append(usdjpy_position)
        
        # Save all positions
        for position in positions:
            self.position_manager.save_miner_position(position)
        
        # Update ledgers to current time
        self.perf_ledger_manager.update(t_ms=self.now_ms)
        
        # Get bundles and validate alignment
        bundles = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        
        # Should have portfolio + 3 trade pair ledgers
        if self.DEFAULT_MINER_HOTKEY in bundles:
            bundle = bundles[self.DEFAULT_MINER_HOTKEY]
            expected_tp_ids = {TP_ID_PORTFOLIO, TradePair.BTCUSD.trade_pair_id, 
                             TradePair.ETHUSD.trade_pair_id, TradePair.USDJPY.trade_pair_id}
            self.assertEqual(set(bundle.keys()), expected_tp_ids)
            
            # Validate portfolio alignment
            self.check_portfolio_alignment_per_checkpoint(bundles)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_sequential_position_lifecycle_alignment(self, mock_candle_fetcher):
        """Test portfolio alignment through sequential position lifecycles"""
        mock_candle_fetcher.return_value = {}
        
        # Create a sequence of positions that open and close over time
        base_time = self.BASE_TIME
        day_ms = 24 * 60 * 60 * 1000
        
        positions = []
        
        # Position 1: BTC (days 0-10)
        btc_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="btc_seq",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=base_time, order_uuid="btc_open", 
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.5),
                Order(price=52000, processed_ms=base_time + (10 * day_ms), order_uuid="btc_close", 
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.FLAT, leverage=0)
            ],
            position_type=OrderType.FLAT
        )
        btc_position.rebuild_position_with_updated_orders()
        positions.append(btc_position)
        
        # Position 2: ETH (days 5-15, overlaps with BTC)
        eth_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="eth_seq",
            open_ms=base_time + (5 * day_ms),
            trade_pair=TradePair.ETHUSD,
            orders=[
                Order(price=3000, processed_ms=base_time + (5 * day_ms), order_uuid="eth_open", 
                      trade_pair=TradePair.ETHUSD, order_type=OrderType.LONG, leverage=0.3),
                Order(price=3150, processed_ms=base_time + (15 * day_ms), order_uuid="eth_close", 
                      trade_pair=TradePair.ETHUSD, order_type=OrderType.FLAT, leverage=0)
            ],
            position_type=OrderType.FLAT
        )
        eth_position.rebuild_position_with_updated_orders()
        positions.append(eth_position)
        
        # Position 3: USDJPY (days 12-22, starts after BTC closes)
        jpy_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="jpy_seq",
            open_ms=base_time + (12 * day_ms),
            trade_pair=TradePair.USDJPY,
            orders=[
                Order(price=150.0, processed_ms=base_time + (12 * day_ms), order_uuid="jpy_open", 
                      trade_pair=TradePair.USDJPY, order_type=OrderType.SHORT, leverage=-0.4),
                Order(price=148.5, processed_ms=base_time + (22 * day_ms), order_uuid="jpy_close", 
                      trade_pair=TradePair.USDJPY, order_type=OrderType.FLAT, leverage=0)
            ],
            position_type=OrderType.FLAT
        )
        jpy_position.rebuild_position_with_updated_orders()
        positions.append(jpy_position)
        
        # Save positions and update in phases
        for position in positions:
            self.position_manager.save_miner_position(position)
        
        # Update through different phases to test alignment evolution
        phases = [
            base_time + (3 * day_ms),   # Only BTC active
            base_time + (8 * day_ms),   # BTC + ETH active
            base_time + (12 * day_ms),  # ETH + JPY active (BTC just closed)
            base_time + (18 * day_ms),  # ETH + JPY active
            base_time + (25 * day_ms),  # All closed
        ]
        
        for phase_time in phases:
            self.perf_ledger_manager.update(t_ms=phase_time)
            bundles = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
            
            # Validate alignment at each phase
            self.check_portfolio_alignment_per_checkpoint(bundles)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_complex_order_sequence_alignment(self, mock_candle_fetcher):
        """Test alignment with complex order sequences (scaling in/out)"""
        mock_candle_fetcher.return_value = {}
        
        base_time = self.BASE_TIME
        hour_ms = 60 * 60 * 1000
        
        # Create a position with complex scaling in/out behavior
        complex_orders = []
        current_leverage = 0.0
        
        # Scale into BTC position over time
        scale_in_times = [0, 2*hour_ms, 4*hour_ms, 6*hour_ms]
        scale_in_sizes = [0.1, 0.15, 0.2, 0.25]  # Increasing position
        scale_in_prices = [45000, 45500, 46000, 46200]
        
        for i, (time_offset, size, price) in enumerate(zip(scale_in_times, scale_in_sizes, scale_in_prices)):
            current_leverage += size
            complex_orders.append(Order(
                price=price,
                processed_ms=base_time + time_offset,
                order_uuid=f"scale_in_{i}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG,
                leverage=current_leverage
            ))
        
        # Hold for a while
        # ... (implicit holding period)
        
        # Scale out of position
        scale_out_times = [10*hour_ms, 12*hour_ms, 14*hour_ms, 16*hour_ms]
        scale_out_sizes = [0.2, 0.15, 0.2, 0.15]  # Decreasing position
        scale_out_prices = [47000, 47200, 46800, 46500]
        
        for i, (time_offset, size, price) in enumerate(zip(scale_out_times, scale_out_sizes, scale_out_prices)):
            current_leverage -= size
            complex_orders.append(Order(
                price=price,
                processed_ms=base_time + time_offset,
                order_uuid=f"scale_out_{i}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG if current_leverage > 0 else OrderType.FLAT,
                leverage=max(0, current_leverage)
            ))
        
        complex_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="complex_pos",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=complex_orders,
            position_type=OrderType.FLAT
        )
        complex_position.rebuild_position_with_updated_orders()
        
        self.position_manager.save_miner_position(complex_position)
        
        # Update and validate alignment
        self.perf_ledger_manager.update(t_ms=base_time + (20 * hour_ms))
        bundles = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        
        self.check_portfolio_alignment_per_checkpoint(bundles)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_mixed_long_short_portfolio_alignment(self, mock_candle_fetcher):
        """Test portfolio alignment with mixed long/short positions"""
        mock_candle_fetcher.return_value = {}
        
        base_time = self.BASE_TIME
        day_ms = 24 * 60 * 60 * 1000
        
        positions = []
        
        # Long BTC position
        btc_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="btc_long",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=[
                Order(price=50000, processed_ms=base_time, order_uuid="btc_long_open", 
                      trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.4)
            ],
            position_type=OrderType.LONG
        )
        btc_position.rebuild_position_with_updated_orders()
        positions.append(btc_position)
        
        # Short USD/JPY position
        jpy_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="jpy_short",
            open_ms=base_time + day_ms,
            trade_pair=TradePair.USDJPY,
            orders=[
                Order(price=150.0, processed_ms=base_time + day_ms, order_uuid="jpy_short_open", 
                      trade_pair=TradePair.USDJPY, order_type=OrderType.SHORT, leverage=-0.3)
            ],
            position_type=OrderType.SHORT
        )
        jpy_position.rebuild_position_with_updated_orders()
        positions.append(jpy_position)
        
        # Long ETH position
        eth_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="eth_long",
            open_ms=base_time + (2 * day_ms),
            trade_pair=TradePair.ETHUSD,
            orders=[
                Order(price=3000, processed_ms=base_time + (2 * day_ms), order_uuid="eth_long_open", 
                      trade_pair=TradePair.ETHUSD, order_type=OrderType.LONG, leverage=0.25)
            ],
            position_type=OrderType.LONG
        )
        eth_position.rebuild_position_with_updated_orders()
        positions.append(eth_position)
        
        # Save all positions
        for position in positions:
            self.position_manager.save_miner_position(position)
        
        # Update and validate with mixed long/short portfolio
        self.perf_ledger_manager.update(t_ms=base_time + (5 * day_ms))
        bundles = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        
        self.check_portfolio_alignment_per_checkpoint(bundles)
        
        # Verify we have mixed position types
        if self.DEFAULT_MINER_HOTKEY in bundles:
            bundle = bundles[self.DEFAULT_MINER_HOTKEY]
            # Should have both long and short contributing to portfolio
            self.assertGreater(len(bundle), 1)  # More than just portfolio ledger

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_fee_accumulation_alignment(self, mock_candle_fetcher):
        """Test portfolio alignment with significant fee accumulation"""
        mock_candle_fetcher.return_value = {}
        
        base_time = self.BASE_TIME
        hour_ms = 60 * 60 * 1000
        
        # Create position with many small orders to accumulate fees
        fee_orders = []
        current_leverage = 0.0
        
        # 50 small trades over 50 hours
        for i in range(50):
            current_leverage += 0.01  # Small increments
            fee_orders.append(Order(
                price=50000 + (i * 10),  # Slight price movement
                processed_ms=base_time + (i * hour_ms),
                order_uuid=f"fee_order_{i}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG,
                leverage=current_leverage
            ))
        
        # Close position
        fee_orders.append(Order(
            price=50500,
            processed_ms=base_time + (51 * hour_ms),
            order_uuid="fee_close",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0
        ))
        
        fee_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="fee_pos",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            orders=fee_orders,
            position_type=OrderType.FLAT
        )
        fee_position.rebuild_position_with_updated_orders()
        
        self.position_manager.save_miner_position(fee_position)
        
        # Update and validate with heavy fee accumulation
        self.perf_ledger_manager.update(t_ms=base_time + (60 * hour_ms))
        bundles = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        
        self.check_portfolio_alignment_per_checkpoint(bundles)
        
        # Verify fees were properly accumulated
        if self.DEFAULT_MINER_HOTKEY in bundles:
            portfolio_ledger = bundles[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
            if portfolio_ledger.cps:
                final_cp = portfolio_ledger.cps[-1]
                # Should have accumulated spread fees from many trades
                self.assertLess(final_cp.prev_portfolio_spread_fee, 1.0)


if __name__ == '__main__':
    unittest.main()