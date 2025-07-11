"""
Test for perf ledger void filling behavior with flat MDD and zero gain/loss.

This test verifies that when a perf ledger needs to fill a void over a long period,
it maintains:
- Flat MDD (Maximum Drawdown) 
- 0 gain
- 0 loss
- Same return for every checkpoint
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
from collections import defaultdict

from tests.shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    PerfLedger,
    PerfLedgerManager,
    TP_ID_PORTFOLIO,
    ParallelizationMode,
)


class TestPerfLedgerVoidFilling(TestBase):
    """Test cases for perf ledger void filling behavior."""

    def setUp(self):
        super().setUp()
        self.test_hotkey = "test_miner_void_fill"
        self.now_ms = TimeUtil.now_in_millis()
        
        # Create mock metagraph with single miner
        self.mmg = MockMetagraph(hotkeys=[self.test_hotkey])
        self.elimination_manager = EliminationManager(self.mmg, None, None)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
        )
        
        # Clear any existing positions
        self.position_manager.clear_all_miner_positions()

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_long_void_flat_performance(self, mock_lpf):
        """
        Test that filling a long void results in flat performance metrics.
        
        This simulates a scenario where:
        1. A position opens at time T0
        2. No updates occur for a long period (void)
        3. Position closes at time T1 (much later)
        4. Perf ledger must fill the void with checkpoints
        
        Expected behavior:
        - All checkpoints in the void should have same return
        - MDD should remain flat (no drawdown)
        - Gain and loss should be 0
        """
        # Mock price fetcher to avoid live data calls
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        # Create perf ledger manager
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        # Time setup: Create a large void
        # Align times to avoid checkpoint boundary issues
        # Use noon times to ensure we're well within checkpoint boundaries
        now_noon = (self.now_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS + (MS_IN_24_HOURS // 2)
        position_open_ms = now_noon - (30 * MS_IN_24_HOURS)  # 30 days ago at noon
        position_close_ms = now_noon - (5 * MS_IN_24_HOURS)   # 5 days ago at noon
        void_duration_ms = self.now_ms - position_close_ms  # ~5 day void to current
        
        # Create a simple position that opens and closes with no intermediate orders
        # This creates a void that needs to be filled
        open_order = Order(
            price=100.0,
            processed_ms=position_open_ms,
            order_uuid="open_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=1.0,
        )
        
        close_order = Order(
            price=100.0,  # Same price = no gain/loss
            processed_ms=position_close_ms,
            order_uuid="close_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0.0,
        )
        
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid="void_position",
            open_ms=position_open_ms,
            close_ms=position_close_ms,
            trade_pair=TradePair.BTCUSD,
            orders=[open_order, close_order],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        
        # Rebuild position to calculate returns
        position.rebuild_position_with_updated_orders()
        assert position.close_ms == position_close_ms
        assert position.is_closed_position
        
        # Save position
        self.position_manager.save_miner_position(position)
        
        # Build perf ledger up to current time
        # This should create checkpoints to fill the void
        update_time_ms = self.now_ms
        plm.update(t_ms=update_time_ms)
        
        # Get the ledger
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.test_hotkey, bundles)

        for tp_id, ledger in bundles[self.test_hotkey].items():

            # Verify checkpoints were created to fill the void
            elapsed_ms = update_time_ms - position_open_ms
            # Checkpoints are 12 hours, so we expect 2 per day
            checkpoint_duration_ms = 12 * 60 * 60 * 1000  # 12 hours
            n_expected_cps = elapsed_ms // checkpoint_duration_ms + 1
            checkpoints = ledger.cps
            self.assertEqual(len(checkpoints), n_expected_cps, f"Should have created {n_expected_cps} checkpoints (12hr intervals)")

            # Verify flat performance metrics
            
            prev_cp = None
            position_close_cp_idx = None
            
            # Analyze all checkpoints
            for i, cp in enumerate(checkpoints):
                if i == 0:
                    prev_cp = cp
                    continue
                
                # Check if position is still open during this checkpoint
                if cp.last_update_ms <= position_close_ms:
                    # Position is still open - expect losses from fees
                    self.assertLess(cp.mdd, prev_cp.mdd, 
                                  f"Checkpoint {i} MDD should decrease due to fees")
                    self.assertLess(cp.carry_fee_loss, 0.0, 
                                  f"Checkpoint {i} should have carry fee loss")
                    self.assertLess(cp.loss, 0.0, 
                                  f"Checkpoint {i} should have loss from fees")
                    self.assertEqual(cp.gain, 0.0, 
                                   f"Checkpoint {i} should have 0 gain")
                    
                    # Check if position closes in next checkpoint period
                    next_cp_start = cp.last_update_ms
                    next_cp_end = cp.last_update_ms + checkpoint_duration_ms
                    if next_cp_start < position_close_ms <= next_cp_end:
                        position_close_cp_idx = i + 1
                        print(f"\nPosition will close in checkpoint {i+1}")
                        print(f"  Position close: {TimeUtil.millis_to_datetime(position_close_ms)}")
                        print(f"  Next CP period: {TimeUtil.millis_to_datetime(next_cp_start)} to {TimeUtil.millis_to_datetime(next_cp_end)}")
                
                else:
                    # Position has closed - void period
                    if position_close_cp_idx is None:
                        position_close_cp_idx = i
                        print(f"\nPosition closed before or in checkpoint {i}")
                    
                    # All void checkpoints should have flat performance
                    if i > position_close_cp_idx:
                        self.assertEqual(cp.gain, 0.0, 
                                       f"Void checkpoint {i} should have 0 gain")
                        self.assertEqual(cp.loss, 0.0, 
                                       f"Void checkpoint {i} should have 0 loss")
                        self.assertEqual(cp.mdd, prev_cp.mdd, 
                                       f"Void checkpoint {i} MDD should be flat")
                        self.assertEqual(cp.prev_portfolio_ret, prev_cp.prev_portfolio_ret, 
                                       f"Void checkpoint {i} returns should be flat")
                
                prev_cp = cp


        
    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_multiple_void_periods(self, mock_lpf):
        """
        Test handling multiple void periods in the same ledger.
        
        Creates positions with gaps between them to verify:
        - Each void is filled appropriately
        - Performance remains flat during voids
        - Transitions between positions are handled correctly
        """
        # Mock price fetcher
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
        
        # Create multiple positions with voids between them
        base_time = self.now_ms - (60 * MS_IN_24_HOURS)  # 60 days ago
        
        positions = []
        position_periods = []
        
        # Position 1: Days 0-10
        pos1_open = base_time
        pos1_close = base_time + (10 * MS_IN_24_HOURS)
        pos1 = self._create_flat_position("pos1", pos1_open, pos1_close, 100.0)
        positions.append(pos1)
        position_periods.append((pos1_open, pos1_close, "Position 1"))
        
        # Void 1: Days 10-20 (no position)
        void1_start = pos1_close
        void1_end = base_time + (20 * MS_IN_24_HOURS)
        
        # Position 2: Days 20-30
        pos2_open = base_time + (20 * MS_IN_24_HOURS)
        pos2_close = base_time + (30 * MS_IN_24_HOURS)
        pos2 = self._create_flat_position("pos2", pos2_open, pos2_close, 100.0)
        positions.append(pos2)
        position_periods.append((pos2_open, pos2_close, "Position 2"))
        
        # Void 2: Days 30-45 (no position)
        void2_start = pos2_close
        void2_end = base_time + (45 * MS_IN_24_HOURS)
        
        # Position 3: Days 45-50
        pos3_open = base_time + (45 * MS_IN_24_HOURS)
        pos3_close = base_time + (50 * MS_IN_24_HOURS)
        pos3 = self._create_flat_position("pos3", pos3_open, pos3_close, 100.0)
        positions.append(pos3)
        position_periods.append((pos3_open, pos3_close, "Position 3"))
        
        # Void 3: Days 50-now (final void)
        void3_start = pos3_close
        void3_end = self.now_ms
        
        # Save all positions
        for pos in positions:
            self.position_manager.save_miner_position(pos)
        
        # Update ledger to current time
        update_time_ms = self.now_ms
        plm.update(t_ms=update_time_ms)
        
        # Get ledger bundles
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.test_hotkey, bundles)
        
        # Check each trade pair ledger
        for tp_id, ledger in bundles[self.test_hotkey].items():
            checkpoints = ledger.cps
            
            # Verify total checkpoint count
            elapsed_ms = update_time_ms - base_time
            n_expected_cps = elapsed_ms // ledger.target_cp_duration_ms + 1
            self.assertEqual(len(checkpoints), n_expected_cps, f"Should have {n_expected_cps} checkpoints for {tp_id}")
            
            # Analyze checkpoints for each position period
            for pos_open, pos_close, pos_name in position_periods:
                # Find checkpoints during position lifetime
                position_cps = []
                close_cp_idx = None
                
                for i, cp in enumerate(checkpoints):
                    if pos_open <= cp.last_update_ms <= pos_close:
                        position_cps.append((i, cp))
                    # Find where position closes
                    if i > 0 and cp.n_updates == 0:
                        prev_cp = checkpoints[i-1]
                        if prev_cp.lowerbound_time_created_ms <= pos_close <= prev_cp.last_update_ms:
                            close_cp_idx = i
                
                # Verify behavior during position lifetime
                prev_cp = None
                for idx, (i, cp) in enumerate(position_cps):
                    if idx == 0:
                        # First checkpoint of position
                        prev_cp = cp
                    else:
                        # During position: should have carry fee losses
                        self.assertLess(cp.mdd, prev_cp.mdd, 
                                      f"{pos_name} CP {i}: MDD should decrease due to fees")
                        self.assertLess(cp.carry_fee_loss, 0.0, 
                                      f"{pos_name} CP {i}: Should have carry fee loss")
                        self.assertLess(cp.loss, 0.0, 
                                      f"{pos_name} CP {i}: Should have loss from fees")
                        self.assertEqual(cp.gain, 0.0, 
                                       f"{pos_name} CP {i}: Should have 0 gain")
                        prev_cp = cp
            
            # Verify void periods (checkpoints after each position closes)
            # Void 1: After position 1
            void1_cps = [cp for cp in checkpoints 
                        if void1_start < cp.last_update_ms <= void1_end]
            if void1_cps:
                print(f"\nVoid 1 checkpoints ({len(void1_cps)} total):")
                for i, cp in enumerate(void1_cps[:3]):  # Show first 3
                    print(f"  CP {i}: loss={cp.loss:.6f}, gain={cp.gain:.6f}, "
                          f"mdd={cp.mdd:.6f}, n_updates={cp.n_updates}")
            self._verify_void_checkpoints(void1_cps, "Void 1")
            
            # Void 2: After position 2  
            void2_cps = [cp for cp in checkpoints
                        if void2_start < cp.last_update_ms <= void2_end]
            self._verify_void_checkpoints(void2_cps, "Void 2")
            
            # Void 3: After position 3 to now
            void3_cps = [cp for cp in checkpoints
                        if void3_start < cp.last_update_ms <= void3_end]
            self._verify_void_checkpoints(void3_cps, "Void 3")
        
    def _create_flat_position(self, position_id: str, open_ms: int, close_ms: int, price: float) -> Position:
        """Helper to create a position with no gain/loss."""
        open_order = Order(
            price=price,
            processed_ms=open_ms,
            order_uuid=f"{position_id}_open",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=1.0,
        )
        
        close_order = Order(
            price=price,  # Same price
            processed_ms=close_ms,
            order_uuid=f"{position_id}_close",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0.0,
        )
        
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid=position_id,
            open_ms=open_ms,
            close_ms=close_ms,
            trade_pair=TradePair.BTCUSD,
            orders=[open_order, close_order],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        
        position.rebuild_position_with_updated_orders()
        return position
        
    def _verify_void_checkpoints(self, checkpoints: list, void_name: str):
        """Helper to verify checkpoints in void periods show flat performance."""
        self.assertGreater(len(checkpoints), 0, f"{void_name} should have checkpoints")
        
        # The first checkpoint might have residual losses from position closing
        # So we check from the second checkpoint onwards for true void behavior
        stabilized_cps = checkpoints[1:] if len(checkpoints) > 1 else checkpoints
        
        if stabilized_cps:
            # Get reference values from first stabilized checkpoint
            ref_mdd = stabilized_cps[0].mdd
            ref_return = stabilized_cps[0].prev_portfolio_ret
            
            for i, cp in enumerate(stabilized_cps):
                # Void checkpoints after the first should have no changes
                self.assertEqual(cp.gain, 0.0, 
                               f"{void_name} stabilized checkpoint {i} should have 0 gain")
                self.assertEqual(cp.loss, 0.0,
                               f"{void_name} stabilized checkpoint {i} should have 0 loss")
                
                # MDD should remain constant during void
                self.assertEqual(cp.mdd, ref_mdd,
                               f"{void_name} stabilized checkpoint {i} MDD should be flat")
                
                # Returns should remain constant during void
                self.assertEqual(cp.prev_portfolio_ret, ref_return,
                               f"{void_name} stabilized checkpoint {i} returns should be flat")
        
        # First checkpoint might have residual effects, just verify it has no gains
        if checkpoints:
            self.assertEqual(checkpoints[0].gain, 0.0,
                           f"{void_name} first checkpoint should have 0 gain")


if __name__ == '__main__':
    unittest.main()