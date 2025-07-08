"""
CRITICAL BOUNDARY BUG TESTS - Performance Ledger Endpoint Behavior

PROBLEM STATEMENT:
Setting end_time to daily boundary (t) vs t+1 produces DIFFERENT RETURNS at the boundary checkpoint.
This should NOT happen - the boundary checkpoint should have identical values regardless of end_time.

ROOT CAUSE:
When end_time = boundary (t):     Updates existing checkpoint with current portfolio value
When end_time = boundary + 1:     Creates void checkpoint with old value, new checkpoint with current value
Result: Same boundary checkpoint has different portfolio return values!

TESTS VERIFY:
1. Boundary consistency with and without void filling
2. Checkpoint timing correctness 
3. Return calculation mathematical accuracy
4. 12-hour boundary alignment

All price fetching is mocked for deterministic results.
"""

import copy
import math
import unittest
from unittest.mock import Mock, patch
from collections import defaultdict

import bittensor as bt
from vali_objects.position import Position
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint, TradePairReturnStatus
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair
from time_util.time_util import TimeUtil


class TestPerfLedgerEndpointBehavior(unittest.TestCase):
    """
    CRITICAL TESTS for Performance Ledger Boundary Bug
    
    These tests verify that checkpoint returns are consistent regardless of exact end_time values.
    The bug manifests when the same boundary checkpoint has different return values depending
    on whether end_time is exactly at a boundary or one millisecond past it.
    """
    
    def setUp(self):
        """Set up test fixtures with realistic trading scenarios."""
        bt.logging.set_trace(False)
        
        # Test miner and trading pair
        self.miner_hotkey = "test_miner_boundary_bug"
        self.trade_pair = TradePair.BTCUSD
        
        # 12-hour checkpoint duration (production setting)
        self.target_cp_duration_ms = 43200000  # 12 hours = 43,200,000 milliseconds
        
        # CRITICAL: Test times must be aligned to 12-hour boundaries (00:00:00 and 12:00:00 UTC)
        # January 1, 2025 00:00:00 UTC = 1735689600000 ms (this is a 12-hour boundary)
        self.base_time_ms = 1735689600000  # 2025-01-01 00:00:00 UTC
        self.boundary_time_ms = self.base_time_ms + self.target_cp_duration_ms  # 2025-01-01 12:00:00 UTC
        self.post_boundary_ms = self.boundary_time_ms + 1  # 2025-01-01 12:00:00.001 UTC
        
        # Portfolio values for testing (realistic % gains)
        self.initial_portfolio_value = 1.0      # Starting value
        self.mid_portfolio_value = 1.05         # 5% gain during trading
        self.final_portfolio_value = 1.10       # 10% total gain
        
        # Fee values (standard defaults)
        self.spread_fee = 1.0
        self.carry_fee = 1.0
        
        print(f"\n" + "="*80)
        print(f"TEST SETUP:")
        print(f"  Base time:     {TimeUtil.millis_to_formatted_date_str(self.base_time_ms)}")
        print(f"  Boundary time: {TimeUtil.millis_to_formatted_date_str(self.boundary_time_ms)}")
        print(f"  Post boundary: {TimeUtil.millis_to_formatted_date_str(self.post_boundary_ms)}")
        print(f"  Checkpoint duration: {self.target_cp_duration_ms} ms ({self.target_cp_duration_ms / 3600000} hours)")
        print(f"="*80)
    
    def create_perf_ledger(self, start_time_ms):
        """Create a performance ledger with proper initialization."""
        print(f"  📊 Creating PerfLedger starting at {TimeUtil.millis_to_formatted_date_str(start_time_ms)}")
        return PerfLedger(
            initialization_time_ms=start_time_ms,
            target_cp_duration_ms=self.target_cp_duration_ms,
            max_return=1.0
        )
    
    def initialize_ledger(self, ledger, start_time_ms):
        """Initialize ledger with first order."""
        order_time = start_time_ms + 1000  # 1 second after start
        print(f"  🚀 Initializing ledger with first order at {TimeUtil.millis_to_formatted_date_str(order_time)}")
        print(f"      Initial portfolio value: {self.initial_portfolio_value}")
        
        ledger.init_with_first_order(
            order_processed_ms=order_time,
            point_in_time_dd=0.0,  # No drawdown initially
            current_portfolio_value=self.initial_portfolio_value,
            current_portfolio_fee_spread=self.spread_fee,
            current_portfolio_carry=self.carry_fee
        )
        return order_time
    
    def update_ledger(self, ledger, portfolio_value, time_ms, description, has_open_positions=False):
        """Update ledger with clear logging."""
        status = TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE if has_open_positions else TradePairReturnStatus.TP_NO_OPEN_POSITIONS
        
        print(f"  📈 {description}")
        print(f"      Time: {TimeUtil.millis_to_formatted_date_str(time_ms)}")
        print(f"      Portfolio value: {portfolio_value}")
        print(f"      Status: {status.name}")
        
        checkpoint = ledger.update_pl(
            current_portfolio_value=portfolio_value,
            now_ms=time_ms,
            miner_hotkey=self.miner_hotkey,
            any_open=status,
            current_portfolio_fee_spread=self.spread_fee,
            current_portfolio_carry=self.carry_fee
        )
        
        print(f"      ✅ Update complete. Total checkpoints: {len(ledger.cps)}")
        return checkpoint
    
    def print_ledger_state(self, ledger, title):
        """Print detailed ledger state for debugging."""
        print(f"\n  📋 {title}")
        print(f"      Total checkpoints: {len(ledger.cps)}")
        
        for i, cp in enumerate(ledger.cps):
            time_str = TimeUtil.millis_to_formatted_date_str(cp.last_update_ms)
            print(f"      CP{i}: time={time_str}, accum_ms={cp.accum_ms}, prev_ret={cp.prev_portfolio_ret:.6f}")
            print(f"            gain={cp.gain:.6f}, loss={cp.loss:.6f}, net={cp.gain + cp.loss:.6f}")
    
    def test_critical_boundary_bug_demonstration(self):
        """
        🚨 CRITICAL TEST: Demonstrates the exact boundary bug
        
        This test shows how the SAME boundary checkpoint gets DIFFERENT return values
        depending on whether end_time is exactly at boundary (t) or boundary+1 (t+1).
        
        EXPECTED: Both scenarios should produce identical boundary checkpoint values
        ACTUAL BUG: Different return values for the same checkpoint
        """
        print(f"\n{'='*100}")
        print(f"🚨 CRITICAL BOUNDARY BUG DEMONSTRATION")
        print(f"{'='*100}")
        
        # ==========================================
        # SCENARIO 1: end_time exactly at boundary
        # ==========================================
        print(f"\n🔍 SCENARIO 1: Update exactly at boundary time")
        print(f"   This should create 1 checkpoint ending exactly at the boundary")
        
        ledger1 = self.create_perf_ledger(self.base_time_ms)
        self.initialize_ledger(ledger1, self.base_time_ms)
        
        # Single update that ends exactly at the 12-hour boundary
        self.update_ledger(
            ledger1, 
            self.final_portfolio_value, 
            self.boundary_time_ms,
            "Final update exactly at 12-hour boundary"
        )
        
        self.print_ledger_state(ledger1, "SCENARIO 1 FINAL STATE")
        
        # ==========================================
        # SCENARIO 2: end_time at boundary + 1ms  
        # ==========================================
        print(f"\n🔍 SCENARIO 2: Update at boundary + 1 millisecond")
        print(f"   This triggers void filling and creates 2 checkpoints")
        
        ledger2 = self.create_perf_ledger(self.base_time_ms)
        self.initialize_ledger(ledger2, self.base_time_ms)
        
        # Update that goes 1ms past boundary, triggering void filling
        self.update_ledger(
            ledger2, 
            self.final_portfolio_value, 
            self.post_boundary_ms,
            "Final update 1ms past boundary (triggers void filling)"
        )
        
        self.print_ledger_state(ledger2, "SCENARIO 2 FINAL STATE")
        
        # ==========================================
        # BUG ANALYSIS
        # ==========================================
        print(f"\n🔬 BUG ANALYSIS:")
        
        # Find boundary checkpoints
        boundary_cp1 = None
        boundary_cp2 = None
        
        # Scenario 1: should have 1 checkpoint at boundary
        self.assertEqual(len(ledger1.cps), 1, "Scenario 1 should have exactly 1 checkpoint")
        boundary_cp1 = ledger1.cps[0]
        self.assertEqual(boundary_cp1.last_update_ms, self.boundary_time_ms, "Checkpoint should end at boundary")
        
        # Scenario 2: should have 2 checkpoints, first one at boundary  
        self.assertEqual(len(ledger2.cps), 2, "Scenario 2 should have exactly 2 checkpoints after void filling")
        boundary_cp2 = ledger2.cps[1]  # Second checkpoint should be the boundary one
        self.assertEqual(ledger2.cps[0].last_update_ms, self.boundary_time_ms, "First checkpoint should end at boundary")
        
        # THE CRITICAL ASSERTION: boundary checkpoints should be IDENTICAL
        print(f"   Boundary checkpoint 1 return: {boundary_cp1.prev_portfolio_ret:.10f}")
        print(f"   Boundary checkpoint 2 return: {boundary_cp2.prev_portfolio_ret:.10f}")
        print(f"   Difference: {abs(boundary_cp1.prev_portfolio_ret - boundary_cp2.prev_portfolio_ret):.10f}")
        
        if abs(boundary_cp1.prev_portfolio_ret - boundary_cp2.prev_portfolio_ret) > 1e-10:
            print(f"   🚨 BUG DETECTED: Boundary checkpoints have different return values!")
            print(f"   🚨 This means the same checkpoint gives different results based on end_time")
        else:
            print(f"   ✅ SUCCESS: Boundary checkpoints have identical return values")
        
        # This assertion will FAIL before the fix, PASS after the fix
        self.assertAlmostEqual(
            boundary_cp1.prev_portfolio_ret, 
            boundary_cp2.prev_portfolio_ret, 
            places=10,
            msg="🚨 CRITICAL BUG: Boundary checkpoints must have identical return values regardless of end_time!"
        )
    
    def test_void_filling_timing_correctness(self):
        """
        ⏰ Test that void filling creates checkpoints with correct timing
        
        Verifies that when void filling occurs, checkpoints have:
        1. Different last_update_ms values (no duplicate times)
        2. Correct accumulated time durations
        3. Proper boundary alignment
        """
        print(f"\n{'='*100}")
        print(f"⏰ VOID FILLING TIMING CORRECTNESS TEST")
        print(f"{'='*100}")
        
        ledger = self.create_perf_ledger(self.base_time_ms)
        self.initialize_ledger(ledger, self.base_time_ms)
        
        # Update to exactly boundary + 1 to trigger void filling
        print(f"\n🎯 Triggering void filling by updating to boundary + 1ms")
        self.update_ledger(
            ledger,
            self.final_portfolio_value,
            self.post_boundary_ms,
            "Update triggering void filling"
        )
        
        self.print_ledger_state(ledger, "AFTER VOID FILLING")
        
        # Verify timing correctness
        self.assertEqual(len(ledger.cps), 2, "Should have exactly 2 checkpoints after void filling")
        
        cp1, cp2 = ledger.cps[0], ledger.cps[1]
        
        print(f"\n🔍 TIMING ANALYSIS:")
        print(f"   CP1 time: {cp1.last_update_ms} ({TimeUtil.millis_to_formatted_date_str(cp1.last_update_ms)})")
        print(f"   CP2 time: {cp2.last_update_ms} ({TimeUtil.millis_to_formatted_date_str(cp2.last_update_ms)})")
        print(f"   Time difference: {cp2.last_update_ms - cp1.last_update_ms} ms")
        
        # Critical assertions
        self.assertNotEqual(cp1.last_update_ms, cp2.last_update_ms, 
                           "Checkpoints must have different timestamps")
        self.assertEqual(cp1.last_update_ms, self.boundary_time_ms,
                        "First checkpoint should end exactly at boundary")
        self.assertEqual(cp2.last_update_ms, self.post_boundary_ms,
                        "Second checkpoint should end at boundary + 1ms")
        self.assertEqual(cp1.accum_ms, self.target_cp_duration_ms,
                        "First checkpoint should have full 12-hour duration")
        self.assertEqual(cp2.accum_ms, 1,
                        f"Second checkpoint should have 1ms duration but has {cp2.accum_ms}ms")
    
    def test_mathematical_return_consistency(self):
        """
        🧮 Test mathematical consistency of return calculations
        
        Verifies that:
        1. Returns follow log(current/previous) formula
        2. Gains and losses sum correctly
        3. Portfolio values are preserved correctly through checkpoints
        """
        print(f"\n{'='*100}")
        print(f"🧮 MATHEMATICAL RETURN CONSISTENCY TEST")
        print(f"{'='*100}")
        
        ledger = self.create_perf_ledger(self.base_time_ms)
        order_time = self.initialize_ledger(ledger, self.base_time_ms)
        
        # Series of updates with known portfolio values for math verification
        test_scenarios = [
            (1.02, order_time + 3600000,  "2% gain after 1 hour"),
            (1.05, order_time + 7200000,  "5% total gain after 2 hours"),
            (1.03, order_time + 10800000, "3% total gain after 3 hours (some loss)"),
            (1.08, order_time + 14400000, "8% total gain after 4 hours")
        ]
        
        print(f"\n📊 APPLYING TEST SCENARIOS:")
        for portfolio_value, time_ms, description in test_scenarios:
            self.update_ledger(ledger, portfolio_value, time_ms, description, has_open_positions=True)
        
        self.print_ledger_state(ledger, "AFTER ALL UPDATES")
        
        # Mathematical verification
        cp = ledger.cps[0]
        total_return = cp.gain + cp.loss
        expected_return = math.log(test_scenarios[-1][0] / self.initial_portfolio_value)  # log(final/initial)
        
        print(f"\n🔢 MATHEMATICAL VERIFICATION:")
        print(f"   Initial portfolio: {self.initial_portfolio_value}")
        print(f"   Final portfolio: {test_scenarios[-1][0]}")
        print(f"   Expected return: log({test_scenarios[-1][0]}/{self.initial_portfolio_value}) = {expected_return:.10f}")
        print(f"   Actual return: gain({cp.gain:.10f}) + loss({cp.loss:.10f}) = {total_return:.10f}")
        print(f"   Difference: {abs(total_return - expected_return):.10f}")
        
        self.assertAlmostEqual(total_return, expected_return, places=10,
                             msg="Total return must equal log(final/initial) for mathematical consistency")
    
    def test_boundary_alignment_verification(self):
        """
        📐 Test that checkpoints align to 12-hour boundaries
        
        Verifies that completed checkpoints end exactly on:
        - 00:00:00 UTC (midnight)
        - 12:00:00 UTC (noon)
        
        This is critical for consistent performance measurement across miners.
        """
        print(f"\n{'='*100}")
        print(f"📐 BOUNDARY ALIGNMENT VERIFICATION TEST")
        print(f"{'='*100}")
        
        # Start at non-aligned time to test alignment logic
        misaligned_start = self.base_time_ms + 3600000  # 1 hour offset from boundary
        print(f"🎯 Starting at misaligned time: {TimeUtil.millis_to_formatted_date_str(misaligned_start)}")
        print(f"   This tests that checkpoints still align to 12-hour boundaries")
        
        ledger = self.create_perf_ledger(misaligned_start)
        order_time = self.initialize_ledger(ledger, misaligned_start)
        
        # Update long enough to trigger multiple checkpoint completions
        final_time = order_time + (2.5 * self.target_cp_duration_ms)  # 2.5 checkpoint periods
        self.update_ledger(
            ledger,
            self.final_portfolio_value,
            int(final_time),
            "Long update to trigger multiple checkpoint completions"
        )
        
        self.print_ledger_state(ledger, "AFTER MULTIPLE CHECKPOINT PERIODS")
        
        print(f"\n🔍 BOUNDARY ALIGNMENT ANALYSIS:")
        for i, cp in enumerate(ledger.cps[:-1]):  # All except the last (potentially incomplete) checkpoint
            boundary_remainder = cp.last_update_ms % self.target_cp_duration_ms
            is_aligned = boundary_remainder == 0
            time_str = TimeUtil.millis_to_formatted_date_str(cp.last_update_ms)
            
            print(f"   CP{i}: {time_str}, remainder={boundary_remainder}, aligned={is_aligned}")
            
            self.assertEqual(boundary_remainder, 0,
                           msg=f"Checkpoint {i} must end exactly on 12-hour boundary (remainder should be 0)")
    
    def test_comprehensive_bug_scenarios(self):
        """
        🎯 Comprehensive test covering multiple bug scenarios
        
        Tests edge cases and combinations that could reveal boundary bugs:
        1. Multiple updates at exact boundaries
        2. Updates that span multiple checkpoint periods  
        3. Mix of open and closed positions
        4. Various portfolio value changes
        """
        print(f"\n{'='*100}")
        print(f"🎯 COMPREHENSIVE BUG SCENARIOS TEST")
        print(f"{'='*100}")
        
        scenarios = [
            {
                'name': 'Single boundary crossing',
                'updates': [(self.mid_portfolio_value, self.boundary_time_ms + 1)]
            },
            {
                'name': 'Multiple boundary crossings', 
                'updates': [
                    (1.02, self.boundary_time_ms + 1),
                    (1.04, self.boundary_time_ms + self.target_cp_duration_ms + 1),
                    (1.06, self.boundary_time_ms + 2 * self.target_cp_duration_ms + 1)
                ]
            },
            {
                'name': 'Exact boundary timing',
                'updates': [
                    (1.03, self.boundary_time_ms),
                    (1.05, self.boundary_time_ms + self.target_cp_duration_ms)
                ]
            }
        ]
        
        for scenario in scenarios:
            print(f"\n🧪 Testing scenario: {scenario['name']}")
            
            ledger = self.create_perf_ledger(self.base_time_ms)
            self.initialize_ledger(ledger, self.base_time_ms)
            
            for i, (portfolio_value, time_ms) in enumerate(scenario['updates']):
                self.update_ledger(
                    ledger,
                    portfolio_value, 
                    time_ms,
                    f"Update {i+1}: {scenario['name']}"
                )
            
            self.print_ledger_state(ledger, f"SCENARIO: {scenario['name']}")
            
            # Basic consistency checks
            self.assertGreater(len(ledger.cps), 0, "Should have at least one checkpoint")
            
            # Verify no duplicate timestamps
            timestamps = [cp.last_update_ms for cp in ledger.cps]
            unique_timestamps = set(timestamps)
            self.assertEqual(len(timestamps), len(unique_timestamps), 
                           "All checkpoint timestamps must be unique")
            
            # Verify increasing timestamps
            for i in range(1, len(timestamps)):
                self.assertGreater(timestamps[i], timestamps[i-1],
                                 "Checkpoint timestamps must be increasing")


if __name__ == '__main__':
    """
    Run these critical tests to verify boundary bug fixes.
    
    BEFORE FIX: test_critical_boundary_bug_demonstration will FAIL
    AFTER FIX:  All tests should PASS
    
    Run with: python -m pytest tests/vali_tests/test_perf_ledger_endpoint_behavior.py -v -s
    """
    print("🚀 STARTING CRITICAL BOUNDARY BUG TESTS")
    print("="*100)
    print("These tests verify that the performance ledger boundary bug is fixed.")
    print("If test_critical_boundary_bug_demonstration FAILS, the bug still exists.")
    print("="*100)
    
    bt.logging.enable_info()
    unittest.main(verbosity=2)