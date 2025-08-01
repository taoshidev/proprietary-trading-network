"""
Core test to verify elimination behavior during backtesting.
Focuses on the essential requirements:
1. Pre-eliminated miners have weight 0 throughout
2. Perf ledger stops updating after elimination
3. Weights always sum to 1.0
4. Metrics track eliminations correctly
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from tests.vali_tests.backtest_metrics import BacktestMetrics


class BacktestSimulator:
    """Simplified backtest simulator for testing elimination behavior"""
    
    def __init__(self, all_miners: List[str], base_time: datetime):
        self.all_miners = all_miners
        self.base_time = base_time
        self.current_time = base_time
        
        # Track eliminations
        self.eliminated_miners: Dict[str, Tuple[str, datetime]] = {}
        
        # Track weights
        self.weights: Dict[str, float] = {}
        
        # Track perf ledger updates
        self.perf_updates: Dict[str, List[datetime]] = defaultdict(list)
        
        # Metrics
        self.metrics = BacktestMetrics()
    
    def add_pre_elimination(self, miner: str, reason: str, timestamp: datetime):
        """Add miner eliminated before backtest"""
        self.eliminated_miners[miner] = (reason, timestamp)
        self.metrics.add_db_elimination(timestamp, miner, reason)
    
    def eliminate_miner(self, miner: str, reason: str):
        """Eliminate miner at current time"""
        if miner in self.eliminated_miners:
            self.metrics.add_duplicate_attempt(miner, reason)
            return False
        
        self.eliminated_miners[miner] = (reason, self.current_time)
        self.weights[miner] = 0.0
        self.metrics.add_generated_elimination(self.current_time, miner, reason)
        return True
    
    def initialize_weights(self):
        """Initialize weight vector with all miners"""
        active_count = len([m for m in self.all_miners if m not in self.eliminated_miners])
        
        for miner in self.all_miners:
            if miner in self.eliminated_miners:
                self.weights[miner] = 0.0
            else:
                self.weights[miner] = 1.0 / active_count if active_count > 0 else 0.0
    
    def normalize_weights(self):
        """Normalize weights maintaining eliminated at 0"""
        active_sum = sum(w for m, w in self.weights.items() if m not in self.eliminated_miners)
        
        if active_sum > 0:
            for miner in self.all_miners:
                if miner not in self.eliminated_miners:
                    self.weights[miner] = self.weights[miner] / active_sum
    
    def update_perf_ledgers(self):
        """Update perf ledgers for non-eliminated miners"""
        for miner in self.all_miners:
            # Check if eliminated
            if miner in self.eliminated_miners:
                elim_reason, elim_time = self.eliminated_miners[miner]
                if self.current_time >= elim_time:
                    continue  # Skip update
            
            # Update allowed
            self.perf_updates[miner].append(self.current_time)
    
    def advance_time(self, hours: int = 1):
        """Advance simulation time"""
        self.current_time += timedelta(hours=hours)
    
    def get_active_miners(self) -> Set[str]:
        """Get currently active miners"""
        return set(self.all_miners) - set(self.eliminated_miners.keys())
    
    def verify_invariants(self) -> Tuple[bool, str]:
        """Verify all invariants hold"""
        # All miners in weights
        if set(self.weights.keys()) != set(self.all_miners):
            return False, "Not all miners in weight vector"
        
        # Eliminated have 0 weight
        for miner in self.eliminated_miners:
            if self.weights[miner] != 0.0:
                return False, f"Eliminated miner {miner} has non-zero weight"
        
        # Weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-10:
            return False, f"Weights sum to {total}, not 1.0"
        
        return True, "All invariants satisfied"


class TestBacktestEliminationCore(unittest.TestCase):
    """Core tests for elimination behavior during backtesting"""
    
    def setUp(self):
        self.base_time = datetime(2024, 1, 1, 10, 0, 0)
        self.all_miners = [f"miner_{i}" for i in range(10)]
    
    def test_complete_elimination_flow(self):
        """Test complete elimination flow through backtest"""
        sim = BacktestSimulator(self.all_miners, self.base_time)
        
        # Add pre-eliminations
        sim.add_pre_elimination("miner_0", "DRAWDOWN", self.base_time - timedelta(days=1))
        sim.add_pre_elimination("miner_1", "ZOMBIE", self.base_time - timedelta(days=2))
        
        # Initialize
        sim.initialize_weights()
        
        # Verify initial state
        self.assertEqual(len(sim.eliminated_miners), 2)
        self.assertEqual(sim.weights["miner_0"], 0.0)
        self.assertEqual(sim.weights["miner_1"], 0.0)
        
        # Run 24-hour backtest
        elimination_schedule = [
            (5, "miner_5", "DRAWDOWN"),
            (10, "miner_6", "PLAGIARISM"),
            (15, "miner_7", "ZOMBIE")
        ]
        
        for hour in range(24):
            # Check for scheduled eliminations
            for elim_hour, miner, reason in elimination_schedule:
                if hour == elim_hour:
                    sim.eliminate_miner(miner, reason)
            
            # Update systems
            sim.update_perf_ledgers()
            sim.normalize_weights()
            
            # Verify invariants
            valid, msg = sim.verify_invariants()
            self.assertTrue(valid, f"Hour {hour}: {msg}")
            
            # Advance time
            sim.advance_time(1)
        
        # Verify final state
        self.assertEqual(len(sim.eliminated_miners), 5)
        self.assertEqual(sim.metrics.n_db_elims, 2)
        self.assertEqual(sim.metrics.n_generated_elims, 3)
        
        # Verify perf updates stopped at elimination
        self.assertEqual(len(sim.perf_updates["miner_0"]), 0)  # Pre-eliminated
        self.assertEqual(len(sim.perf_updates["miner_1"]), 0)  # Pre-eliminated  
        self.assertEqual(len(sim.perf_updates["miner_5"]), 5)  # Eliminated hour 5
        self.assertEqual(len(sim.perf_updates["miner_6"]), 10) # Eliminated hour 10
        self.assertEqual(len(sim.perf_updates["miner_7"]), 15) # Eliminated hour 15
        
        # Active miners updated all 24 hours
        for i in [2, 3, 4, 8, 9]:
            self.assertEqual(len(sim.perf_updates[f"miner_{i}"]), 24)
    
    def test_weight_normalization_maintained(self):
        """Test weights always sum to 1.0"""
        sim = BacktestSimulator(self.all_miners, self.base_time)
        
        # Various elimination patterns to test
        test_cases = [
            ("no_eliminations", []),
            ("single", [("miner_3", "DRAWDOWN")]),
            ("multiple", [("miner_1", "ZOMBIE"), ("miner_4", "PLAGIARISM"), ("miner_7", "DRAWDOWN")]),
            ("extreme", [(f"miner_{i}", "DRAWDOWN") for i in range(8)])  # 80% eliminated
        ]
        
        for case_name, eliminations in test_cases:
            with self.subTest(case=case_name):
                # Reset
                sim = BacktestSimulator(self.all_miners, self.base_time)
                sim.initialize_weights()
                
                # Apply eliminations
                for miner, reason in eliminations:
                    sim.eliminate_miner(miner, reason)
                
                # Normalize
                sim.normalize_weights()
                
                # Verify sum
                total = sum(sim.weights.values())
                self.assertAlmostEqual(total, 1.0, places=10,
                                     msg=f"{case_name}: weights sum to {total}")
                
                # Verify eliminated have 0
                for miner, _ in eliminations:
                    self.assertEqual(sim.weights[miner], 0.0)
    
    def test_duplicate_elimination_prevention(self):
        """Test that duplicate eliminations are prevented"""
        sim = BacktestSimulator(self.all_miners, self.base_time)
        
        # First elimination
        result1 = sim.eliminate_miner("miner_3", "DRAWDOWN")
        self.assertTrue(result1)
        
        # Attempt duplicates
        result2 = sim.eliminate_miner("miner_3", "DRAWDOWN")
        self.assertFalse(result2)
        
        result3 = sim.eliminate_miner("miner_3", "PLAGIARISM")
        self.assertFalse(result3)
        
        # Verify metrics
        self.assertEqual(sim.metrics.n_generated_elims, 1)
        self.assertEqual(sim.metrics.n_duplicate_attempts, 2)
    
    def test_perf_updates_stop_at_elimination(self):
        """Test perf ledger updates stop exactly at elimination time"""
        sim = BacktestSimulator(self.all_miners, self.base_time)
        sim.initialize_weights()
        
        target_miner = "miner_4"
        elimination_hour = 10
        
        # Run simulation
        for hour in range(20):
            # Eliminate at specified hour
            if hour == elimination_hour:
                sim.eliminate_miner(target_miner, "DRAWDOWN")
            
            # Update perf ledgers
            sim.update_perf_ledgers()
            
            # Advance time
            sim.advance_time(1)
        
        # Verify updates stopped
        updates = sim.perf_updates[target_miner]
        self.assertEqual(len(updates), elimination_hour)
        
        # All updates should be before elimination
        elim_time = sim.base_time + timedelta(hours=elimination_hour)
        for update_time in updates:
            self.assertLess(update_time, elim_time)
    
    def test_conflict_resolution(self):
        """Test earliest elimination wins in conflicts"""
        sim = BacktestSimulator(self.all_miners, self.base_time)
        
        # Multiple reasons detected for same miner
        conflicts = [
            ("PLAGIARISM", sim.current_time + timedelta(seconds=5)),
            ("DRAWDOWN", sim.current_time + timedelta(seconds=2)),
            ("ZOMBIE", sim.current_time + timedelta(seconds=8))
        ]
        
        # Sort by time - earliest wins
        conflicts.sort(key=lambda x: x[1])
        winner_reason, winner_time = conflicts[0]
        
        # Set time to winner time
        sim.current_time = winner_time
        
        # Apply winning elimination
        sim.eliminate_miner("miner_8", winner_reason)
        
        # Track conflict
        sim.metrics.add_conflict("miner_8", [(r, t) for r, t in conflicts])
        
        # Verify
        self.assertEqual(sim.eliminated_miners["miner_8"][0], "DRAWDOWN")
        self.assertEqual(sim.metrics.n_conflicts_resolved, 1)
    
    def test_metrics_validation(self):
        """Test metrics remain consistent"""
        sim = BacktestSimulator(self.all_miners, self.base_time)
        
        # Add various eliminations
        sim.add_pre_elimination("miner_0", "ZOMBIE", self.base_time - timedelta(days=1))
        sim.eliminate_miner("miner_3", "DRAWDOWN")
        sim.eliminate_miner("miner_5", "PLAGIARISM")
        
        # Attempt duplicate
        sim.eliminate_miner("miner_3", "ZOMBIE")
        
        # Validate
        sim.metrics.validate()
        
        # Check specific counts
        self.assertEqual(sim.metrics.n_db_elims, 1)
        self.assertEqual(sim.metrics.n_generated_elims, 2)
        self.assertEqual(sim.metrics.n_duplicate_attempts, 1)
        
        # Check timeline
        self.assertEqual(len(sim.metrics.elimination_timeline), 3)


if __name__ == '__main__':
    unittest.main()