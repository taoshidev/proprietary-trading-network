"""
Simple test to verify elimination metrics tracking functionality.
Tests the BacktestMetrics class in isolation.
"""

import unittest
from datetime import datetime, timedelta
from tests.vali_tests.backtest_metrics import BacktestMetrics


class TestSimpleEliminationMetrics(unittest.TestCase):
    """Test BacktestMetrics functionality in isolation"""
    
    def setUp(self):
        self.metrics = BacktestMetrics()
        self.base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    def test_basic_metric_tracking(self):
        """Test basic metric tracking functionality"""
        # Add DB elimination
        self.metrics.add_db_elimination(
            self.base_time - timedelta(days=1),
            "miner_1",
            "DRAWDOWN"
        )
        
        # Add generated elimination  
        self.metrics.add_generated_elimination(
            self.base_time + timedelta(hours=1),
            "miner_2",
            "ZOMBIE"
        )
        
        # Add duplicate attempt
        self.metrics.add_duplicate_attempt("miner_2", "PLAGIARISM")
        
        # Verify counts
        self.assertEqual(self.metrics.n_db_elims, 1)
        self.assertEqual(self.metrics.n_generated_elims, 1)
        self.assertEqual(self.metrics.n_duplicate_attempts, 1)
        
        # Verify timeline
        self.assertEqual(len(self.metrics.elimination_timeline), 2)
        
        # Verify by reason
        self.assertEqual(self.metrics.elimination_by_reason["DRAWDOWN"], 1)
        self.assertEqual(self.metrics.elimination_by_reason["ZOMBIE"], 1)
        
        # Validate consistency
        self.metrics.validate()
    
    def test_conflict_tracking(self):
        """Test conflict resolution tracking"""
        hotkey = "conflict_miner"
        reasons = [
            ("DRAWDOWN", self.base_time),
            ("PLAGIARISM", self.base_time + timedelta(seconds=1))
        ]
        
        # Add conflict
        self.metrics.add_conflict(hotkey, reasons)
        
        # Verify
        self.assertEqual(self.metrics.n_conflicts_resolved, 1)
        self.assertEqual(len(self.metrics.elimination_conflicts), 1)
        
        conflict = self.metrics.elimination_conflicts[0]
        self.assertEqual(conflict['hotkey'], hotkey)
        self.assertEqual(conflict['winner'], 'DRAWDOWN')  # First one wins
    
    def test_metrics_validation(self):
        """Test metrics validation catches inconsistencies"""
        # Add some eliminations
        self.metrics.add_db_elimination(self.base_time, "miner_1", "DRAWDOWN")
        self.metrics.add_generated_elimination(self.base_time, "miner_2", "ZOMBIE")
        
        # Should validate successfully
        self.metrics.validate()
        
        # Manually corrupt the data
        self.metrics.n_db_elims = 10  # Wrong count
        
        # Should raise assertion error
        with self.assertRaises(AssertionError):
            self.metrics.validate()
    
    def test_chronological_timeline(self):
        """Test timeline maintains chronological order"""
        # Add eliminations out of order
        self.metrics.add_generated_elimination(
            self.base_time + timedelta(hours=2),
            "miner_3",
            "PLAGIARISM"
        )
        self.metrics.add_generated_elimination(
            self.base_time,
            "miner_1", 
            "DRAWDOWN"
        )
        self.metrics.add_generated_elimination(
            self.base_time + timedelta(hours=1),
            "miner_2",
            "ZOMBIE"
        )
        
        # Sort timeline
        self.metrics.elimination_timeline.sort(key=lambda x: x[0])
        
        # Verify order
        self.assertEqual(self.metrics.elimination_timeline[0][1], "miner_1")
        self.assertEqual(self.metrics.elimination_timeline[1][1], "miner_2")
        self.assertEqual(self.metrics.elimination_timeline[2][1], "miner_3")
        
        # Should still validate
        self.metrics.validate()


if __name__ == '__main__':
    unittest.main()