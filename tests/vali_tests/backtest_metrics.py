"""
Backtest metrics tracking utility for elimination testing.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict


@dataclass
class BacktestMetrics:
    """Track elimination metrics during backtest"""
    n_db_elims: int = 0
    n_generated_elims: int = 0
    n_duplicate_attempts: int = 0
    n_conflicts_resolved: int = 0
    elimination_timeline: List[Tuple[datetime, str, str]] = None
    elimination_conflicts: List[Dict] = None
    elimination_by_reason: Dict[str, int] = None
    
    def __init__(self):
        self.elimination_timeline = []
        self.elimination_conflicts = []
        self.elimination_by_reason = defaultdict(int)
    
    def add_db_elimination(self, timestamp: datetime, hotkey: str, reason: str):
        """Add pre-existing elimination from database"""
        self.n_db_elims += 1
        self.elimination_timeline.append((timestamp, hotkey, reason))
        self.elimination_by_reason[reason] += 1
    
    def add_generated_elimination(self, timestamp: datetime, hotkey: str, reason: str):
        """Add new elimination generated during backtest"""
        self.n_generated_elims += 1
        self.elimination_timeline.append((timestamp, hotkey, reason))
        self.elimination_by_reason[reason] += 1
    
    def add_duplicate_attempt(self, hotkey: str, reason: str):
        """Track attempted duplicate elimination"""
        self.n_duplicate_attempts += 1
    
    def add_conflict(self, hotkey: str, reasons: List[Tuple[str, datetime]]):
        """Track conflicting elimination reasons"""
        self.n_conflicts_resolved += 1
        self.elimination_conflicts.append({
            'hotkey': hotkey,
            'reasons': reasons,
            'winner': reasons[0][0]  # Earliest wins
        })
    
    def validate(self):
        """Validate metric consistency"""
        total_eliminations = len(self.elimination_timeline)
        assert total_eliminations == self.n_db_elims + self.n_generated_elims, \
            f"Timeline count {total_eliminations} != db {self.n_db_elims} + generated {self.n_generated_elims}"
        
        reason_total = sum(self.elimination_by_reason.values())
        assert reason_total == total_eliminations, \
            f"Reason counts {reason_total} != total eliminations {total_eliminations}"
        
        # Verify timeline is chronological
        for i in range(1, len(self.elimination_timeline)):
            assert self.elimination_timeline[i-1][0] <= self.elimination_timeline[i][0], \
                "Timeline not chronological"