import copy
import math
import random

import numpy as np

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_config import ValiConfig


class TestMetrics(TestBase):

    def test_return_no_positions(self):
        self.assertEqual(Metrics.base_return([]), 0.0)

    def test_negative_returns(self):
        """Test that the returns scoring function works properly for only negative returns"""
        log_returns = [-0.2, -0.1, -0.3, -0.2, -0.1, -0.3]

        base_return = Metrics.base_return(log_returns)
        self.assertLess(base_return, 0.0)

    def test_positive_returns(self):
        """Test that the returns scoring function works properly for only positive returns"""
        log_returns = [0.2, 0.1, 0.3, 0.2, 0.1, 0.3]

        base_return = Metrics.base_return(log_returns)
        self.assertGreater(base_return, 0.0)

    def test_typical_omega(self):
        """Test that the omega function works as expected for only positive returns"""
        log_returns = [0.003, -0.002] * 50  # Sum = 0.015
        omega = Metrics.omega(log_returns)

        # Should always be greater or equal to 0
        self.assertGreater(omega, 0.0)

    def test_negative_omega(self):
        """Test that the omega function works as expected for only negative returns"""
        log_returns = [-0.003, -0.001] * 50  # Sum = -0.06
        omega = Metrics.omega(log_returns)

        # Should always be less or equal to 0
        self.assertEqual(omega, 0.0)

    def test_positive_omega(self):
        """Test that the omega function works as expected for only positive returns - the default loss works"""
        log_returns = [0.002, 0.001] * 50  # Sum = 0.045
        omega = Metrics.omega(log_returns)

        # Should always be greater or equal to 0, cannot be massive
        self.assertGreater(omega, 0.0)
        self.assertLess(omega, np.inf)

    def test_positive_omega_small_loss(self):
        """Test that the omega function works as expected for only positive returns - the default loss works"""
        log_returns = [0.002, -0.001, 0.001] * 50  # Sum = 0.01
        omega = Metrics.omega(log_returns)

        # Should always be greater or equal to 0, cannot be massive
        self.assertGreater(omega, 0.0)
        self.assertLess(omega, np.inf)

    def test_omega_no_returns(self):
        """Test that the omega function works as expected for no returns"""
        returns = []

        omega = Metrics.omega(returns)

        # Expected value is zero
        self.assertEqual(omega, ValiConfig.OMEGA_NOCONFIDENCE_VALUE)
    def test_omega_weighting(self):
        #No positive returns means empty numerator
        returns = [-0.01] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 1)
        omega = Metrics.omega(returns, weighting=True)
        #Numerator is 0 so will be 0
        self.assertEqual(omega, 0)

        #No negative returns means empty denominator
        returns = [0.01] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 1)
        omega = Metrics.omega(returns, weighting=True)
        #Should be small in magnitude, but positive
        self.assertGreater(omega, 0)





    def test_sharpe_positive(self):
        """Test that the sharpe function is positive when all returns are positive"""
        log_returns = [0.002] * 100  # Sum = 0.06
        sharpe = Metrics.sharpe(log_returns)

        # Should always be greater or equal to 0
        self.assertGreater(sharpe, 0.0)

    def test_sharpe_no_returns_no_variance(self):
        """Test that the sharpe function is 0 with 0 returns"""
        log_returns = [0.0] * 100  # Sum = 0.0
        sharpe = Metrics.sharpe(log_returns)

        # Expected value is zero
        self.assertLess(sharpe, 0.0)

    def test_sharpe_no_returns(self):
        """Test that the sharpe function works as expected"""
        returns = []

        sharpe = Metrics.sharpe(returns)

        # Expected value is zero
        self.assertEqual(sharpe, ValiConfig.SHARPE_NOCONFIDENCE_VALUE)

    def test_sharpe_perfect_positive_year(self):
        """Test that the sharpe function works for 365 days of returns"""
        log_returns = [0.10/ValiConfig.DAYS_IN_YEAR_CRYPTO for _ in range(ValiConfig.DAYS_IN_YEAR_CRYPTO)]

        sharpe = Metrics.sharpe(log_returns)

        # Expected value is between zero and 10
        self.assertGreater(sharpe, 0.0)
        self.assertLess(sharpe, 10)

    def test_sortino_no_returns(self):
        """Test that the Sortino function returns 0.0 when there are no returns"""
        log_returns = []

        sortino = Metrics.sortino(log_returns)

        # Expected value is zero
        self.assertEqual(sortino, ValiConfig.SORTINO_NOCONFIDENCE_VALUE)

    def test_sortino_noconfidence_limit(self):
        """Test that the Sortino function returns 0.0 when there are no returns"""
        log_returns = [0.1] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N - 1)

        sortino = Metrics.sortino(log_returns)

        # Expected value is zero
        self.assertEqual(sortino, ValiConfig.SORTINO_NOCONFIDENCE_VALUE)

    def test_sortino_no_returns_no_variance(self):
        """Test that the Sortino function returns 0.0 when there are no returns and no variance"""
        log_returns = [0.0] * 100
        sortino = Metrics.sortino(log_returns)

        # Expected value will be the annual tbill rate, if there is no variance
        self.assertAlmostEqual(sortino, -ValiConfig.ANNUAL_RISK_FREE_PERCENTAGE)

    def test_sortino_no_losses(self):
        """Test that the Sortino function returns 0.0 when there are no losses"""
        log_returns = [0.002] * 100
        sortino = Metrics.sortino(log_returns)

        # Expected value is zero, as there are no losses and volatility is inf
        self.assertEqual(sortino, 0.0)

    def test_sortino_only_losses(self):
        """Test that the Sortino function returns 0.0 when there are no losses"""
        log_returns = [-0.002] * 100

        downside_volatility = Metrics.ann_downside_volatility(log_returns)
        volatility = Metrics.ann_volatility(log_returns)

        self.assertEqual(downside_volatility, volatility)

    def test_sortino_general(self):
        """Test that the Sortino function returns a positive value for general returns"""
        log_returns = [0.003, -0.002] * 50
        sortino = Metrics.sortino(log_returns)
        self.assertGreater(sortino, 0.0)

    def test_sortino_negative(self):
        """Test that the Sortino function returns a negative value for negative log returns"""
        log_returns = [-0.001, 0.001] * 60
        log_returns.append(-0.015)
        sortino = Metrics.sortino(log_returns)

        # Expected value less than zero for negative returns
        self.assertLess(sortino, 0.0)
        self.assertGreater(sortino, -10)

    def test_statistical_confidence_no_returns(self):
        """Test that the statistical confidence function returns 0.0 when there are no returns"""
        log_returns = []

        confidence = Metrics.statistical_confidence(log_returns)

        # Expected value is zero
        self.assertEqual(confidence, ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE)

    def test_statistical_confidence_noconfidence_limit(self):
        """Test that the statistical confidence function returns 0.0 when there are no returns"""
        log_returns = [0.1] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N - 1)

        confidence = Metrics.statistical_confidence(log_returns)

        # Expected value is zero
        self.assertEqual(confidence, ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE)

    def test_statistical_confidence_general(self):
        """Test that the statistical confidence function returns a positive value for general returns"""
        log_returns = [0.003, -0.002] * 50
        confidence = Metrics.statistical_confidence(log_returns)
        self.assertGreater(confidence, 0.0)

    def test_statistical_confidence_negative(self):
        """Test that the statistical confidence function returns a negative value for negative log returns"""
        log_returns = [0.001, -0.002] * 50
        confidence = Metrics.statistical_confidence(log_returns)

        # Expected value less than zero for negative returns
        self.assertLess(confidence, 0.0)

    def test_statistical_confidence_monotonic(self):
        """Test that the statistical confidence function is monotonic"""
        log_returns = [0.003, -0.002] * 50
        confidence = Metrics.statistical_confidence(log_returns)

        log_returns = [0.003, -0.002] * 70
        confidence_new = Metrics.statistical_confidence(log_returns)

        log_returns = [0.003, -0.002] * 100
        confidence_newest = Metrics.statistical_confidence(log_returns)

        self.assertLess(confidence, confidence_new)
        self.assertLess(confidence_new, confidence_newest)

    def test_ann_volatility(self):
        a = [9/252, 10/252, 11/252]
        b = [8/252, 10/252, 12/252]
        self.assertGreater(Metrics.ann_volatility(b), Metrics.ann_volatility(a))

    def test_ann_downside_volatility(self):
        a = [9/252, 10/252, 11/252]
        b = [8/252, 10/252, 12/252]
        self.assertEqual(Metrics.ann_downside_volatility(b), Metrics.ann_downside_volatility(a))
        self.assertEqual(Metrics.ann_downside_volatility(a), np.inf)

        c = [-9/252, -10/252, -11/252]
        d = [-8/252, -10/252, -12/252]
        self.assertGreater(Metrics.ann_downside_volatility(d), Metrics.ann_downside_volatility(c))

        e = copy.deepcopy(c)
        e.append(12/252)
        self.assertEqual(Metrics.ann_downside_volatility(e), Metrics.ann_downside_volatility(c))

    def test_weighting(self):
        log_returns = []
        empty_distribution = Metrics.weighting_distribution(log_returns).tolist()
        self.assertEqual(log_returns, empty_distribution)

        self.assertEqual(log_returns, Metrics.weighted_log_returns(log_returns))


        log_returns = [float('-inf')]
        one_entry_dist = Metrics.weighting_distribution(log_returns).tolist()
        self.assertEqual([1], one_entry_dist)

        #Test distribution
        log_returns = [random.uniform(-1, 1) for _ in range(ValiConfig.TARGET_LEDGER_WINDOW_DAYS)]
        large_distribution = Metrics.weighting_distribution(log_returns).tolist()
        for value in large_distribution:
            self.assertGreaterEqual(value, ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN)
            self.assertLessEqual(value, ValiConfig.WEIGHTED_AVERAGE_DECAY_MAX)
        #first day should be minimum
        self.assertAlmostEqual(large_distribution[0], ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN, places=2)
        #most recent day should be maximum
        self.assertAlmostEqual(large_distribution[-1], ValiConfig.WEIGHTED_AVERAGE_DECAY_MAX, places=2)

    def test_variance(self):
        log_returns = []
        self.assertEqual(0, Metrics.variance(log_returns))

        #Case for degrees of freedom
        log_returns = [1]
        self.assertEqual(np.inf, Metrics.variance(log_returns))

        log_returns = [1, 2]
        self.assertNotEqual(np.inf, Metrics.variance(log_returns))
        self.assertNotEqual(0, Metrics.variance(log_returns))

    def test_average(self):
        log_returns = []
        self.assertEqual(0, Metrics.average(log_returns))
        self.assertEqual(0, Metrics.average(log_returns, weighting=True))
        log_returns = [0.01, -20, 0.01]

        #The negative returns shouldn't be included in the average
        self.assertGreaterEqual(Metrics.average(log_returns, weighting=True, indices=[0, 2]), 0)

        #Without indices it should be negative
        self.assertLessEqual(Metrics.average(log_returns, weighting=True, indices=None), 0)

    def test_daily_max_drawdown_empty(self):
        """Test that daily_max_drawdown returns 0.0 for empty input"""
        # Test with empty list
        self.assertEqual(Metrics.daily_max_drawdown([]), 0.0)

    def test_daily_max_drawdown_constant(self):
        """Test that daily_max_drawdown returns 0.0 for constant returns"""
        # All zeros
        log_returns = [0.0] * 10
        self.assertEqual(Metrics.daily_max_drawdown(log_returns), 0.0)

        # All same positive value
        log_returns = [0.01] * 10
        self.assertEqual(Metrics.daily_max_drawdown(log_returns), 0.0)

    def test_daily_max_drawdown_monotonic(self):
        """Test daily_max_drawdown with monotonically increasing/decreasing returns"""
        # Monotonically increasing returns should have zero drawdown
        log_returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        self.assertEqual(Metrics.daily_max_drawdown(log_returns), 0.0)

        # Monotonically decreasing returns should have drawdown
        log_returns = [0.05, -0.02, -0.03, -0.04, -0.05]
        mdd = Metrics.daily_max_drawdown(log_returns)
        self.assertGreater(mdd, 0.0)

    def test_daily_max_drawdown_recovery(self):
        """Test daily_max_drawdown with a pattern that drops and recovers"""
        # Peak, drawdown, and recovery
        log_returns = [0.1, 0.1, -0.15, -0.1, 0.05, 0.2, 0.1]
        mdd = Metrics.daily_max_drawdown(log_returns)

        # Should have a significant drawdown
        self.assertGreater(mdd, 0.0)
        # But less than 100%
        self.assertLess(mdd, 1.0)

    def test_daily_max_drawdown_detailed(self):
        """Test daily_max_drawdown with a specific pattern to confirm exact calculation"""
        # Design a specific scenario with known drawdown
        # Initial value 100, goes to 110, drops to 77, recovers to 99
        # Drawdown should be (77/110) - 1 = -30%
        log_returns = [
            math.log(1.1),  # Day 1: +10% (value = 110)
            math.log(0.7),  # Day 2: -30% (value = 77)
            math.log(1.1),  # Day 3: +10% (value = 84.7)
            math.log(1.1),  # Day 4: +10% (value = 93.17)
            math.log(1.06),  # Day 5: +6% (value = 98.76)
        ]

        mdd = Metrics.daily_max_drawdown(log_returns)
        self.assertAlmostEqual(mdd, 0.3, delta=0.01,
                              msg="Maximum drawdown should be approximately 30%")

    def test_daily_max_drawdown_all_negative(self):
        """Test daily_max_drawdown with all negative returns"""
        log_returns = [-0.01, -0.02, -0.03, -0.04, -0.05]
        mdd = Metrics.daily_max_drawdown(log_returns)

        # Should have a significant drawdown
        self.assertGreater(mdd, 0.0)
        # For small negative returns, drawdown should be less than 15%
        self.assertLess(mdd, 0.15)

    def test_daily_max_drawdown_extreme(self):
        """Test daily_max_drawdown with extreme market crash scenario"""
        # Simulate a market crash with -50% in one day
        log_returns = [0.01, 0.02, math.log(0.5), 0.01, 0.02]
        mdd = Metrics.daily_max_drawdown(log_returns)

        # Max drawdown should be close to 50%
        self.assertAlmostEqual(mdd, 0.5, delta=0.02)

    def test_pnl_score_empty_ledger(self):
        """Test pnl_score with empty or None ledger"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        
        # Test with None ledger
        result = Metrics.pnl_score([], None)
        self.assertEqual(result, ValiConfig.PNL_NOCONFIDENCE_VALUE)
        
        # Test with empty ledger
        empty_ledger = PerfLedger(cps=[])
        result = Metrics.pnl_score([], empty_ledger)
        self.assertEqual(result, ValiConfig.PNL_NOCONFIDENCE_VALUE)

    def test_pnl_score_no_daily_pnl(self):
        """Test pnl_score when daily_pnl returns empty list"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint
        from unittest.mock import patch
        
        # Create a ledger that will have empty daily_pnl
        ledger = PerfLedger(cps=[
            PerfCheckpoint(
                last_update_ms=1000,
                prev_portfolio_ret=1.0,
                accum_ms=100,  # Partial accumulation - won't count as complete day
                portfolio_realized_pnl=10.0,
                portfolio_unrealized_pnl=5.0
            )
        ])
        
        result = Metrics.pnl_score([], ledger)
        self.assertEqual(result, ValiConfig.PNL_NOCONFIDENCE_VALUE)

    def test_pnl_score_with_positive_pnl(self):
        """Test pnl_score with positive daily PnL values"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl to return positive values
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [100.0, 150.0, 200.0]  # 3 days of positive PnL
            
            ledger = PerfLedger(cps=[])  # Actual content doesn't matter due to mocking
            result = Metrics.pnl_score([], ledger, weighting=False)
            
            # Should return average of the daily PnL values
            expected = (100.0 + 150.0 + 200.0) / 3
            self.assertEqual(result, expected)

    def test_pnl_score_with_negative_pnl(self):
        """Test pnl_score with negative daily PnL values"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl to return negative values
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [-50.0, -75.0, -25.0]  # 3 days of negative PnL
            
            ledger = PerfLedger(cps=[])
            result = Metrics.pnl_score([], ledger, weighting=False)
            
            # Should return average of the daily PnL values
            expected = (-50.0 + -75.0 + -25.0) / 3
            self.assertEqual(result, expected)

    def test_pnl_score_with_mixed_pnl(self):
        """Test pnl_score with mixed positive and negative daily PnL values"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl to return mixed values
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [100.0, -50.0, 75.0, -25.0]  # Mixed PnL
            
            ledger = PerfLedger(cps=[])
            result = Metrics.pnl_score([], ledger, weighting=False)
            
            # Should return average of the daily PnL values
            expected = (100.0 + -50.0 + 75.0 + -25.0) / 4
            self.assertEqual(result, expected)

    def test_pnl_score_with_zero_pnl(self):
        """Test pnl_score with zero daily PnL values"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl to return zero values
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [0.0, 0.0, 0.0]  # 3 days of zero PnL
            
            ledger = PerfLedger(cps=[])
            result = Metrics.pnl_score([], ledger, weighting=False)
            
            # Should return 0.0
            self.assertEqual(result, 0.0)

    def test_pnl_score_without_weighting(self):
        """Test pnl_score without time weighting"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl and Metrics.average
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl, \
             patch('vali_objects.utils.metrics.Metrics.average') as mock_average:
            
            mock_daily_pnl.return_value = [10.0, 20.0, 30.0]
            mock_average.return_value = 20.0
            
            ledger = PerfLedger(cps=[])
            result = Metrics.pnl_score([], ledger, weighting=False)
            
            # Verify average was called with weighting=False
            mock_average.assert_called_once_with([10.0, 20.0, 30.0], weighting=False)
            self.assertEqual(result, 20.0)

    def test_pnl_score_with_weighting(self):
        """Test pnl_score with time weighting enabled"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl and Metrics.average
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl, \
             patch('vali_objects.utils.metrics.Metrics.average') as mock_average:
            
            mock_daily_pnl.return_value = [10.0, 20.0, 30.0]
            mock_average.return_value = 25.0  # Different result due to weighting
            
            ledger = PerfLedger(cps=[])
            result = Metrics.pnl_score([], ledger, weighting=True)
            
            # Verify average was called with weighting=True
            mock_average.assert_called_once_with([10.0, 20.0, 30.0], weighting=True)
            self.assertEqual(result, 25.0)

    def test_pnl_score_single_day(self):
        """Test pnl_score with single day of PnL data"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl to return single value
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [42.5]  # Single day
            
            ledger = PerfLedger(cps=[])
            result = Metrics.pnl_score([], ledger, weighting=False)
            
            # Should return the single value
            self.assertEqual(result, 42.5)

    def test_pnl_score_large_pnl_values(self):
        """Test pnl_score with large PnL values"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl to return large values
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [10000.0, 15000.0, 12000.0]  # Large PnL values
            
            ledger = PerfLedger(cps=[])
            result = Metrics.pnl_score([], ledger, weighting=False)
            
            # Should handle large values correctly
            expected = (10000.0 + 15000.0 + 12000.0) / 3
            self.assertEqual(result, expected)

    def test_pnl_score_bypass_confidence_parameter(self):
        """Test that pnl_score accepts bypass_confidence parameter (though it may not use it)"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [50.0, 60.0]
            
            ledger = PerfLedger(cps=[])
            
            # Test with bypass_confidence=True
            result1 = Metrics.pnl_score([], ledger, bypass_confidence=True, weighting=False)
            
            # Test with bypass_confidence=False
            result2 = Metrics.pnl_score([], ledger, bypass_confidence=False, weighting=False)
            
            # Should return same result regardless (as noted in function comment)
            expected = (50.0 + 60.0) / 2
            self.assertEqual(result1, expected)
            self.assertEqual(result2, expected)

    def test_pnl_score_log_returns_parameter_unused(self):
        """Test that pnl_score doesn't use the log_returns parameter"""
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
        from unittest.mock import patch
        
        # Mock LedgerUtils.daily_pnl
        with patch('vali_objects.utils.metrics.LedgerUtils.daily_pnl') as mock_daily_pnl:
            mock_daily_pnl.return_value = [30.0, 40.0]
            
            ledger = PerfLedger(cps=[])
            
            # Test with different log_returns values - should not affect result
            result1 = Metrics.pnl_score([0.1, 0.2, 0.3], ledger, weighting=False)
            result2 = Metrics.pnl_score([], ledger, weighting=False)
            result3 = Metrics.pnl_score([100, 200], ledger, weighting=False)
            
            # All should return same result since log_returns is unused
            expected = (30.0 + 40.0) / 2
            self.assertEqual(result1, expected)
            self.assertEqual(result2, expected)
            self.assertEqual(result3, expected)
