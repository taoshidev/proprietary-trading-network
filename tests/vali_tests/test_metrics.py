import copy

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
        log_returns = [0.05/ 365 for _ in range(365)]

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

        # Expected value is minimum downside loss
        self.assertAlmostEqual(sortino, 0.0)

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
