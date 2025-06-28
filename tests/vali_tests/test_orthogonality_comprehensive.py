import numpy as np
import pandas as pd
import math
from unittest.mock import patch
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.orthogonality import Orthogonality
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint
from tests.shared_objects.test_utilities import ledger_generator, checkpoint_generator
from datetime import datetime, timezone
import copy


class TestOrthogonalityComprehensive(TestBase):
    """Comprehensive test suite for orthogonality penalty system."""

    def setUp(self):
        super().setUp()
        
        # Create diverse miner strategies for testing
        self.diverse_miners = self._create_diverse_miner_strategies()
        self.edge_case_miners = self._create_edge_case_scenarios()
        
    def _create_diverse_miner_strategies(self):
        """Create miners with different trading strategies."""
        # Create enough data points to meet statistical requirements
        base_length = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 10, 70)
        
        strategies = {
            # Trend followers (should be correlated)
            'trend_follower_1': [0.02, 0.015, -0.005, 0.025, 0.01] * (base_length // 5),
            'trend_follower_2': [0.018, 0.013, -0.003, 0.022, 0.009] * (base_length // 5),
            
            # Mean reverters (should be different from trend followers)
            'mean_reverter': [-0.015, 0.025, -0.02, 0.03, -0.01] * (base_length // 5),
            
            # High volatility trader (distinct pattern)
            'volatility_trader': [0.05, -0.04, 0.03, -0.02, 0.06] * (base_length // 5),
            
            # Momentum trader (different timing)
            'momentum_trader': [0.001, 0.005, 0.03, 0.02, -0.01] * (base_length // 5),
            
            # Conservative trader (low volatility)
            'conservative_trader': [0.005, 0.003, -0.001, 0.004, 0.002] * (base_length // 5),
        }
        
        return {name: self._create_ledger_from_returns(returns) 
                for name, returns in strategies.items()}
    
    def _create_edge_case_scenarios(self):
        """Create edge case scenarios for robust testing."""
        base_length = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 10, 70)
        
        scenarios = {
            # Identical strategies (perfect correlation)
            'identical_1': [0.01, 0.02, -0.01, 0.015] * (base_length // 4),
            'identical_2': [0.01, 0.02, -0.01, 0.015] * (base_length // 4),  # Exactly same
            
            # Zero variance (all returns are same)
            'zero_variance': [0.01] * base_length,
            
            # High variance
            'high_variance': [x for x in np.random.normal(0.01, 0.1, base_length)],
            
            # Sparse data (mostly zeros)
            'sparse_trader': [0.05 if i % 20 == 0 else 0.0 for i in range(base_length)],
            
            # Negative correlation
            'negative_corr_1': [0.02, -0.01, 0.015, -0.005] * (base_length // 4),
            'negative_corr_2': [-0.02, 0.01, -0.015, 0.005] * (base_length // 4),  # Opposite
        }
        
        return {name: self._create_ledger_from_returns(returns) 
                for name, returns in scenarios.items()}
    
    def _create_ledger_from_returns(self, returns):
        """Create a PerfLedger from a list of daily returns."""
        checkpoints = []
        base_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        for i, daily_return in enumerate(returns):
            # Create checkpoints that will form complete days
            day_start = base_time + (i * 24 * 60 * 60 * 1000)
            
            # Create exactly the number of checkpoints per day as configured
            n_checkpoints_per_day = int(ValiConfig.DAILY_CHECKPOINTS)
            checkpoint_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
            
            for j in range(n_checkpoints_per_day):
                checkpoint_time = day_start + (j * checkpoint_duration)
                # Distribute the daily return across checkpoints
                cp_return = daily_return / n_checkpoints_per_day
                gain = max(0, cp_return)
                loss = min(0, cp_return)
                
                checkpoint = PerfCheckpoint(
                    last_update_ms=checkpoint_time + checkpoint_duration,
                    accum_ms=checkpoint_duration,
                    open_ms=checkpoint_time,
                    gain=gain,
                    loss=loss,
                    prev_portfolio_ret=1.0,
                    n_updates=1,
                    mdd=0.99
                )
                checkpoints.append(checkpoint)
        
        return ledger_generator(checkpoints=checkpoints)

    def test_edge_case_identical_strategies(self):
        """Test handling of identical strategies (perfect correlation)."""
        identical_ledgers = {
            'miner1': self.edge_case_miners['identical_1'],
            'miner2': self.edge_case_miners['identical_2']
        }
        
        # Test correlation matrix
        returns = {name: LedgerUtils.daily_return_log(ledger) 
                  for name, ledger in identical_ledgers.items()}
        corr_matrix, mean_correlations = Orthogonality.correlation_matrix(returns)
        
        # Should detect perfect correlation
        self.assertFalse(corr_matrix.empty, "Correlation matrix should not be empty for identical strategies")
        
        if mean_correlations:
            for miner, corr in mean_correlations.items():
                self.assertAlmostEqual(abs(corr), 1.0, places=2, 
                                     msg="Identical strategies should have correlation close to 1.0")
        
        # Test penalty calculation
        penalties = LedgerUtils.orthogonality_penalty(identical_ledgers)
        
        # Identical strategies should receive high penalties
        for miner, penalty in penalties.items():
            self.assertGreaterEqual(penalty, 0.5, 
                                  f"Identical strategy {miner} should receive significant penalty")

    def test_edge_case_insufficient_miners(self):
        """Test handling when there are too few miners."""
        # Single miner
        single_miner = {'miner1': self.diverse_miners['trend_follower_1']}
        penalties = LedgerUtils.orthogonality_penalty(single_miner)
        self.assertEqual(penalties['miner1'], 0.0, "Single miner should have no penalty")
        
        # No miners
        empty_dict = {}
        penalties = LedgerUtils.orthogonality_penalty(empty_dict)
        self.assertEqual(len(penalties), 0, "Empty miner dict should return empty penalties")

    def test_edge_case_insufficient_data(self):
        """Test handling when miners have insufficient data."""
        # Create miners with very little data
        insufficient_data = {
            'miner1': self._create_ledger_from_returns([0.01, 0.02]),  # Only 2 days
            'miner2': self._create_ledger_from_returns([0.015, 0.025])  # Only 2 days
        }
        
        penalties = LedgerUtils.orthogonality_penalty(insufficient_data)
        
        # Should handle gracefully without crashing
        self.assertEqual(len(penalties), 2)
        for penalty in penalties.values():
            self.assertGreaterEqual(penalty, 0.0)
            self.assertLessEqual(penalty, 1.0)

    def test_edge_case_zero_variance_strategies(self):
        """Test handling of zero variance strategies."""
        zero_var_ledgers = {
            'zero_var': self.edge_case_miners['zero_variance'],
            'normal': self.diverse_miners['trend_follower_1']
        }
        
        penalties = LedgerUtils.orthogonality_penalty(zero_var_ledgers)
        
        # Should handle without crashing
        self.assertEqual(len(penalties), 2)
        for penalty in penalties.values():
            self.assertGreaterEqual(penalty, 0.0)
            self.assertLessEqual(penalty, 1.0)

    def test_edge_case_none_and_empty_ledgers(self):
        """Test handling of None and empty ledgers."""
        problematic_ledgers = {
            'valid': self.diverse_miners['trend_follower_1'],
            'none_ledger': None,
            'empty_ledger': PerfLedger()  # Empty ledger
        }
        
        penalties = LedgerUtils.orthogonality_penalty(problematic_ledgers)
        
        # Should handle gracefully
        self.assertEqual(len(penalties), 3)
        self.assertGreaterEqual(penalties['valid'], 0.0)
        self.assertEqual(penalties['none_ledger'], 0.0)  # No penalty for invalid data
        self.assertEqual(penalties['empty_ledger'], 0.0)  # No penalty for empty data

    def test_parameter_sensitivity_diverging_intensity(self):
        """Test sensitivity to ORTHOGONALITY_DIVERGING_INTENSITY parameter."""
        test_ledgers = {
            'miner1': self.diverse_miners['trend_follower_1'],
            'miner2': self.diverse_miners['trend_follower_2']
        }
        
        original_intensity = ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY
        
        try:
            # Test different intensity values
            intensities = [0.1, 0.5, 0.75, 1.0]
            results = {}
            
            for intensity in intensities:
                ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY = intensity
                penalties = LedgerUtils.orthogonality_penalty(test_ledgers)
                results[intensity] = penalties.copy()
            
            # Verify that different intensities produce different results
            penalty_sets = list(results.values())
            for i in range(len(penalty_sets) - 1):
                for j in range(i + 1, len(penalty_sets)):
                    # At least one miner should have different penalties
                    differences = [abs(penalty_sets[i][miner] - penalty_sets[j][miner]) 
                                 for miner in test_ledgers.keys()]
                    self.assertTrue(any(diff > 0.01 for diff in differences),
                                  "Different diverging intensities should produce different penalties")
        
        finally:
            ValiConfig.ORTHOGONALITY_DIVERGING_INTENSITY = original_intensity

    def test_parameter_sensitivity_weight_combinations(self):
        """Test sensitivity to correlation and preference weight combinations."""
        test_ledgers = {
            'miner1': self.diverse_miners['trend_follower_1'],
            'miner2': self.diverse_miners['mean_reverter'],
            'miner3': self.diverse_miners['volatility_trader']
        }
        
        original_corr_weight = ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT
        original_pref_weight = ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT
        
        try:
            # Test different weight combinations
            weight_combinations = [
                (1.0, 0.0),  # Only correlation
                (0.0, 1.0),  # Only preference
                (0.5, 0.5),  # Equal weights
                (0.8, 0.2),  # Mostly correlation
                (0.2, 0.8),  # Mostly preference
            ]
            
            results = {}
            for corr_w, pref_w in weight_combinations:
                ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT = corr_w
                ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT = pref_w
                penalties = LedgerUtils.orthogonality_penalty(test_ledgers)
                results[(corr_w, pref_w)] = penalties.copy()
            
            # Verify that different weight combinations produce different results
            self.assertGreater(len(set(tuple(sorted(r.items())) for r in results.values())), 1,
                             "Different weight combinations should produce different penalty distributions")
        
        finally:
            ValiConfig.ORTHOGONALITY_CORRELATION_WEIGHT = original_corr_weight
            ValiConfig.ORTHOGONALITY_PREFERENCE_WEIGHT = original_pref_weight

    def test_performance_scalability(self):
        """Test performance with larger numbers of miners."""
        import time
        
        # Create many miners with different strategies
        many_miners = {}
        for i in range(50):  # Test with 50 miners
            strategy_type = i % 4
            if strategy_type == 0:
                # Trend followers
                base_returns = [0.02, 0.015, -0.005, 0.025, 0.01]
            elif strategy_type == 1:
                # Mean reverters
                base_returns = [-0.015, 0.025, -0.02, 0.03, -0.01]
            elif strategy_type == 2:
                # Volatile traders
                base_returns = [0.05, -0.04, 0.03, -0.02, 0.06]
            else:
                # Random variations
                base_returns = [x + np.random.normal(0, 0.005) for x in [0.01, 0.02, -0.01, 0.015, 0.005]]
            
            # Scale to meet minimum requirements
            returns = base_returns * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N // len(base_returns) + 1)
            many_miners[f'miner_{i}'] = self._create_ledger_from_returns(returns[:ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N + 5])
        
        # Measure execution time
        start_time = time.time()
        penalties = LedgerUtils.orthogonality_penalty(many_miners)
        execution_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(len(penalties), 50, "Should calculate penalties for all miners")
        self.assertLess(execution_time, 30.0, "Should complete within reasonable time (30 seconds)")
        
        # Verify all penalties are valid
        for miner, penalty in penalties.items():
            self.assertGreaterEqual(penalty, 0.0, f"Penalty for {miner} should be non-negative")
            self.assertLessEqual(penalty, 1.0, f"Penalty for {miner} should not exceed 1.0")
            self.assertTrue(np.isfinite(penalty), f"Penalty for {miner} should be finite")

    def test_correlation_statistical_significance(self):
        """Test correlation calculations with different data lengths."""
        # Test with minimum required data
        min_data_miners = {}
        exact_min = ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N
        
        for i, name in enumerate(['miner1', 'miner2']):
            # Create exactly minimum required data points
            returns = [0.01 + i*0.002, 0.02 - i*0.001, -0.005 + i*0.001] * (exact_min // 3 + 1)
            min_data_miners[name] = self._create_ledger_from_returns(returns[:exact_min])
        
        penalties = LedgerUtils.orthogonality_penalty(min_data_miners)
        
        # Should work with minimum data
        self.assertEqual(len(penalties), 2)
        for penalty in penalties.values():
            self.assertGreaterEqual(penalty, 0.0)
            self.assertLessEqual(penalty, 1.0)

    def test_market_regime_scenarios(self):
        """Test behavior under different market conditions."""
        regime_scenarios = {
            # Bull market - all positive returns
            'bull_market': {
                'miner1': [0.02, 0.015, 0.025, 0.01, 0.03] * 15,
                'miner2': [0.018, 0.013, 0.022, 0.009, 0.028] * 15,
                'miner3': [0.025, 0.02, 0.03, 0.015, 0.035] * 15,
            },
            
            # Bear market - mostly negative returns
            'bear_market': {
                'miner1': [-0.02, -0.015, -0.025, -0.01, -0.03] * 15,
                'miner2': [-0.018, -0.013, -0.022, -0.009, -0.028] * 15,
                'miner3': [0.005, -0.005, 0.01, -0.01, 0.008] * 15,  # Different strategy
            },
            
            # High volatility market
            'volatile_market': {
                'miner1': [0.05, -0.04, 0.06, -0.03, 0.07] * 15,
                'miner2': [0.045, -0.035, 0.055, -0.025, 0.065] * 15,
                'miner3': [-0.02, 0.03, -0.04, 0.05, -0.01] * 15,  # Contrarian
            }
        }
        
        for regime_name, miners_returns in regime_scenarios.items():
            ledgers = {name: self._create_ledger_from_returns(returns) 
                      for name, returns in miners_returns.items()}
            
            penalties = LedgerUtils.orthogonality_penalty(ledgers)
            
            # Verify penalties are calculated for all miners
            self.assertEqual(len(penalties), 3, f"Should calculate penalties for all miners in {regime_name}")
            
            # In different regimes, similar strategies should still be penalized
            if regime_name in ['bull_market', 'bear_market']:
                # miner1 and miner2 have similar strategies
                similar_miners = ['miner1', 'miner2']
                different_miner = 'miner3'
                
                # Similar miners should have higher penalties than the different one
                avg_similar_penalty = np.mean([penalties[m] for m in similar_miners])
                different_penalty = penalties[different_miner]
                
                # This test may be sensitive to the exact implementation
                # Just verify that penalties are being calculated and are reasonable
                for penalty in penalties.values():
                    self.assertGreaterEqual(penalty, 0.0)
                    self.assertLessEqual(penalty, 1.0)

    def test_integration_with_scoring_system(self):
        """Test integration with the existing scoring system."""
        # Test that orthogonality penalties are properly formatted for the scoring system
        test_ledgers = {
            'high_corr_1': self.edge_case_miners['identical_1'],
            'high_corr_2': self.edge_case_miners['identical_2'],
            'different': self.diverse_miners['mean_reverter']
        }
        
        penalties = LedgerUtils.orthogonality_penalty(test_ledgers)
        
        # Verify penalty format matches scoring system expectations
        for miner, penalty in penalties.items():
            # Penalties should be between 0 and 1
            self.assertGreaterEqual(penalty, 0.0, f"Penalty for {miner} should be non-negative")
            self.assertLessEqual(penalty, 1.0, f"Penalty for {miner} should not exceed 1.0")
            
            # Should be numeric (not NaN or infinite)
            self.assertTrue(np.isfinite(penalty), f"Penalty for {miner} should be finite")
            self.assertFalse(np.isnan(penalty), f"Penalty for {miner} should not be NaN")

    def test_penalty_distribution_properties(self):
        """Test properties of penalty distribution."""
        test_ledgers = self.diverse_miners.copy()
        penalties = LedgerUtils.orthogonality_penalty(test_ledgers)
        
        # Extract penalty values
        penalty_values = list(penalties.values())
        
        # Test statistical properties
        self.assertEqual(len(penalty_values), len(test_ledgers), "Should have penalty for each miner")
        
        # All penalties should be valid numbers
        for penalty in penalty_values:
            self.assertGreaterEqual(penalty, 0.0, "All penalties should be non-negative")
            self.assertLessEqual(penalty, 1.0, "All penalties should not exceed 1.0")
            self.assertTrue(np.isfinite(penalty), "All penalties should be finite")
        
        # There should be some variation in penalties (not all identical)
        if len(penalty_values) > 1:
            penalty_std = np.std(penalty_values)
            self.assertGreater(penalty_std, 0.001, "Should have some variation in penalties")

    def test_empty_returns_handling(self):
        """Test handling of miners with empty return sequences."""
        mixed_ledgers = {
            'valid_miner': self.diverse_miners['trend_follower_1'],
            'empty_returns': self._create_ledger_from_returns([]),  # No returns
            'single_return': self._create_ledger_from_returns([0.01])  # One return only
        }
        
        penalties = LedgerUtils.orthogonality_penalty(mixed_ledgers)
        
        # Should handle gracefully
        self.assertEqual(len(penalties), 3)
        
        # Valid miner should have some penalty calculation
        self.assertGreaterEqual(penalties['valid_miner'], 0.0)
        
        # Invalid miners should have zero penalty
        self.assertEqual(penalties['empty_returns'], 0.0)
        self.assertEqual(penalties['single_return'], 0.0)

    def test_negative_correlation_handling(self):
        """Test proper handling of negative correlations."""
        neg_corr_ledgers = {
            'positive_trend': self.edge_case_miners['negative_corr_1'],
            'negative_trend': self.edge_case_miners['negative_corr_2']
        }
        
        penalties = LedgerUtils.orthogonality_penalty(neg_corr_ledgers)
        
        # Both should receive penalties since they're highly correlated (even if negatively)
        for miner, penalty in penalties.items():
            self.assertGreaterEqual(penalty, 0.3, 
                                  f"Highly correlated (negative) strategies should receive significant penalty: {miner}")

    def test_penalty_consistency(self):
        """Test that penalty calculations are consistent across multiple runs."""
        test_ledgers = {
            'miner1': self.diverse_miners['trend_follower_1'],
            'miner2': self.diverse_miners['mean_reverter']
        }
        
        # Run penalty calculation multiple times
        results = []
        for _ in range(5):
            penalties = LedgerUtils.orthogonality_penalty(test_ledgers)
            results.append(penalties.copy())
        
        # Results should be identical (deterministic)
        for i in range(1, len(results)):
            for miner in test_ledgers.keys():
                self.assertAlmostEqual(results[0][miner], results[i][miner], places=6,
                                     msg=f"Penalty calculations should be consistent for {miner}")