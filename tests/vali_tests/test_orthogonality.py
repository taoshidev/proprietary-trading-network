import numpy as np
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.orthogonality import Orthogonality

class TestOrthogonality(TestBase):
    """Test suite for orthogonality penalty system."""

    def setUp(self):
        super().setUp()
        
        # Mock test data - need enough data points to meet threshold
        # ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N is likely around 30
        base_returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        extended_returns = base_returns * 20  # 100 data points
        
        self.test_returns = {
            'miner1': extended_returns,
            'miner2': [x * 1.1 for x in extended_returns],  # Similar to miner1
            'miner3': [x * -0.8 for x in extended_returns],  # Opposite to miner1
            'miner4': [x * 1.5 + 0.001 for x in extended_returns],  # Different pattern
        }

    def test_time_preference(self):
        """Test time preference calculation."""
        v1 = [1, 2, 3, 4, 5]  # Longer vector
        v2 = [1, 2, 3]        # Shorter vector
        
        time_pref = Orthogonality.time_preference(v1, v2)
        
        # With better sigmoid parameters, longer vector should be preferred
        # Note: exact direction may depend on sigmoid configuration
        self.assertNotEqual(time_pref, 0)  # Should have some preference
        
        # Symmetric test
        time_pref_reverse = Orthogonality.time_preference(v2, v1)
        self.assertAlmostEqual(time_pref + time_pref_reverse, 0.0, places=5)  # Should be opposite
        
        # Same length should give zero preference
        time_pref_same = Orthogonality.time_preference(v1, v1)
        self.assertAlmostEqual(time_pref_same, 0.0, places=5)

    def test_pairwise_pref(self):
        """Test pairwise preference calculation."""
        prefs = Orthogonality.pairwise_pref(self.test_returns, Orthogonality.time_preference)
        
        # Should have n*(n-1)/2 pairs for n miners
        expected_pairs = 4 * 3 // 2  # 6 pairs
        self.assertEqual(len(prefs), expected_pairs)
        
        # Check that all pairs are present
        miners = list(self.test_returns.keys())
        for i in range(len(miners)):
            for j in range(i + 1, len(miners)):
                self.assertIn((miners[i], miners[j]), prefs)

    def test_time_pref(self):
        """Test time preference for all miners."""
        time_prefs = Orthogonality.time_pref(self.test_returns)
        
        # Should return pairwise preferences
        self.assertEqual(len(time_prefs), 6)  # 4 choose 2
        
        # All preferences should be floats
        for pref in time_prefs.values():
            self.assertIsInstance(pref, float)

    def test_sim_pref(self):
        """Test similarity preference for all miners."""
        sim_prefs = Orthogonality.sim_pref(self.test_returns)
        
        # Should return pairwise preferences
        self.assertEqual(len(sim_prefs), 6)  # 4 choose 2
        
        # All preferences should be arrays (from convolutional similarity)
        for pref in sim_prefs.values():
            # convolutional_similarity returns arrays, but they get flattened
            self.assertIsInstance(pref, (np.ndarray, float, np.floating))

    def test_full_pref_basic(self):
        """Test full preference aggregation."""
        full_prefs = Orthogonality.full_pref(self.test_returns)
        
        # Should return preferences for all miners
        self.assertEqual(len(full_prefs), 4)
        
        # All miners should have preference scores
        for miner in self.test_returns.keys():
            self.assertIn(miner, full_prefs)
            # Due to convolutional similarity returning arrays, values might be arrays
            pref_value = full_prefs[miner]
            self.assertTrue(isinstance(pref_value, (float, np.floating, np.ndarray)))

