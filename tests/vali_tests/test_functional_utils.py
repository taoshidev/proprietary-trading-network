# developer: trdougherty
import numpy as np

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.functional_utils import FunctionalUtils

class TestFunctionalUtils(TestBase):
    def setUp(self):
        super().setUp()

    def test_sigmoid_basic(self):
        # Test basic functionality with no shift or spread
        self.assertAlmostEqual(FunctionalUtils.sigmoid(0), 0.5, places=3)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(10), 0, places=3)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(-10), 1, places=3)

    def test_sigmoid_shift(self):
        # Test the effect of shifting the sigmoid
        self.assertAlmostEqual(FunctionalUtils.sigmoid(0, shift=2), FunctionalUtils.sigmoid(-2))
        self.assertAlmostEqual(FunctionalUtils.sigmoid(2, shift=2), 0.5)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(4, shift=2), FunctionalUtils.sigmoid(2))

    def test_sigmoid_spread(self):
        # Test the effect of different spread values
        self.assertAlmostEqual(FunctionalUtils.sigmoid(0, spread=2), np.clip(1 / (1 + np.exp(2 * 0)), 0, 1))
        self.assertAlmostEqual(FunctionalUtils.sigmoid(1, spread=2), np.clip(1 / (1 + np.exp(2 * (1 - 0))), 0, 1))
        self.assertAlmostEqual(FunctionalUtils.sigmoid(1, spread=-2), np.clip(1 / (1 + np.exp(-2 * (1 - 0))), 0, 1))

    def test_sigmoid_spread_zero(self):
        # Test that providing spread=0 raises a ValueError
        with self.assertRaises(ValueError):
            FunctionalUtils.sigmoid(0, spread=0)

    def test_sigmoid_extreme_values(self):
        # Test extreme large and small values of x
        self.assertAlmostEqual(FunctionalUtils.sigmoid(1000), 0.0)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(-1000), 1.0)

    def test_sigmoid_extreme_spread(self):
        # Test with very large positive and negative spread
        self.assertAlmostEqual(FunctionalUtils.sigmoid(1, spread=1000), 0.0)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(1, spread=-1000), 1.0)

    def test_sigmoid_extreme_shift(self):
        # Test with extreme shifts
        self.assertAlmostEqual(FunctionalUtils.sigmoid(1000, shift=1000), 0.5)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(-1000, shift=-1000), 0.5)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(0, shift=1000), 1.0)
        self.assertAlmostEqual(FunctionalUtils.sigmoid(0, shift=-1000), 0.0)