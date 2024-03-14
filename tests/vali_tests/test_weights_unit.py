# developer: trdougherty
# Copyright © 2024 Taoshi Inc
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring

import numpy as np
import random

class TestWeights(TestBase):

    def setUp(self):
        super().setUp()

        ## seeding
        np.random.seed(0)
        random.seed(0)

        n_miners = 50
        self.n_miners = n_miners

        miner_names = [ 'miner'+str(x) for x in range(n_miners) ]
        miner_values = [ random.random() / 5 + 1 for x in range(n_miners) ]

        returns = dict(zip(miner_names, miner_values))
        returns['miner0'] = 0.7 # this is going to be the outlier - the one which should get filtered

        self.returns = returns

    def test_miner_filtering(self):
        """
        Test that the miner filtering works as expected
        """
        filtered_results = Scoring.filter_results(self.returns)
        filtered_netuids = [ x[0] for x in filtered_results ]

        self.assertIsInstance(filtered_results, list)

        self.assertTrue('miner0' not in filtered_netuids)
        self.assertFalse('miner1' not in filtered_netuids)

        filtered_results_names = [ x[0] for x in filtered_results ]
        self.assertTrue('miner0' not in filtered_results_names)
        self.assertFalse('miner1' not in filtered_results_names)

    def test_miner_filtering_no_miners(self):
        """
        Test that the miner filtering works as expected when there are no miners
        """
        returns = {}
        filtered_results = Scoring.filter_results(returns)
        self.assertEqual(filtered_results, list(returns.items()))

    def test_miner_filtering_one_miner(self):
        """
        Test that the miner filtering works as expected when there is only one miner
        """
        returns = {'miner0': 1.1}
        filtered_results = Scoring.filter_results(returns)
        filtered_netuids = [ x[0] for x in filtered_results ]
        filtered_values = [ x[1] for x in filtered_results ]

        self.assertEqual(sorted(filtered_netuids), sorted(list(returns.keys())))
        self.assertNotEqual(sorted(filtered_values), sorted(list(returns.values())))
        self.assertEqual(filtered_results, [('miner0', 1.0)])

    def test_transform_and_scale_results(self):
        """Test that the transform and scale results works as expected"""
        filtered_results = Scoring.filter_results(self.returns)
        filtered_netuids = [ x[0] for x in filtered_results ]

        scaled_transformed_list: list[tuple[str, float]] = Scoring.transform_and_scale_results(filtered_results)

        # Check that the result is a list of tuples with string and float elements
        self.assertIsInstance(scaled_transformed_list, list)
        for item in scaled_transformed_list:
            self.assertIsInstance(item, tuple)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

        # Check that the values are sorted in descending order
        values = [x[1] for x in scaled_transformed_list]
        self.assertEqual(values, sorted(values, reverse=True))

        # Check that the values are scaled correctly
        self.assertAlmostEqual(sum(values), 1.0, places=5)

        # Check that the names are in the same order as the filtered results
        filtered_returns_sorted = sorted(filtered_results, key=lambda x: x[1], reverse=True)
        filtered_return_names = [x[0] for x in filtered_returns_sorted]
        scaled_transformed_names = [x[0] for x in scaled_transformed_list]
        self.assertEqual(scaled_transformed_names, filtered_return_names)
        
        # Check that the max miner was returned and still has the highest value
        max_minername = max(self.returns, key=self.returns.get)
        self.assertEqual(scaled_transformed_list[0][0], max_minername)

        # Check that the values are scaled correctly, assuming we filtered one miner
        # 0.152 = exponential_decay_returns(49)[0]
        self.assertAlmostEqual(scaled_transformed_list[0][1], 0.152, places=3)
        self.assertAlmostEqual(scaled_transformed_list[-1][1], 0.0, places=3)


    def test_no_miners(self):
        """
        Test that the function returns an empty list when there are no miners
        """
        returns = {}
        filtered_results = Scoring.filter_results(returns)
        self.assertEqual(filtered_results, [])

    def test_one_miner(self):
        """
        Test that the function returns the same list when there is only one miner
        """
        returns = {'miner0': 0.9}
        filtered_results = Scoring.filter_results(returns)
        filtered_netuids = [ x[0] for x in filtered_results ]

        self.assertEqual(filtered_results, [('miner0', 1.0)])
        self.assertEqual(filtered_netuids, ['miner0'])




