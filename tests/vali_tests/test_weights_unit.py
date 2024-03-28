# developer: trdougherty
# Copyright © 2024 Taoshi Inc
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring
from vali_config import ValiConfig

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

        n_returns = np.random.randint(5, 100, n_miners)
        miner_returns = []
        for n_return in list(n_returns):
            returns = np.random.uniform(
                low=0.9,
                high=1.5,
                size=n_return
            ).tolist()
            miner_returns.append(returns)

        returns = dict(zip(miner_names, miner_returns))
        returns['miner0'] = [0.7,0.8,0.7,0.6,0.9] # this is going to be the outlier - the one which should get filtered if filtering happens

        self.returns: list[str, list[float]] = list(returns.items())

    def test_miner_scoring_no_miners(self):
        """
        Test that the miner filtering works as expected when there are no miners
        """
        returns = []
        filtered_results = Scoring.transform_and_scale_results(returns)
        self.assertEqual(filtered_results, [])

    def test_miner_scoring_one_miner(self):
        """
        Test that the miner filtering works as expected when there is only one miner
        """
        returns = [('miner0', 1.1)]
        filtered_results = Scoring.transform_and_scale_results(returns)
        filtered_netuids = [ x[0] for x in filtered_results ]
        filtered_values = [ x[1] for x in filtered_results ]

        original_netuids = [ x[0] for x in returns ]
        original_values = [ x[1] for x in returns ]

        self.assertEqual(sorted(filtered_netuids), sorted(original_netuids))
        self.assertNotEqual(sorted(filtered_values), sorted(original_values))
        self.assertEqual(filtered_results, [('miner0', 1.0)])

    def test_transform_and_scale_results_defaults(self):
        """Test that the transform and scale results works as expected"""
        scaled_transformed_list: list[tuple[str, float]] = Scoring.transform_and_scale_results(self.returns)

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

    def test_positive_omega(self):
        """Test that the omega function works as expected for only positive returns"""
        sample_returns = [1.4, 1.1, 1.2, 1.3, 1.4, 1.2]
        risk_free_rate = ValiConfig.LOOKBACK_RANGE_DAYS_RISK_FREE_RATE

        sample_returns = [ x + risk_free_rate for x in sample_returns ]
        omega_positive = Scoring.omega(sample_returns)

        ## omega_minimum_denominator should kick in
        omega_minimum_denominator = ValiConfig.OMEGA_MINIMUM_DENOMINATOR
        no_loss_benefit = 1 / omega_minimum_denominator

        self.assertGreaterEqual(omega_positive, no_loss_benefit)

    def test_negative_omega(self):
        """Test that the omega function works as expected for all negative returns"""
        sample_returns = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

        risk_free_rate = ValiConfig.LOOKBACK_RANGE_DAYS_RISK_FREE_RATE

        sample_returns = [ x - risk_free_rate for x in sample_returns ]
        omega_negative = Scoring.omega(sample_returns)

        # sum above threshold should be 0
        self.assertEqual(omega_negative, 0.0)

    def test_omega(self):
        """Test that the omega function works as expected"""
        sample_returns = [0.9, 0.7, 1.1, 1.2, 1.3, 1.4, 1.2]

        ## returns - ( 1 + threshold ) -> we're ignoring threshold for internal calculations
        ## positive returns should be [ 1.1, 1.2, 1.3, 1.4, 1.2 ] -> [ 0.1, 0.2, 0.3, 0.4, 0.2 ]
        ## negative returns should be [ 0.9, 0.7 ] -> [ -0.1, -0.3 ]

        positive_sum = sum([ 0.1, 0.2, 0.3, 0.4, 0.2 ])
        negative_sum = sum([ -0.1, -0.3 ])
        hand_computed_omega = positive_sum / abs(negative_sum)

        ## omega should be [ 1.1 + 1.2 + 1.3 + 1.4 + 1.2 ] / [ 0.9 + 0.7 ]
        omega = Scoring.omega(sample_returns)
        self.assertEqual(omega, hand_computed_omega)

    def test_omega_zero_length_returns(self):
        """Test that the omega function works as expected with zero length returns"""
        sample_returns = []
        omega = Scoring.omega(sample_returns)

        self.assertEqual(omega, 0.0)

    def test_total_return(self):
        """Test that the total return function works as expected"""
        sample_returns = [0.9, 0.7, 1.1, 1.2, 1.3, 1.4, 1.2]
        ## hand computed total return
        hand_computed_total_return = 0.9 * 0.7 * 1.1 * 1.2 * 1.3 * 1.4 * 1.2
        self.assertAlmostEqual(hand_computed_total_return, 1.816, places=3)

        total_return = Scoring.total_return(sample_returns)
        self.assertAlmostEqual(total_return, hand_computed_total_return, places=3)

    def test_total_return_zero_length_returns(self):
        """Test that the total return function works as expected with zero length returns"""
        sample_returns = []
        total_return = Scoring.total_return(sample_returns)

        self.assertEqual(total_return, 0.0)

    def test_transform_and_scale_results_empty_returns_one(self):
        """Test that the transform and scale results works as expected with empty returns"""
        sample_returns = [
            ('miner0', []),
            ('miner1', [1.5, 0.5, 1.4, 0.6, 1.4]), # yolo miner
            ('miner2', [1.01, 1.00, 1.02, 1.01, 0.99]), # decent miner
            ('miner3', [1.03, 1.03, 1.01, 1.04, 1.01]), # consistently great miner
        ]

        scaled_transformed_list = Scoring.transform_and_scale_results(sample_returns)
        transformed_minernames = [ x[0] for x in scaled_transformed_list ]

        self.assertEqual(
            transformed_minernames, 
            ['miner3', 'miner2', 'miner1']
        )

    def test_transform_and_scale_results(self):
        sample_returns = [
            ('miner0', [1.15, 0.95, 1.20, 0.90, 1.10, 1.05, 0.85]),
            ('miner1', [1.05, 1.02, 0.98, 1.03, 1.01, 0.99, 1.04, 1.00, 1.02, 0.97]),
        ]

        ## miner1 has a higher return, but lower omega
        m0_return = Scoring.total_return(sample_returns[0][1])
        m1_return = Scoring.total_return(sample_returns[1][1])
        self.assertGreater(
            m0_return, 
            m1_return
        )

        m0_omega = Scoring.omega(sample_returns[0][1])
        m1_omega = Scoring.omega(sample_returns[1][1])
        self.assertGreater(
            m1_omega, 
            m0_omega
        )

        ## the transformation should be some kind of average of the two
        scaled_transformed_list = Scoring.transform_and_scale_results(sample_returns)

        ## if omega is being prioritized, then miner1 should have the higer result
        transformed_minernames = [ x[0] for x in scaled_transformed_list ]

        ## miner 1 should be listed first
        self.assertEqual(transformed_minernames, ['miner1', 'miner0'])

        ## the score (miner 1) should be higher than the score (miner 0)
        transformed_minervalues = [ x[1] for x in scaled_transformed_list ]
        self.assertGreater(transformed_minervalues[0], transformed_minervalues[1])






