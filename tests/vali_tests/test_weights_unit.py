# developer: trdougherty
# Copyright © 2024 Taoshi Inc
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring
from vali_objects.position import Position
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter

from vali_config import TradePair

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
        self.subtensor_weight_setter = SubtensorWeightSetter(
            config=None,
            wallet=None,
            metagraph=None,
            running_unit_tests=True
        )

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
        self.assertAlmostEqual(sum(values), 1.0, places=3)

    def test_positive_omega(self):
        """Test that the omega function works as expected for only positive returns"""
        sample_returns = [1.4, 1.1, 1.2, 1.3, 1.4, 1.2]
        risk_free_rate = ValiConfig.OMEGA_LOG_RATIO_THRESHOLD

        sample_returns = [ np.log(x) - risk_free_rate for x in sample_returns ]
        omega_positive = Scoring.omega(sample_returns)

        ## omega_minimum_denominator should kick in
        omega_minimum_denominator = ValiConfig.OMEGA_MINIMUM_DENOMINATOR
        no_loss_benefit = 1 / omega_minimum_denominator

        self.assertGreaterEqual(omega_positive, no_loss_benefit)

    def test_negative_omega(self):
        """Test that the omega function works as expected for all negative returns"""
        sample_returns = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

        sample_returns = [ np.log(x) for x in sample_returns ]
        omega_negative = Scoring.omega(sample_returns)

        # sum above threshold should be 0
        self.assertEqual(omega_negative, 0.0)

    def test_omega(self):
        """Test that the omega function works as expected"""
        sample_returns = [0.9, 0.7, 1.1, 1.2, 1.3, 1.4, 1.2]

        ## returns - ( 1 + threshold ) -> we're ignoring threshold for internal calculations
        ## positive returns should be [ 1.1, 1.2, 1.3, 1.4, 1.2 ] -> [ 0.1, 0.2, 0.3, 0.4, 0.2 ]
        ## negative returns should be [ 0.9, 0.7 ] -> [ -0.1, -0.3 ]

        positive_sum = np.sum(np.log(np.array([ 1.1, 1.2, 1.3, 1.4, 1.2 ])))
        negative_sum = np.sum(np.log(np.array([ 1-0.1, 1-0.3 ])))
        hand_computed_omega = positive_sum / abs(negative_sum)

        ## omega should be [ 1.1 + 1.2 + 1.3 + 1.4 + 1.2 ] / [ 0.9 + 0.7 ]
        omega = Scoring.omega([ np.log(x) for x in sample_returns ])
        self.assertEqual(omega, hand_computed_omega)

    def test_omega_zero_length_returns(self):
        """Test that the omega function works as expected with zero length returns"""
        sample_returns = []
        omega = Scoring.omega(sample_returns)

        self.assertEqual(omega, 0.0)

    def test_total_return(self):
        """Test that the total return function works as expected"""
        sample_returns = [ np.log(x) for x in [0.9, 0.7, 1.1, 1.2, 1.3, 1.4, 1.2] ]
        ## hand computed total return
        hand_computed_total_return = 0.9 * 0.7 * 1.1 * 1.2 * 1.3 * 1.4 * 1.2
        self.assertAlmostEqual(hand_computed_total_return, 1.816, places=3)

        total_return = Scoring.total_return(sample_returns)
        self.assertAlmostEqual(total_return, hand_computed_total_return, places=3)

    def test_positive_sharp_ratio(self):
        """Test that the sharp ratio function works as expected for only positive returns"""
        sample_returns = [1.4, 1.1, 1.2, 1.3, 1.4, 1.2]
        sharp_ratio_positive = Scoring.sharpe_ratio(sample_returns)

        self.assertGreater(sharp_ratio_positive, 0.0)

    def test_negative_sharp_ratio(self):
        """Test that the sharp ratio function works as expected for all negative returns"""
        sample_returns = [ np.log(x) for x in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4] ]
        sharp_ratio_negative = Scoring.sharpe_ratio(sample_returns)

        self.assertLess(sharp_ratio_negative, 1.0)

    def test_sharp_zero_length_returns(self):
        """Test that the sharp ratio function works as expected with zero length returns"""
        sample_returns = []
        sharp_ratio = Scoring.sharpe_ratio(sample_returns)

        self.assertEqual(sharp_ratio, 0.0)

    def test_sharpe_ratio(self):
        """Test that the sharpe ratio function works as expected"""
        sample_returns = [ np.log(x) for x in [0.9, 0.7, 1.1, 1.2, 1.3, 1.4, 1.2] ]

        ## should be the product of returns over the std dev.
        threshold = np.log(1 + ValiConfig.PROBABILISTIC_LOG_SHARPE_RATIO_THRESHOLD)
        hand_sharpe_log = np.mean([ x - threshold for x in sample_returns ]) / np.std(sample_returns)
        hand_sharpe = np.exp(hand_sharpe_log)

        sharpe_ratio = Scoring.sharpe_ratio(sample_returns)
        self.assertAlmostEqual(sharpe_ratio, hand_sharpe, places=3)

    def test_total_return_zero_length_returns(self):
        """Test that the total return function works as expected with zero length returns"""
        sample_returns = []
        total_return = Scoring.total_return(sample_returns)

        self.assertEqual(total_return, 0.0)

    def test_transform_and_scale_results_empty_returns_one(self):
        """Test that the transform and scale results works as expected with empty returns"""
        sample_returns = [
            ('miner0', []),
            ('miner1', [ np.log(x) for x in [1.5, 0.5, 1.1, 0.6, 1.2] ]), # yolo miner
            ('miner2', [ np.log(x) for x in [1.01, 1.00, 1.02, 1.01, 0.99] ]), # decent miner
            ('miner3', [ np.log(x) for x in [1.03, 1.03, 1.01, 1.04, 1.01] ]), # consistently great miner
        ]

        scaled_transformed_list = Scoring.transform_and_scale_results(sample_returns)
        transformed_minernames = [ x[0] for x in scaled_transformed_list ]

        self.assertEqual(
            transformed_minernames, 
            ['miner1', 'miner2', 'miner3'] 
        )

    def test_transform_and_scale_results(self):
        sample_returns = [
            ('miner0', [ np.log(x) for x in [1.15, 0.95, 1.20, 0.90, 1.10, 1.05, 0.85] ]),
            ('miner1', [ np.log(x) for x in [1.05, 1.02, 0.98, 1.03, 1.01, 0.99, 1.04, 1.00, 1.02, 0.97] ]),
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

    def test_transform_and_scale_results_grace_period(self):
        """Test that the transform and scale results works as expected with a grace period"""
        sample_returns = [
            ('miner0', [ np.log(x) for x in [1.15, 0.95, 1.20, 0.90, 1.10, 1.05, 0.85] ]), # grace period miner, should return almost 0
            ('miner1', [ np.log(x) for x in [1.05, 1.02, 0.98, 1.03, 1.01, 0.99, 1.04, 1.00, 1.02, 0.97, 1.04] ]),
        ]

        ## the transformation should be some kind of average of the two
        scaled_transformed_list = Scoring.transform_and_scale_results(sample_returns)

        minimum_miner_benefit = ValiConfig.SET_WEIGHT_MINER_GRACE_PERIOD_VALUE
        
        miner_scores_dict = dict(scaled_transformed_list)
        graceperiod_miner_score = miner_scores_dict['miner0']

        self.assertGreater(graceperiod_miner_score, 0)
        self.assertGreaterEqual(graceperiod_miner_score, minimum_miner_benefit)

        self.assertGreater(miner_scores_dict['miner1'], miner_scores_dict['miner0'])

    def test_transform_and_scale_results_grace_period_no_positions(self):
        """Test that the transform and scale results works as expected with a grace period"""
        sample_returns = [
            ('miner0', []),
            ('miner1', [ np.log(x) for x in [1.05, 1.02, 0.98, 1.03, 1.01, 0.99, 1.04, 1.00, 1.02, 0.97, 1.04] ]),
            ('miner2', [ np.log(x) for x in [1.05, 1.02, 0.98, 1.03, 1.01, 0.99, 1.04, 1.00, 1.02, 0.97, 1.08] ]),
        ]

        ## the transformation should be some kind of average of the two
        scaled_transformed_list = Scoring.transform_and_scale_results(sample_returns)        
        miner_scores_dict = dict(scaled_transformed_list)

        # should not exist in the returned dict
        graceperiod_miner_score = miner_scores_dict.get('miner0', None)

        self.assertEqual(graceperiod_miner_score, None)

    def test_miner_filter_graceperiod(self):
        """Test that the miner filter function works as expected"""

        # this should be around 2.59e9 for one month
        current_time = int(2.5e9)

        minimum_positions = ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS
        minimum_positions = max(minimum_positions - 1, 0)

        time_partition = np.linspace(0, 2.5, minimum_positions) # want one less than the minimum
        opened_ms_list = [ 0 for _ in time_partition ]
        closed_ms_list = [ int(x * 1e9) for x in time_partition ]

        example_miner: list[Position] = [
            Position(
                miner_hotkey='miner0',
                position_uuid=str(i),
                open_ms=opened_ms_list[i],
                close_ms=closed_ms_list[i],
                trade_pair=TradePair.BTCUSD,
                orders=[],
                current_return=1.0,
                return_at_close=1.0,
                net_leverage=0.0,
                average_entry_price=0.0,
                initial_entry_price=0.0,
                position_type=None,
                is_closed_position=True
            )
            for i in range(len(time_partition))
        ]

        ## should be filtered out
        filter_miner_logic = self.subtensor_weight_setter._filter_miner(example_miner, current_time)
        self.assertFalse(filter_miner_logic)

    def test_miner_filter_older_than_graceperiod(self):
        """Test that the miner filter function works as expected"""

        # this should be around 2.59e9 for one month
        current_time = int(8.1e9) # one month prior should be around 5.5

        # some of these open positions should be older than the grace period
        minimum_positions = ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS
        minimum_positions = max(minimum_positions - 1, 0)

        time_partition = np.linspace(4, 8, minimum_positions)
        opened_ms_list = [ 0 for _ in time_partition ]
        closed_ms_list = [ int(x * 1e9) for x in time_partition ]

        example_miner: list[Position] = [
            Position(
                miner_hotkey='miner0',
                position_uuid=str(i),
                open_ms=opened_ms_list[i],
                close_ms=closed_ms_list[i],
                trade_pair=TradePair.BTCUSD,
                orders=[],
                current_return=1.0,
                return_at_close=1.0,
                net_leverage=0.0,
                average_entry_price=0.0,
                initial_entry_price=0.0,
                position_type=None,
                is_closed_position=True
            )
            for i in range(len(time_partition))
        ]

        ## should be filtered out
        filter_miner_logic = self.subtensor_weight_setter._filter_miner(example_miner, current_time)
        self.assertTrue(filter_miner_logic)


    def test_miner_filter_graceperiod_no_positions(self):
        """Test that the miner filter function works as expected with no positions"""
        current_time = int(4.1e9)

        example_miner = []

        ## should be filtered out
        filter_miner_logic = self.subtensor_weight_setter._filter_miner(example_miner, current_time)
        self.assertTrue(filter_miner_logic)





