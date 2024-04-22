# developer: trdougherty
# Copyright © 2024 Taoshi Inc
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring
from vali_objects.position import Position
from vali_objects.utils.position_utils import PositionUtils

from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter

from vali_config import TradePair
from vali_config import ValiConfig

import numpy as np
import random

from tests.shared_objects.test_utilities import get_time_in_range, order_generator, position_generator

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
            ('miner1', [ np.log(x) for x in [1.0001, 0.9999, 1.002, 1.0002, 1.4, 0.999, 1.0001, 1.0001, 1.001, 0.999] ]), # yolo miner
            ('miner2', [ np.log(x) for x in [1.01, 1.00, 1.02, 1.01, 0.99, 1.05, 1.04, 1.03, 1.06, 0.98] ]), # decent miner
            ('miner3', [ np.log(x) for x in [1.03, 1.03, 1.01, 1.04, 1.01, 1.05, 1.08, 1.05, 1.03, 1.08] ]), # consistently great miner
        ]

        scaled_transformed_list = Scoring.transform_and_scale_results(sample_returns)
        transformed_minernames = [ x[0] for x in scaled_transformed_list ]

        self.assertEqual(
            transformed_minernames, 
            ['miner3','miner1', 'miner2'] 
        )

    def test_transform_and_scale_results(self):
        sample_returns = [
            ('miner0', [ np.log(x) for x in [1.15, 0.95, 1.20, 0.90, 1.10, 1.05, 0.85, 0.88, 0.92, 1.2] ]),
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

        ## miner 1 should be listed first if omega is prioritized, otherwise miner 0 should be listed first
        self.assertEqual(transformed_minernames, ['miner0', 'miner1'])

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
        miner_scores_hotkeys = [ x[0] for x in scaled_transformed_list ]

        self.assertNotIn('miner0', miner_scores_hotkeys)

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

    def test_miner_filter_challengeperiod(self):
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
                net_leverage=0.1,
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

    def test_miner_filter_older_than_challengeperiod(self):
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
                net_leverage=0.1,
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


    def test_miner_filter_challengeperiod_no_positions(self):
        """Test that the miner filter function works as expected with no positions"""
        current_time = int(4.1e9)

        example_miner = []

        ## should be filtered out
        filter_miner_logic = self.subtensor_weight_setter._filter_miner(example_miner, current_time)
        self.assertTrue(filter_miner_logic)


    def test_challengeperiod_screening_onepass(self):
        """Test that challengeperiod screening passes all miners who meet the criteria"""
        ## some sample positions and their orders, want to make sure we return

        # criteria for passing
        # 1. total time duration
        # 2. total number of positions
        # 3. average leverage
        # 4. total return
        # 5. sharpe

        start_time = 0
        end_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS - 10
        n_positions = 20
        order_leverage = 1.5

        ## we are going to have a sample miner who is incredible
        start_times = sorted([ get_time_in_range(x, start_time, end_time) for x in np.random.rand(n_positions) ])
        end_times = sorted([ get_time_in_range(x, start_times[c], end_time) for c,x in enumerate(np.random.rand(n_positions)) ])

        # to make it simple, each of the positions will only have two orders, open and close with the same time as the position open and close

        order_opens = []
        order_closes = []

        for i in range(len(start_times)):
            order_opens.append(
                order_generator(
                    order_type=OrderType.LONG,
                    processed_ms=start_times[i],
                    leverage=order_leverage,
                    n_orders=1
                )[0]
            )
            order_closes.append(
                order_generator(
                    order_type=OrderType.FLAT,
                    processed_ms=end_times[i],
                    leverage=0.1,
                    n_orders=1
                )[0]
            )

        # each postions has str
        positions = []
        for i in range(len(order_opens)):
            sample_position = position_generator(
                open_time_ms=start_times[i],
                close_time_ms=end_times[i],
                trade_pair=TradePair.BTCUSD,
                orders=[ order_opens[i], order_closes[i] ],
                return_at_close=1.05
            )
            positions.append(sample_position)

        current_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS + 100
        chellengeperiod_logic = self.subtensor_weight_setter._challengeperiod_check(
            positions,
            current_time
        )

        self.assertEqual(chellengeperiod_logic, True)

    def test_challengeperiod_screening_challengeperiod_onechallenge(self):
        start_time = 0
        end_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS / 2

        # not enough positions to pass
        n_positions = 2
        order_leverage = 1.5

        ## we are going to have a sample miner who is incredible
        start_times = sorted([ get_time_in_range(x, start_time, end_time) for x in np.random.rand(n_positions) ])
        end_times = sorted([ get_time_in_range(x, start_times[c], end_time) for c,x in enumerate(np.random.rand(n_positions)) ])

        # to make it simple, each of the positions will only have two orders, open and close with the same time as the position open and close

        order_opens = []
        order_closes = []

        for i in range(len(start_times)):
            order_opens.append(
                order_generator(
                    order_type=OrderType.LONG,
                    leverage=order_leverage,
                    n_orders=1
                )[0]
            )
            order_closes.append(
                order_generator(
                    order_type=OrderType.FLAT,
                    leverage=0.1,
                    n_orders=1
                )[0]
            )

        # each postions has str
        positions = []
        for i in range(len(order_opens)):
            sample_position = position_generator(
                miner_hotkey='miner0',
                open_time_ms=start_times[i],
                close_time_ms=end_times[i],
                trade_pair=TradePair.BTCUSD,
                orders=[ order_opens[i], order_closes[i] ],
                return_at_close=1.05
            )
            positions.append(sample_position)

        current_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS + 100
        chellengeperiod_logic = self.subtensor_weight_setter._challengeperiod_check(
            positions,
            current_time
        )

        self.assertEqual(chellengeperiod_logic, None)

    def test_challengeperiod_screening_challengeperiod_one_elimination(self):
        start_time = 0
        end_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS + 100

        # not enough positions to pass
        n_positions = 4
        order_leverage = 1.5

        ## we are going to have a sample miner who is incredible
        start_times = sorted([ start_time for x in np.random.rand(n_positions) ])
        end_times = sorted([ get_time_in_range(x, start_times[c], end_time) for c,x in enumerate(np.random.rand(n_positions)) ])

        # to make it simple, each of the positions will only have two orders, open and close with the same time as the position open and close

        order_opens = []
        order_closes = []

        for i in range(len(start_times)):
            order_opens.append(
                order_generator(
                    order_type=OrderType.LONG,
                    processed_ms=start_times[i],
                    leverage=order_leverage,
                    n_orders=1
                )[0]
            )
            order_closes.append(
                order_generator(
                    order_type=OrderType.FLAT,
                    processed_ms=end_times[i],
                    leverage=0.1,
                    n_orders=1
                )[0]
            )
    
        # each postions has str
        positions = []
        for i in range(n_positions):
            sample_position = position_generator(
                miner_hotkey='miner0',
                open_time_ms=start_times[i],
                close_time_ms=end_times[i],
                trade_pair=TradePair.BTCUSD,
                orders=[ order_opens[i], order_closes[i] ],
                return_at_close=1.05
            )
            positions.append(sample_position)

        current_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS + 100
        chellengeperiod_logic = self.subtensor_weight_setter._challengeperiod_check(
            positions,
            current_time
        )

        self.assertEqual(chellengeperiod_logic, False)

    def test_challengeperiod_screening_challengeperiod_set_miners(self):
        start_time = 0
        end_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS * (9 / 10)
        current_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS + 100

        # not enough positions to pass
        n_positions = 20
        order_leverage = 1.1
        return_at_close = 1.05

        ## we are going to have a sample miner who is incredible
        start_times = sorted([ get_time_in_range(x, start_time, end_time) for x in np.random.rand(n_positions) ])
        end_times = sorted([ get_time_in_range(x, start_times[c], end_time) for c,x in enumerate(np.random.rand(n_positions)) ])

        # to make it simple, each of the positions will only have two orders, open and close with the same time as the position open and close

        order_opens = []
        order_closes = []

        for i in range(len(start_times)):
            order_opens.append(
                order_generator(
                    order_type=OrderType.LONG,
                    processed_ms=start_times[i],
                    leverage=order_leverage,
                    n_orders=1
                )[0]
            )
            order_closes.append(
                order_generator(
                    order_type=OrderType.FLAT,
                    processed_ms=end_times[i],
                    leverage=0.1,
                    n_orders=1
                )[0]
            )

        # each postions has str
        positions = []
        for i in range(len(order_opens)):
            sample_position = position_generator(
                miner_hotkey='miner0',
                open_time_ms=start_times[i],
                close_time_ms=end_times[i],
                trade_pair=TradePair.BTCUSD,
                orders=[ order_opens[i], order_closes[i] ],
                return_at_close=return_at_close
            )
            positions.append(sample_position)


        challengeperiod_logic = self.subtensor_weight_setter._challengeperiod_returns_logic(positions)
        self.assertEqual(challengeperiod_logic, True)

    def test_total_position_duration(self):
        """Test that the total position duration function works as expected"""
        start_time = 0
        end_time = 1000
        n_positions = 4

        start_times = sorted([ get_time_in_range(x, start_time, end_time) for x in [0.2,0.4,0.6,0.7] ])
        end_times = sorted([ get_time_in_range(x, start_times[c], end_time) for c,x in enumerate([0.3,0.3,0.3,0.3]) ])

        # position start and end times
        # 1. 200 to 200 + (1000-200)*0.3 = 200 to 240
        # 2. 400 to 400 + (1000-400)*0.3 = 400 to 580
        # 3. 600 to 600 + (1000-600)*0.3 = 600 to 720
        # 4. 700 to 700 + (1000-700)*0.3 = 700 to 790

        total_handcomputed_time = (440 - 200) + (580 - 400) + (720 - 600) + (790 - 700)
        positions = [
            position_generator(
                open_time_ms=start_times[i],
                close_time_ms=end_times[i],
                trade_pair=TradePair.BTCUSD,
                return_at_close=1.0,
                orders=[]
            )
            for i in range(n_positions)
        ]

        total_position_duration = PositionUtils.compute_total_position_duration(positions)
        self.assertEqual(total_position_duration, total_handcomputed_time)

    def test_average_leverage(self):
        """Test that the average leverage function is working correctly"""
        start_time = 0
        end_time = 1000
        n_positions = 4

        start_times = [start_time] * n_positions

        # each of the end times will be 1/4th of the way through the total time, so the final order will have the most influence in the leverage time calculation
        end_times = [ end_time * (i / n_positions)**2 for i in range(1, n_positions + 1) ]
        leverages = [ 1.1, 0.2, 0.5, 0.2 ] # these are modifications as if new orders -> so 1.1 to 1.3 to 1.8

        hand_computed_average_leverage = 0
        total_time = 0

        time_leverages = []
        deltas = []

        for i in range(1, n_positions):
            time_delta = end_times[i] - end_times[i-1]
            deltas.append(time_delta)
            total_time += time_delta
            running_leverage = sum(leverages[:i])
            time_leverage = running_leverage * (time_delta)
            time_leverages.append(time_leverage)
            hand_computed_average_leverage += time_leverage

        hand_computed_average_leverage /= total_time

        orders = [ order_generator(
                processed_ms=end_times[i],
                leverage=leverages[i],
                n_orders=1
            )[0] for i in range(n_positions) ]

        positions = [
            position_generator(
                open_time_ms=start_times[i],
                close_time_ms=end_times[i],
                trade_pair=TradePair.BTCUSD,
                return_at_close=1.0,
                orders=orders
            )
            for i in range(n_positions)
        ]

        average_leverage_calculation = PositionUtils.compute_average_leverage(positions)
        self.assertAlmostEqual(average_leverage_calculation, hand_computed_average_leverage, places=3)

    def test_yolo_challenge_fail(self):
        """Test that hte yolo miner fails the challenge as expected"""
        start_time = 0
        end_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS - 10

        # enough positions to pass
        n_positions = 19
        yolo_leverage = 20
        small_leverage = 0.01

        ## we are going to have a sample miner who is incredible
        start_times = sorted([ get_time_in_range(x, start_time, end_time) for x in np.random.rand(n_positions) ])
        end_times = sorted([ get_time_in_range(x, start_times[c], end_time) for c,x in enumerate(np.random.rand(n_positions)) ])
        random_returns = (np.random.rand(n_positions) - 0.5) / 100

        # to make it simple, each of the positions will only have two orders, open and close with the same time as the position open and close

        order_opens = [ 
            order_generator(
                processed_ms=x,
                leverage=small_leverage,
                n_orders=1
            )[0] for x in start_times 
        ]

        order_closes = [ 
            order_generator(
                processed_ms=x,
                leverage=0.1,
                n_orders=1
            )[0] for x in end_times 
        ]

        # each postions has str
        tiny_positions = []
        for i in range(len(order_opens)):
            sample_position = position_generator(
                miner_hotkey='miner0',
                open_time_ms=start_times[i],
                close_time_ms=end_times[i],
                trade_pair=TradePair.BTCUSD,
                orders=[ order_opens[i], order_closes[i] ],
                return_at_close=random_returns[i]
            )
            tiny_positions.append(sample_position)

        yolo_order_start = order_generator(
            processed_ms=start_time,
            leverage=yolo_leverage,
            n_orders=1
        )[0]

        yolo_order_end = order_generator(
            processed_ms=end_time,
            leverage=1.0,
            n_orders=1
        )[0]

        yolo_position = position_generator(
            miner_hotkey='miner0',
            open_time_ms=start_time,
            close_time_ms=end_time,
            trade_pair=TradePair.BTCUSD,
            orders=[ yolo_order_start, yolo_order_end ],
            return_at_close=1.3 # some crazy positive return
        )

        all_positions = tiny_positions + [ yolo_position ]

        # want to make sure that we are now running evaluation
        current_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS + 100
        chellengeperiod_logic = self.subtensor_weight_setter._challengeperiod_check(
            all_positions,
            current_time
        )

        self.assertEqual(chellengeperiod_logic, False)

    




