# developer: trdougherty
import math
import numpy as np
import random
import copy
import bittensor as bt

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring
from vali_objects.position import Position
from vali_objects.utils.position_utils import PositionUtils

from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger

from vali_config import TradePair
from vali_config import ValiConfig

from tests.shared_objects.test_utilities import get_time_in_range, order_generator, position_generator, ledger_generator, checkpoint_generator

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
        start_time = 0
        end_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS

        self.start_time = start_time
        self.end_time = end_time

        ## each of the miners should be hitting the filtering criteria except for the bad miner
        ledger_dict = {}
        for c,n_return in enumerate(list(n_returns)):
            checkpoint_times = np.linspace(start_time, end_time, n_return, dtype=int).tolist()
            position_time_accumulation = np.diff(checkpoint_times, prepend=0)
            miner_checkpoints = []
            gains = np.random.uniform(
                low=0.0,
                high=0.05,
                size=n_return
            ).tolist()
            losses = np.random.uniform(
                low=-0.05,
                high=0.0,
                size=n_return
            ).tolist()

            for i in range(n_return):
                miner_checkpoints.append(
                    PerfCheckpoint(
                        last_update_ms=checkpoint_times[i],
                        gain=gains[i],
                        loss=losses[i],
                        prev_portfolio_ret=1.0,
                        open_ms=position_time_accumulation[i]
                    )
                )

            ledger_dict[miner_names[c]] = ledger_generator(checkpoints=miner_checkpoints)

        ## Losing miner
        bad_checkpoint_list = []
        n_bad_checkpoints = 5
        bad_checkpoint_times = np.linspace(start_time, end_time, n_bad_checkpoints, dtype=int).tolist()
        bad_checkpoint_time_accumulation = np.diff(bad_checkpoint_times, prepend=0)
        for i in range(n_bad_checkpoints):
            bad_checkpoint_list.append(
                PerfCheckpoint(
                    last_update_ms=bad_checkpoint_times[i],
                    gain=0.0,
                    loss=-0.2,
                    prev_portfolio_ret=1.0,
                    open_ms=bad_checkpoint_time_accumulation[i]
                )
            )

        ledger_dict['bad'] = ledger_generator(
            checkpoints=bad_checkpoint_list
        )

        ## Winning miner
        good_checkpoint_list = []
        n_good_checkpoints = 5
        good_checkpoint_times = np.linspace(start_time, end_time, n_good_checkpoints, dtype=int).tolist()
        good_checkpoint_time_accumulation = np.diff(good_checkpoint_times, prepend=0)
        for i in range(n_good_checkpoints):
            good_checkpoint_list.append(
                PerfCheckpoint(
                    last_update_ms=good_checkpoint_times[i],
                    gain=0.2,
                    loss=-0.001,
                    prev_portfolio_ret=1.0,
                    open_ms=good_checkpoint_time_accumulation[i]
                )
            )
        
        ledger_dict['good'] = ledger_generator(
            checkpoints=good_checkpoint_list
        )


        ## for both of the swings miner, we want the positive to be a proportion higher than the negative
        value_augment = 1.1

        ## High Swings Miner
        high_swing_checkpoint_list = []

        highswings_value = 0.3
        n_high_swing_checkpoints = 15
        high_swing_checkpoint_times = np.linspace(start_time, end_time, n_high_swing_checkpoints, dtype=int).tolist()
        high_swing_checkpoint_time_accumulation = np.diff(high_swing_checkpoint_times, prepend=0)
        for i in range(n_high_swing_checkpoints):
            high_swing_checkpoint_list.append(
                PerfCheckpoint(
                    last_update_ms=high_swing_checkpoint_times[i],
                    gain=highswings_value*value_augment,
                    loss=-highswings_value,
                    prev_portfolio_ret=1.0,
                    open_ms=high_swing_checkpoint_time_accumulation[i]
                )
            )
        
        ledger_dict['highswings'] = ledger_generator(
            checkpoints=high_swing_checkpoint_list
        )

        ## Low Swings Miner
        low_swing_checkpoint_list = []
        lowswings_value = 0.001
        n_low_swing_checkpoints = 15
        low_swing_checkpoint_times = np.linspace(start_time, end_time, n_low_swing_checkpoints, dtype=int).tolist()
        low_swing_checkpoint_time_accumulation = np.diff(low_swing_checkpoint_times, prepend=0)
        for i in range(n_low_swing_checkpoints):
            low_swing_checkpoint_list.append(
                PerfCheckpoint(
                    last_update_ms=low_swing_checkpoint_times[i],
                    gain=lowswings_value*value_augment,
                    loss=-lowswings_value,
                    prev_portfolio_ret=1.0,
                    open_ms=low_swing_checkpoint_time_accumulation[i]
                )
            )

        ledger_dict['lowswings'] = ledger_generator(
            checkpoints=low_swing_checkpoint_list
        )

        ## also want to mock the scenario where two identical miners diverge on a position - the miner who makes the additional transaction should benefit

        highswings_copy = copy.deepcopy(high_swing_checkpoint_list)
        highswings_copy[-1].gain = highswings_copy[-1].gain * 1.1
        highswings_copy[-1].loss = highswings_copy[-1].loss
        ledger_dict['highswingsactive'] = ledger_generator(
            checkpoints=highswings_copy
        )

        highswings_inactive_copy = copy.deepcopy(high_swing_checkpoint_list)
        highswings_inactive_copy[-1].gain = 0.0
        highswings_inactive_copy[-1].loss = 0.0
        highswings_inactive_copy[-1].open_ms = 0
        ledger_dict['highswingsinactive'] = ledger_generator(
            checkpoints=highswings_inactive_copy
        )

        lowswings_copy = copy.deepcopy(low_swing_checkpoint_list)
        lowswings_copy[-1].gain = lowswings_copy[-1].gain * 1.1
        lowswings_copy[-1].loss = lowswings_copy[-1].loss
        ledger_dict['lowswingsactive'] = ledger_generator(
            checkpoints=lowswings_copy
        )

        lowswings_inactive_copy = copy.deepcopy(low_swing_checkpoint_list)
        lowswings_inactive_copy[-1].gain = 0.0
        lowswings_inactive_copy[-1].loss = 0.0
        lowswings_inactive_copy[-1].open_ms = 0
        ledger_dict['lowswingsinactive'] = ledger_generator(
            checkpoints=lowswings_inactive_copy
        )

        ## not enough time to be considered, should be filtered
        time_constrained_checkpoint_list = []
        n_time_constrained = 5

        time_constrained_end = max(ValiConfig.SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS - 100, 100)
        time_constrained_times = np.linspace(start_time, time_constrained_end, n_time_constrained, dtype=int).tolist()
        time_constrained_time_accumulation = np.diff(time_constrained_times, prepend=0)

        for i in range(n_time_constrained):
            time_constrained_checkpoint_list.append(
                PerfCheckpoint(
                    last_update_ms=time_constrained_times[i],
                    gain=0.1,
                    loss=-0.05,
                    prev_portfolio_ret=1.0,
                    open_ms=time_constrained_time_accumulation[i]
                )
            )

        ledger_dict['timeconstrained'] = ledger_generator(
            checkpoints=time_constrained_checkpoint_list
        )

        ## no positions miner
        ledger_dict['nopositions'] = ledger_generator(checkpoints=[])

        ## now to test scenarios to make sure the variable decay is working properly on the ledger
        ## miner whose consistency decreases over time
        n_good_checkpoints = 20
        good_checkpoint_times = np.linspace(start_time, end_time, n_good_checkpoints, dtype=int).tolist()
        good_checkpoint_time_accumulation = np.diff(good_checkpoint_times, prepend=0)

        increasing_gains = np.linspace(0.1, 0.2, n_good_checkpoints).tolist()
        increasing_losses = np.linspace(-0.09, -0.19, n_good_checkpoints).tolist()

        decreasing_gains = increasing_gains[::-1]
        decreasing_losses = increasing_losses[::-1]

        increasing_list = []
        decreasing_list = []

        for i in range(n_good_checkpoints):
            increasing_list.append(
                PerfCheckpoint(
                    last_update_ms=good_checkpoint_times[i],
                    gain=increasing_gains[i],
                    loss=increasing_losses[i],
                    prev_portfolio_ret=1.0,
                    open_ms=good_checkpoint_time_accumulation[i]
                )
            )
        
        ledger_dict['increasing'] = ledger_generator(
            checkpoints=increasing_list
        )

        for i in range(n_good_checkpoints):
            decreasing_list.append(
                PerfCheckpoint(
                    last_update_ms=good_checkpoint_times[i],
                    gain=decreasing_gains[i],
                    loss=decreasing_losses[i],
                    prev_portfolio_ret=1.0,
                    open_ms=good_checkpoint_time_accumulation[i]
                )
            )
        
        ledger_dict['decreasing'] = ledger_generator(
            checkpoints=decreasing_list
        )


        ## miner whose consistency increases over time
        increasing_gains = np.zeros(n_good_checkpoints)
        increasing_losses = np.zeros(n_good_checkpoints)

        increasing_gains[int(len(increasing_gains)*3/4)] = 0.1
        increasing_losses[int(len(increasing_losses)*3/4)] = -0.05

        decreasing_gains = increasing_gains[::-1]
        decreasing_losses = increasing_losses[::-1]

        increasing_list = []
        decreasing_list = []

        for i in range(n_good_checkpoints):
            increasing_list.append(
                PerfCheckpoint(
                    last_update_ms=good_checkpoint_times[i],
                    gain=increasing_gains[i],
                    loss=increasing_losses[i],
                    prev_portfolio_ret=1.0,
                    open_ms=good_checkpoint_time_accumulation[i]
                )
            )
        
        ledger_dict['increasing_singular'] = ledger_generator(
            checkpoints=increasing_list
        )

        for i in range(n_good_checkpoints):
            decreasing_list.append(
                PerfCheckpoint(
                    last_update_ms=good_checkpoint_times[i],
                    gain=decreasing_gains[i],
                    loss=decreasing_losses[i],
                    prev_portfolio_ret=1.0,
                    open_ms=good_checkpoint_time_accumulation[i]
                )
            )
        
        ledger_dict['decreasing_singular'] = ledger_generator(
            checkpoints=decreasing_list
        )

        self.ledger_dict: dict[str, PerfLedger] = ledger_dict
        self.subtensor_weight_setter = SubtensorWeightSetter(
            config=None,
            wallet=None,
            metagraph=None,
            running_unit_tests=True
        )

        self.challengeperiod_manager = ChallengePeriodManager(
            config=None,
            metagraph=None,
            running_unit_tests=True
        )

    def test_miner_scoring_no_miners(self):
        """
        Test that the miner filtering works as expected when there are no miners
        """
        ledger_dict = {}
        filtered_results = Scoring.compute_results_checkpoint(ledger_dict)
        self.assertEqual(filtered_results, [])

    def test_miner_scoring_one_miner(self):
        """
        Test that the miner filtering works as expected when there is only one miner
        """
        ledger_dict = { 'miner0': ledger_generator(checkpoints=[checkpoint_generator(gain=0.1, loss=-0.05)]) }
        filtered_results = Scoring.compute_results_checkpoint(ledger_dict)
        filtered_netuids = [ x[0] for x in filtered_results ]
        filtered_values = [ x[1] for x in filtered_results ]

        original_netuids = ledger_dict.keys()
        original_values = ledger_dict['miner0'].get_product_of_gains() + ledger_dict['miner0'].get_product_of_loss()

        self.assertEqual(sorted(filtered_netuids), sorted(original_netuids))
        self.assertNotEqual(filtered_values[0], original_values)
        self.assertEqual(filtered_results, [('miner0', 1.0)])
 
    def test_transform_and_scale_results_defaults(self):
        """Test that the transform and scale results works as expected"""
        scaled_transformed_list: list[tuple[str, float]] = Scoring.compute_results_checkpoint(self.ledger_dict)

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

    def test_positive_returns(self):
        """Test that the returns scoring function works properly for only positive returns"""
        sample_gains = [0.4, 0.1, 0.2, 0.3, 0.4, 0.2]
        sample_losses = [0.0, 0.0, 0.0, -0.1, -0.3, 0.0] # contains losses
        sample_n_updates = [ 1, 1, 1, 1, 1, 1 ]
        sample_open_ms = [ 100, 200, 300, 400, 500, 600 ]

        return_positive = Scoring.return_cps(
            gains=sample_gains,
            losses=sample_losses,
            n_updates=sample_n_updates,
            open_ms=sample_open_ms,
        )

        self.assertGreaterEqual(return_positive, 0.0) # should always be greater than 0

    def test_negative_returns(self):
        """Test that the returns scoring function works properly for only negative returns"""
        sample_gains = [0.0, 0.1, 0.0, 0.1, 0.0, 0.0]
        sample_losses = [0.0, -0.05, -0.1, -0.1, -0.3, 0.0]
        sample_n_updates = [ 1, 1, 1, 1, 1, 1 ]
        sample_open_ms = [ 100, 200, 300, 400, 500, 600 ]

        return_negative = Scoring.return_cps(
            gains=sample_gains,
            losses=sample_losses,
            n_updates=sample_n_updates,
            open_ms=sample_open_ms,
        )

        self.assertLessEqual(return_negative, 0.0)

    def test_returns_zero_length_returns(self):
        """Test that the returns scoring function works properly with zero length returns"""
        return_zero = Scoring.return_cps(
            gains=[],
            losses=[],
            n_updates=[],
            open_ms=[]
        )

        self.assertLess(return_zero, 0.0)

    def test_positive_omega(self):
        """Test that the omega function works as expected for only positive returns"""
        sample_gains = [0.4, 0.1, 0.2, 0.3, 0.4, 0.2]
        sample_losses = [0.0, 0.0, 0.0, -0.1, -0.3, 0.0] # contains losses
        sample_n_updates = [ 1, 1, 1, 1, 1, 1 ]
        sample_open_ms = [ 100, 200, 300, 400, 500, 600 ]

        omega_positive = Scoring.omega_cps(
            gains=sample_gains,
            losses=sample_losses,
            n_updates=sample_n_updates,
            open_ms=sample_open_ms,
        )

        self.assertGreaterEqual(omega_positive, 0.0) # should always be greater than 0
        self.assertGreaterEqual(omega_positive, 1.0)

    def test_negative_omega(self):
        """Test that the omega function works as expected for all negative returns"""
        sample_gains = [0.0, 0.1, 0.0, 0.1, 0.0, 0.0]
        sample_losses = [0.0, -0.05, -0.1, -0.1, -0.3, 0.0]
        sample_n_updates = [ 1, 1, 1, 1, 1, 1 ]
        sample_open_ms = [ 100, 200, 300, 400, 500, 600 ]

        omega_negative = Scoring.omega_cps(
            gains=sample_gains,
            losses=sample_losses,
            n_updates=sample_n_updates,
            open_ms=sample_open_ms,
        )

        # Omega should be less than 1
        self.assertLessEqual(omega_negative, 1.0)
        self.assertGreaterEqual(omega_negative, 0.0)

    def test_omega_cps_handcalculation(self):
        """Test that the omega function works as expected"""
        sample_gains = [0.0, 0.1, 0.0, 0.1, 0.0, 0.0]
        sample_losses = [0.0, -0.05, -0.1, -0.1, -0.3, 0.0]
        sample_n_updates = [ 1, 1, 1, 1, 1, 1 ]
        sample_open_ms = [ 100, 200, 300, 400, 500, 600 ]

        ## returns - ( 1 + threshold ) -> we're ignoring threshold for internal calculations
        ## positive returns should be [ 1.1, 1.2, 1.3, 1.4, 1.2 ] -> [ 0.1, 0.2, 0.3, 0.4, 0.2 ]
        ## negative returns should be [ 0.9, 0.7 ] -> [ -0.1, -0.3 ]
        hand_computed_omega = sum(sample_gains) / abs(sum(sample_losses))

        ## omega should be [ 1.1 + 1.2 + 1.3 + 1.4 + 1.2 ] / [ 0.9 + 0.7 ]
        omega = Scoring.omega_cps(
            gains=sample_gains,
            losses=sample_losses,
            n_updates=sample_n_updates,
            open_ms=sample_open_ms,
        )
        self.assertEqual(omega, hand_computed_omega)

    def test_omega_cps(self):
        """Test inverted sortino function works as expected"""
        omega_scores = {}
        for miner, minerledger in self.ledger_dict.items():
            gains = [ cp.gain for cp in minerledger.cps ]
            losses = [ cp.loss for cp in minerledger.cps ]
            n_updates = [ cp.n_updates for cp in minerledger.cps ]
            open_ms = [ cp.open_ms for cp in minerledger.cps ]

            score = Scoring.omega_cps(
                gains=gains,
                losses=losses,
                n_updates=n_updates,
                open_ms=open_ms
            )

            omega_scores[miner] = score

        ## the good miner should have the highest score
        self.assertGreater(omega_scores['good'], omega_scores['bad'])

        ## Good miner should be better than the average miner
        self.assertGreater(omega_scores['good'], omega_scores['miner0'])

        ## High swings miner will likely be better than the lowswings miner for omega
        self.assertGreater(omega_scores['highswings'], omega_scores['lowswings'])
        self.assertGreater(omega_scores['highswingsactive'], omega_scores['highswingsinactive'])
        self.assertGreater(omega_scores['lowswingsactive'], omega_scores['lowswingsinactive'])

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
        omega = Scoring.omega_cps(
            gains=[],
            losses=[],
            n_updates=[],
            open_ms=[]
        )

        self.assertEqual(omega, 0.0)

    def test_omega_zero_loss(self):
        """Test that the omega function works as expected with zero loss"""
        sample_gains = [0.1, 0.2, 0.3, 0.4, 0.2]
        sample_losses = [0.0, 0.0, 0.0, 0.0, 0.0]
        sample_n_updates = [ 1, 1, 1, 1, 1 ]
        sample_open_ms = [ 100, 200, 300, 400, 500 ]

        omega = Scoring.omega_cps(
            gains=sample_gains,
            losses=sample_losses,
            n_updates=sample_n_updates,
            open_ms=sample_open_ms
        )

        ## omega_minimum_denominator should kick in
        omega_minimum_denominator = ValiConfig.OMEGA_MINIMUM_DENOMINATOR
        no_loss_benefit = 1 / omega_minimum_denominator

        self.assertGreaterEqual(omega, no_loss_benefit)
        self.assertGreaterEqual(omega, 0.0) # should always be greater than 0
        self.assertGreaterEqual(omega, 1.0)

    def test_sortino_zero_length(self):
        """Test that the sortino function works as expected with zero length returns"""        
        sortino = Scoring.inverted_sortino_cps(
            gains=[],
            losses=[],
            n_updates=[],
            open_ms=[]
        )

        self.assertEqual(sortino, 0.0)

    def test_sortino_zero_loss(self):
        """Test that the sortino function works as expected with zero loss"""
        sample_gains = [0.1, 0.2, 0.3, 0.4, 0.2]
        sample_losses = [0.0, 0.0, 0.0, 0.0, 0.0]
        sample_n_updates = [ 1, 1, 1, 1, 1 ]
        sample_open_ms = [ 100, 200, 300, 400, 500 ]

        inverted_sortino = Scoring.inverted_sortino_cps(
            gains=sample_gains,
            losses=sample_losses,
            n_updates=sample_n_updates,
            open_ms=sample_open_ms
        )

        self.assertEqual(inverted_sortino, 0)

    def test_inverted_sortino_cps(self):
        """Test inverted sortino function works as expected"""
        sortino_scores = {}
        for miner, minerledger in self.ledger_dict.items():
            gains = [ cp.gain for cp in minerledger.cps ]
            losses = [ cp.loss for cp in minerledger.cps ]
            n_updates = [ cp.n_updates for cp in minerledger.cps ]
            open_ms = [ cp.open_ms for cp in minerledger.cps ]

            score = Scoring.inverted_sortino_cps(
                gains=gains,
                losses=losses,
                n_updates=n_updates,
                open_ms=open_ms
            )

            sortino_scores[miner] = score

        ## the good miner should have the highest score
        self.assertGreater(sortino_scores['good'], sortino_scores['bad'])

        ## Good miner should be better than the average miner
        self.assertGreater(sortino_scores['good'], sortino_scores['miner0'])

        ## High swings miner should be worse than the low swings miner for sortino
        self.assertGreater(sortino_scores['lowswings'], sortino_scores['highswings'])

        ## Check that the active highswings miner is better than the inactive highswings miner
        self.assertAlmostEqual(
            sortino_scores['highswingsactive'], 
            sortino_scores['highswings']
        )

        self.assertAlmostEqual(
            sortino_scores['lowswingsactive'],
            sortino_scores['lowswings']
        )

        self.assertAlmostEqual(
            sortino_scores['highswingsactive'],
            sortino_scores['highswingsinactive']
        )

        self.assertAlmostEqual(
            sortino_scores['lowswingsactive'],
            sortino_scores['lowswingsinactive']
        )

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

    # def test_filter_position_duration(self):
    #     """Test that the transform and scale results works as expected with a grace period"""
    #     ## the transformation should be some kind of average of the two
    #     nonpassing_miners = [ 'nopositions', 'timeconstrained' ]
    #     flagged_miners = []

    #     for miner, minerledger in self.ledger_dict.items():
    #         checkpoint_meets_criteria = self.subtensor_weight_setter._filter_checkpoint_list(minerledger.cps)
    #         if not checkpoint_meets_criteria:
    #             flagged_miners.append(miner)

    #     self.assertEqual(sorted(flagged_miners), sorted(nonpassing_miners))

    def test_filter_single_checkpoint(self):
        """Test that the filter checkpoint function works as expected"""
        ## the transformation should be some kind of average of the two
        minimum_passing_checkpoint_time = ValiConfig.SET_WEIGHT_MINIMUM_SINGLE_CHECKPOINT_DURATION_MS
        passing_checkpoint = checkpoint_generator(
            gain=0.1,
            loss=-0.05,
            open_ms=minimum_passing_checkpoint_time
        )

        nonpassing_checkpoint = checkpoint_generator(
            gain=0.1,
            loss=-0.05,
            open_ms=minimum_passing_checkpoint_time - 10
        )

        passing_checkpoint_list = self.subtensor_weight_setter._filter_checkpoint_elements(
            [ passing_checkpoint ]
        )

        nonpassing_checkpoint_list = self.subtensor_weight_setter._filter_checkpoint_elements(
            [ nonpassing_checkpoint ]
        )

        self.assertEqual(passing_checkpoint_list, [ passing_checkpoint ])
        self.assertEqual(nonpassing_checkpoint_list, [])

    def test_miner_filter_older_than_challengeperiod(self):
        """Test that the challengeperiod function eliminates miners older than the period"""
        ## some sample positions and their orders, want to make sure we return
        minimum_checkpoint_duration = ValiConfig.SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS
        typical_checkpoint_duration = minimum_checkpoint_duration // 5

        checkpoints = [ checkpoint_generator(gain=0.05, loss=-0.1, open_ms=typical_checkpoint_duration) ] * 2
        sample_ledger = ledger_generator(checkpoints=checkpoints)

        passing, failing = self.challengeperiod_manager.inspect( { "miner": sample_ledger }, inspection_hotkeys={ "miner": self.start_time }, current_time=self.end_time + 100 )
        self.assertNotIn("miner", passing)
        self.assertIn("miner", failing)

    def test_miner_filter_challengeperiod_no_positions(self):
        """Test that the miner filter function works as expected with no positions"""
        ## some sample positions and their orders, want to make sure we return
        checkpoints = []
        sample_ledger = ledger_generator(checkpoints=checkpoints)

        passing, failing = self.challengeperiod_manager.inspect( { "miner": sample_ledger }, inspection_hotkeys={ "miner": self.start_time }, current_time=self.end_time + 100 )
        self.assertNotIn("miner", passing)
        self.assertIn("miner", failing)

    def test_challengeperiod_screening_onepass(self):
        """Test that challengeperiod screening passes all miners who meet the criteria"""
        ## some sample positions and their orders, want to make sure we return
        challengeperiod_keys = self.ledger_dict.keys()
        inspection_hotkeys = { key: self.start_time for key in challengeperiod_keys }

        challengeperiod_success, challengeperiod_eliminations = self.challengeperiod_manager.inspect(
            ledger = self.ledger_dict,
            inspection_hotkeys = inspection_hotkeys,
            current_time = self.end_time,
        )

        self.assertIn("nopositions", challengeperiod_eliminations)
        # self.assertIn("timeconstrained", challengeperiod_eliminations)

    # def test_challengeperiod_screening_challengeperiod_one_elimination(self):
    #     inspection_hotkeys = {
    #         'nopositions': self.start_time
    #     }

    #     challengeperiod_success, challengeperiod_eliminations = self.challengeperiod_manager.inspect(
    #         self.ledger_dict,
    #         inspection_hotkeys=inspection_hotkeys,
    #         current_time=self.end_time
    #     )

    #     ## this miner should be eliminated, as the time they spent was greater than the challenge period and they still haven't met the critieria
    #     self.assertIn("nopositions", challengeperiod_eliminations)

    def test_challengeperiod_remaining_in_challenge(self):
        """Test that the challengeperiod function works as expected"""
        ## some sample positions and their orders, want to make sure we return
        minimum_checkpoint_duration = ValiConfig.SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS
        typical_checkpoint_duration = minimum_checkpoint_duration // 5

        checkpoints = [ checkpoint_generator(gain=0.05, loss=-0.1, open_ms=typical_checkpoint_duration) ] * 3
        sample_ledger = ledger_generator(checkpoints=checkpoints)

        passing, failing = self.challengeperiod_manager.inspect( { "miner": sample_ledger }, inspection_hotkeys={ "miner": self.start_time }, current_time=self.end_time - 100 )
        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_challengeperiod_no_return(self):
        """Test that the total position duration function works as expected"""
        minimum_checkpoint_duration = ValiConfig.SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS
        typical_checkpoint_duration = minimum_checkpoint_duration // 5

        checkpoints = [ checkpoint_generator(gain=0.05, loss=-0.1, open_ms=typical_checkpoint_duration) ] * 10
        sample_ledger = ledger_generator(checkpoints=checkpoints)

        ledger_logic = self.challengeperiod_manager.screen_ledger(sample_ledger)
        self.assertFalse(ledger_logic)

    def test_challengeperiod_total_checkpoint_duration(self):
        """Test that the total position duration function works as expected"""
        minimum_checkpoint_duration = ValiConfig.SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS
        typical_checkpoint_duration = minimum_checkpoint_duration // 5

        checkpoints = [ checkpoint_generator(gain=0.1, loss=-0.05, open_ms=typical_checkpoint_duration) ] * 10
        sample_ledger = ledger_generator(checkpoints=checkpoints)

        ledger_logic = self.challengeperiod_manager.screen_ledger(sample_ledger)
        self.assertTrue(ledger_logic)

    def test_dampen_value(self):
        """Test that the dampen value function works as expected"""
        dampen_value = 0.5
        lookback_fraction1 = 0.1
        lookback_fraction2 = 0.5
        lookback_fraction3 = 0.9

        dampened_one = PositionUtils.dampen_value(
            dampen_value,
            lookback_fraction=lookback_fraction1
        )

        dampened_two = PositionUtils.dampen_value(
            dampen_value,
            lookback_fraction=lookback_fraction2
        )

        dampened_three = PositionUtils.dampen_value(
            dampen_value,
            lookback_fraction=lookback_fraction3
        )

        self.assertGreater(dampened_one, dampened_two)
        self.assertGreater(dampened_two, dampened_three)

    def test_augment_ledger_increasing(self):
        """Test that the augment ledger function works as expected, increasing position will score better with more decay (more recent consideration)"""
        highdecay = PositionManager.augment_perf_ledger(
            self.ledger_dict,
            evaluation_time_ms=self.end_time,
            time_decay_coefficient=0.2
        )['increasing']

        lowdecay = PositionManager.augment_perf_ledger(
            self.ledger_dict,
            evaluation_time_ms=self.end_time,
            time_decay_coefficient=0.8
        )['increasing']

        ## with higher amount of historical decay, the sortino will be lower as the losses were lower initially, which will now weigh more heavily. The returns should also be lower.
        highdecay_return = Scoring.return_cps(
            gains = [ cp.gain for cp in highdecay.cps ],
            losses = [ cp.loss for cp in highdecay.cps ],
            n_updates = [ cp.n_updates for cp in highdecay.cps ],
            open_ms = [ cp.open_ms for cp in highdecay.cps ]
        )

        lowdecay_return = Scoring.return_cps(
            gains = [ cp.gain for cp in lowdecay.cps ],
            losses = [ cp.loss for cp in lowdecay.cps ],
            n_updates = [ cp.n_updates for cp in lowdecay.cps ],
            open_ms = [ cp.open_ms for cp in lowdecay.cps ]
        )

        self.assertGreater(lowdecay_return, highdecay_return)

        highdecay_sortino = Scoring.inverted_sortino_cps(
            gains = [ cp.gain for cp in highdecay.cps ],
            losses = [ cp.loss for cp in highdecay.cps ],
            n_updates = [ cp.n_updates for cp in highdecay.cps ],
            open_ms = [ cp.open_ms for cp in highdecay.cps ]
        )

        lowdecay_sortino = Scoring.inverted_sortino_cps(
            gains = [ cp.gain for cp in lowdecay.cps ],
            losses = [ cp.loss for cp in lowdecay.cps ],
            n_updates = [ cp.n_updates for cp in lowdecay.cps ],
            open_ms = [ cp.open_ms for cp in lowdecay.cps ]
        )

        self.assertGreater(lowdecay_sortino, highdecay_sortino)

    def test_augment_ledger_decreasing(self):
        """Test that the augment ledger function works as expected, increasing position will score better with more decay (more recent consideration)"""
        highdecay = PositionManager.augment_perf_ledger(
            self.ledger_dict,
            evaluation_time_ms=self.end_time,
            time_decay_coefficient=0.45
        )['decreasing_singular']

        lowdecay = PositionManager.augment_perf_ledger(
            self.ledger_dict,
            evaluation_time_ms=self.end_time,
            time_decay_coefficient=1.0
        )['decreasing_singular']

        highdecay_sortino = Scoring.inverted_sortino_cps(
            gains = [ cp.gain for cp in highdecay.cps ],
            losses = [ cp.loss for cp in highdecay.cps ],
            n_updates = [ cp.n_updates for cp in highdecay.cps ],
            open_ms = [ cp.open_ms for cp in highdecay.cps ]
        )

        lowdecay_sortino = Scoring.inverted_sortino_cps(
            gains = [ cp.gain for cp in lowdecay.cps ],
            losses = [ cp.loss for cp in lowdecay.cps ],
            n_updates = [ cp.n_updates for cp in lowdecay.cps ],
            open_ms = [ cp.open_ms for cp in lowdecay.cps ]
        )

        self.assertGreaterEqual(highdecay_sortino, lowdecay_sortino)

    def test_augment_sortino(self):
        """Test that sortino is working as expected"""
        gains = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        losses = np.ones(10) * -0.1
        open_ms = (np.ones(10) * 100).tolist()

        increasing_losses = copy.deepcopy(losses)
        increasing_losses[:5] = 0

        decreasing_losses = copy.deepcopy(losses)
        decreasing_losses[5:] = 0

        increasing_sortino = Scoring.inverted_sortino_cps(
            gains=gains,
            losses=increasing_losses,
            n_updates=[1] * 10,
            open_ms=open_ms
        )

        decreasing_sortino = Scoring.inverted_sortino_cps(
            gains=gains,
            losses=decreasing_losses,
            n_updates=[1] * 10,
            open_ms=open_ms
        )

        self.assertEqual(decreasing_sortino, increasing_sortino)

    def test_augment_checkpoint(self):
        """Test that the checkpoint augmentation works as expected"""
        increasing_cps_highdecay = PositionManager.augment_perf_checkpoint(
            self.ledger_dict['increasing'].cps,
            evaluation_time_ms=self.end_time,
            time_decay_coefficient=0.2
        )

        increasing_cps_lowdecay = PositionManager.augment_perf_checkpoint(
            self.ledger_dict['increasing'].cps,
            evaluation_time_ms=self.end_time,
            time_decay_coefficient=0.9
        )

        self.assertGreater(
            increasing_cps_lowdecay[len(increasing_cps_lowdecay)//2].gain, # low decay should have higher historical
            increasing_cps_highdecay[len(increasing_cps_highdecay)//2].gain,
        )

        self.assertLess(
            increasing_cps_lowdecay[len(increasing_cps_lowdecay)//2].loss,
            increasing_cps_highdecay[len(increasing_cps_highdecay)//2].loss,
        )

        self.assertGreater(
            increasing_cps_lowdecay[len(increasing_cps_lowdecay)//2].open_ms,
            increasing_cps_highdecay[len(increasing_cps_highdecay)//2].open_ms,
        )


