# developer: trdougherty
import numpy as np
import copy
import math

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring
from vali_objects.position import Position
from vali_objects.enums.order_type_enum import OrderType
from vali_config import TradePair

from vali_config import ValiConfig

from tests.shared_objects.test_utilities import generate_ledger


class TestWeights(TestBase):

    def setUp(self):
        super().setUp()

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_CLOSE_MS = 2000
        self.EVALUATION_TIME_MS = self.DEFAULT_CLOSE_MS + 1

        self.DEFAULT_ORDER_MS = 1000
        self.MS_IN_DAY = 86400000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_POSITION = Position(
            position_type=OrderType.LONG,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )

        self.DEFAULT_LEDGER = generate_ledger(0.1)

    def test_transform_and_scale_results_defaults(self):
        """Test that the transform and scale results works as expected"""
        ledger = {}
        miner_positions = {}
        for i in range(10):
            ledger[f"miner{i}"] = generate_ledger(0.1)
            miner_positions[f"miner{i}"] = [copy.deepcopy(self.DEFAULT_POSITION)]

        # Test the default values
        scaled_transformed_list: list[tuple[str, float]] = Scoring.compute_results_checkpoint(
            ledger,
            miner_positions,
            evaluation_time_ms=self.EVALUATION_TIME_MS
        )

        # With no miners, we should not have any results
        self.assertListEqual(scaled_transformed_list, [])

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
        self.assertListEqual(values, [])

    def test_return_no_positions(self):
        self.assertEqual(Scoring.base_return([]), 0.0)

    def test_negative_returns(self):
        """Test that the returns scoring function works properly for only negative returns"""
        positional_returns = [0.8, 0.9, 0.7, 0.8, 0.9, 0.7]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        # Switch to log returns
        base_return = Scoring.base_return(positions)
        self.assertLess(base_return, 0.0)

    def test_positive_returns(self):
        """Test that the returns scoring function works properly for only positive returns"""
        positional_returns = [1.2, 1.1, 1.3, 1.2, 1.1, 1.3]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        # Switch to log returns
        base_return = Scoring.base_return(positions)
        self.assertGreater(base_return, 0.0)

    def test_typical_omega(self):
        """Test that the omega function works as expected for only positive returns"""
        positional_returns = [1.2, 0.9, 1.3, 0.8, 1.1, 1.3]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        omega = Scoring.omega(positions, self.DEFAULT_LEDGER)

        # Should always be greater or equal to 0
        self.assertGreater(omega, 0.0)

    def test_negative_omega(self):
        """Test that the omega function works as expected for only negative returns"""
        positional_returns = [0.8, 0.9, 0.7, 0.8, 0.9, 0.7]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        omega = Scoring.omega(positions, self.DEFAULT_LEDGER)

        # Should always be less or equal to 0
        self.assertEqual(omega, 0.0)

    def test_positive_omega(self):
        """Test that the omega function works as expected for only positive returns - the default loss works"""
        positional_returns = [1.2, 1.1, 1.3, 1.2, 1.1, 1.3]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        omega = Scoring.omega(positions, self.DEFAULT_LEDGER)

        # Should always be greater or equal to 0, cannot be massive
        self.assertGreater(omega, 0.0)
        self.assertLess(omega, np.inf)

        # should be the same as taking a default loss of self.OMEGA_LOSS_MINIMUM
        minimum_omega_loss = ValiConfig.OMEGA_LOSS_MINIMUM
        p = copy.deepcopy(self.DEFAULT_POSITION)
        p.return_at_close = 1 - minimum_omega_loss

        prior_positional_sum = [math.log(x.return_at_close) for x in positions]

        numerator = sum(prior_positional_sum)
        denominator = abs(math.log(p.return_at_close))

        positions.append(p)
        omega_calculated = Scoring.omega(positions, self.DEFAULT_LEDGER)
        omega_hand_calculated = numerator / denominator

        self.assertAlmostEqual(omega_calculated, omega_hand_calculated)

    def test_omega_no_returns(self):
        """Test that the omega function works as expected for no returns"""
        positions = []

        omega = Scoring.omega(positions, self.DEFAULT_LEDGER)

        # Expected value is zero
        self.assertEqual(omega, 0.0)

    def test_positive_omega_small_loss(self):
        """Test that the omega function works as expected for only positive returns - the default loss works"""
        positional_returns = [1.2, 1.1, 1.3, 1.2, 1.1, 1.3, 0.99]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        omega = Scoring.omega(positions, self.DEFAULT_LEDGER)

        # Should always be greater or equal to 0, cannot be massive
        self.assertGreater(omega, 0.0)
        self.assertLess(omega, np.inf)

        # should be the same as taking a default loss of self.OMEGA_LOSS_MINIMUM
        minimum_omega_loss = ValiConfig.OMEGA_LOSS_MINIMUM

        gains = [math.log(x.return_at_close) for x in positions if x.return_at_close > 1.0]
        losses = [math.log(x.return_at_close) for x in positions if x.return_at_close < 1.0]

        losses_omega = max(abs(sum(losses)), minimum_omega_loss)

        omega_hand_calculated = sum(gains) / losses_omega
        self.assertAlmostEqual(omega, omega_hand_calculated)

    def test_sharpe_positive(self):
        """Test that the sharpe function works as expected"""
        positional_returns = [1.2, 1.1, 1.3, 1.2, 1.1, 1.3]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        sharpe = Scoring.sharpe(positions, self.DEFAULT_LEDGER)

        # Should always be greater or equal to 0
        self.assertGreater(sharpe, 0.0)

    def test_sharpe_negative(self):
        """Test that the sharpe function works as expected"""
        positional_returns = [0.8, 0.9, 0.7, 0.8, 0.9, 0.7]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        sharpe = Scoring.sharpe(positions, self.DEFAULT_LEDGER)

        # Expected value less than zero
        self.assertLess(sharpe, 0.0)
        self.assertGreater(sharpe, -10)

    def test_sharpe_no_returns(self):
        """Test that the sharpe function works as expected"""
        positions = []

        sharpe = Scoring.sharpe(positions, self.DEFAULT_LEDGER)

        # Expected value is zero
        self.assertEqual(sharpe, 0.0)

    def test_sharpe_no_returns_no_variance(self):
        """Test that the sharpe function works as expected"""
        positional_returns = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        sharpe = Scoring.sharpe(positions, self.DEFAULT_LEDGER)

        # Expected value is zero
        self.assertEqual(sharpe, 0.0)

    def test_sharpe_perfect_positive(self):
        """Test that the sharpe function works as expected"""
        positional_returns = [1.001, 1.001, 1.001, 1.001, 1.001, 1.001]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        sharpe = Scoring.sharpe(positions, self.DEFAULT_LEDGER)

        # Expected value is zero
        self.assertGreater(sharpe, 0.0)
        self.assertLess(sharpe, 10)

    def test_swing_miners(self):
        """Test that the sharpe function works as expected"""
        m1 = []

        # First miner spreads returns equally across all positions
        total_return_1 = 0.2  # 20% total return
        n_positions_1 = 10
        per_position_return_1 = (1 + total_return_1) ** (1 / n_positions_1) - 1

        for i in range(n_positions_1):
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = 1 + per_position_return_1
            m1.append(p)

        # Second miner has small returns in most positions and a large return in one
        m2 = []

        small_return_2 = 0.001
        n_positions_2 = 10
        # Calculate the large return to ensure the total return matches m1
        large_return_2 = (1 + total_return_1) / (1 + small_return_2) ** (n_positions_2 - 1) - 1

        for i in range(n_positions_2 - 1):
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = 1 + small_return_2
            m2.append(p)

        high_return_position = copy.deepcopy(self.DEFAULT_POSITION)
        high_return_position.return_at_close = 1 + large_return_2
        m2.append(high_return_position)

        # Assertions to compare performance metrics
        self.assertAlmostEqual(Scoring.base_return(m1), Scoring.base_return(m2), places=2)
        self.assertGreater(Scoring.sharpe(m1, self.DEFAULT_LEDGER), Scoring.sharpe(m2, self.DEFAULT_LEDGER))
        self.assertAlmostEqual(Scoring.omega(m2, self.DEFAULT_LEDGER), Scoring.omega(m1, self.DEFAULT_LEDGER))

    def test_no_miners(self):
        """Test when there are no miners in the list"""
        miner_scores = []
        result = Scoring.miner_scores_percentiles(miner_scores)
        self.assertEqual(result, [])

    def test_one_miner(self):
        """Test when there is only one miner in the list"""
        miner_scores = [("miner1", 10.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)
        self.assertEqual(result, [("miner1", 1.0)])

    def test_all_same_scores(self):
        """Test when all miners have the same scores"""
        miner_scores = [("miner1", 10.0), ("miner2", 10.0), ("miner3", 10.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)
        expected_result = [("miner1", 0.6667), ("miner2", 0.6667), ("miner3", 0.6667)]

        for i in range(len(result)):
            self.assertAlmostEqual(result[i][1], expected_result[i][1], places=3)

    def test_zero_value_conditions(self):
        """Test when all scores are zero"""
        miner_scores = [("miner1", 0.0), ("miner2", 0.0), ("miner3", 0.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)
        expected_result = [("miner1", 0.6667), ("miner2", 0.6667),
                           ("miner3", 0.6667)]  # All scores are zero, so all are ranked the same
        for i in range(len(result)):
            self.assertAlmostEqual(result[i][1], expected_result[i][1], places=3)

    def test_typical_conditions(self):
        """Test when miners have different scores"""
        miner_scores = [("miner1", 20.0), ("miner2", 30.0), ("miner3", 10.0), ("miner4", 40.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)

        # Expected percentiles:
        # "miner3" with score 10.0 -> 0.25 (25th percentile)
        # "miner1" with score 20.0 -> 0.50 (50th percentile)
        # "miner2" with score 30.0 -> 0.75 (75th percentile)
        # "miner4" with score 40.0 -> 1.00 (100th percentile)
        expected_result = [
            ("miner1", 0.50),
            ("miner2", 0.75),
            ("miner3", 0.25),
            ("miner4", 1.00)
        ]

        self.assertEqual(result, expected_result)
    def test_no_miners_softmax(self):
       """Test when there are no miners in the list"""
       miner_scores = []
       result = Scoring.softmax_scores(miner_scores)
       self.assertEqual(result, [])

    def test_one_miner_softmax(self):
        """Test when there is only one miner in the list"""
        miner_scores = [("miner1", 10.0)]
        result = Scoring.softmax_scores(miner_scores)
        self.assertEqual(result, [("miner1", 1.0)])

    def test_ordering_softmax(self):
        returns = [("miner1", 10.0), ("miner2", 5.0), ("miner3", 1.0), ("miner4", 15.0)]
        result = Scoring.softmax_scores(returns)
        
        #Sort the list by order of softmax output values
        result.sort(key=lambda x: x[1])
        ordered_keys = [s[0] for s in result]
        self.assertEqual(ordered_keys, ["miner3", "miner2", "miner1", "miner4"])

    def test_sum_to_one_softmax(self):

        returns = [("miner1", 10.0), ("miner2", 5.0), ("miner3", 1.0), ("miner4", 15.0),("miner5", 15.0)]
        result = Scoring.softmax_scores(returns)
        values = [v[1] for v in result]
        self.assertAlmostEqual(sum(values), 1.0, places=3)