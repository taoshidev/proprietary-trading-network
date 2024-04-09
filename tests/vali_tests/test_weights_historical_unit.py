# developer: trdougherty
# Copyright © 2024 Taoshi Inc
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring
from vali_objects.position import Position
import pickle
import hashlib
from vali_config import ValiConfig, TradePair
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.scoring.historical_scoring import HistoricalScoring

import numpy as np
import random

class TestWeights(TestBase):

    def setUp(self):
        super().setUp()

        def hash_object(obj):
            serialized_obj = pickle.dumps(obj)
            hash_obj = hashlib.sha256()
            hash_obj.update(serialized_obj)
            hashed_str = hash_obj.hexdigest()
            return hashed_str[:10]
        
        def get_time_in_range(percent, start, end):
            return int(start + ((end - start) * percent))

        def position_generator(
            open_time_ms, 
            close_time_ms,
            trade_pair,
            return_at_close
        ):
            generated_position = Position(
                miner_hotkey='miner0',
                position_uuid=hash_object((
                    open_time_ms,
                    close_time_ms,
                    return_at_close,
                    trade_pair
                )),
                open_ms=open_time_ms,
                trade_pair=trade_pair,
            )

            generated_position.close_out_position(
                close_ms = close_time_ms
            )
            generated_position.return_at_close = return_at_close
            return generated_position
        

        ## seeding
        np.random.seed(0)
        random.seed(0)

        n_positions = 50
        self.n_positions = n_positions

        self.start_time = 1710521764446
        self.end_time = self.start_time + ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_MS

        self.start_times = sorted([ get_time_in_range(x, self.start_time, self.end_time) for x in np.random.rand(n_positions) ])
        self.end_times = sorted([ get_time_in_range(x, self.start_times[c], self.end_time) for c,x in enumerate(np.random.rand(n_positions)) ])

        self.returns_at_close = [ 1 + (np.random.rand() - 0.5) * 0.1 for x in range(n_positions) ]

        mock_positions = []
        for position in range(n_positions):
            mock_positions.append(
                position_generator(
                    open_time_ms=self.start_times[position],
                    close_time_ms=self.end_times[position],
                    trade_pair=TradePair.BTCUSD,
                    return_at_close=self.returns_at_close[position]
                )
            )

        ## positions in the last 80% of the time range
        imbalanced_position_close_times = sorted([
            get_time_in_range(x, (self.end_time - self.start_time) * 0.8, self.end_time) for x in np.random.rand(n_positions)
        ])

        ## this should score much lower
        self.imbalanced_positions = [
            position_generator(
                open_time_ms=self.start_time,
                close_time_ms=imbalanced_position_close_times[c],
                trade_pair=TradePair.BTCUSD,
                return_at_close=1.1
            ) for c in range(n_positions)
        ]

        ## this should score well in balance
        self.balanced_positions = [
            position_generator(
                open_time_ms=self.start_times[c],
                close_time_ms=self.end_times[c],
                trade_pair=TradePair.BTCUSD,
                return_at_close=1.1
            ) for c in range(n_positions)
        ]

        self.mock_positions = mock_positions

    def test_compute_consistency_penalty(self):
        """
        Test that the consistency penalty works as expected
        """
        evaluation_time = self.end_time


        penalty = PositionUtils.compute_consistency_penalty(
            self.mock_positions, 
            evaluation_time
        )
        self.assertGreaterEqual(penalty, 0)
        self.assertLessEqual(penalty, 1)

    def test_compute_consistency_penalty_empty(self):
        """
        Test that the consistency penalty works as expected when the close times are empty
        """
        close_times = []
        evaluation_time = self.end_time
        penalty = PositionUtils.compute_consistency_penalty(close_times, evaluation_time)
        self.assertEqual(penalty, 0) # if no score then we multiply everything by zero

    def test_compute_consistency_penalty_single(self):
        """
        Test that the consistency penalty works as expected when there is only one position
        """
        evaluation_time = self.end_time
        penalty = PositionUtils.compute_consistency_penalty(
            [self.mock_positions[0]], 
            evaluation_time
        )

        self.assertGreater(penalty, 0)
        self.assertLessEqual(penalty, 0.5)

    def test_compute_consistency_penalty_known(self):
        """
        Test that the consistency penalty works as expected when there is only one position
        """
        evaluation_time = self.end_time

        imbalanced_penalty = PositionUtils.compute_consistency_penalty(
            self.imbalanced_positions, 
            evaluation_time
        )

        balanced_penalty = PositionUtils.compute_consistency_penalty(
            self.balanced_positions, 
            evaluation_time
        )

        # want to make sure that the imbalanced penalty will multiply returns with a lower value
        self.assertLess(imbalanced_penalty, balanced_penalty)

    def test_compute_consistency_hand_check(self):
        """
        Test that the consistency penalty works as expected when there is only one position
        """

        percentage_windowed = [ 0.1, 0.3, 0.5, 0.7, 0.9 ] # balanced
        imbalanced_percentage_windowed = [ 0.02, 0.1, 0.3, 0.35, 0.38, 0.42 ] # imbalanced
        evaluation_time = self.end_time

        close_times = [ int(self.start_time + (self.end_time - self.start_time) * x) for x in percentage_windowed ]
        imbalanced_close_times = [ int(self.start_time + (self.end_time - self.start_time) * x) for x in imbalanced_percentage_windowed ]

        balanced_positions = [
            Position(
                miner_hotkey='miner0',
                position_uuid='balanced',
                open_ms=self.start_time,
                close_ms=x,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=True
            ) for x in close_times
        ]

        imbalanced_positions = [
            Position(
                miner_hotkey='miner0',
                position_uuid='imbalanced',
                open_ms=self.start_time,
                close_ms=x,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=True
            ) for x in imbalanced_close_times
        ]

        penalty = PositionUtils.compute_consistency_penalty(
            balanced_positions, 
            evaluation_time
        )

        imbalanced_penalty = PositionUtils.compute_consistency_penalty(
            imbalanced_positions, 
            evaluation_time
        )

        self.assertLess(imbalanced_penalty, penalty)


    def test_log_transform(self):
        """
        Test that the log transform works as expected
        """
        for position in self.mock_positions:
            return_value = position.return_at_close
            log_return = PositionUtils.log_transform(return_value)
            self.assertAlmostEqual(np.exp(log_return), return_value, places=5)

    def test_exp_transform(self):
        """
        Test that the exp transform works as expected
        """
        for position in self.mock_positions:
            return_value = position.return_at_close
            exp_return = PositionUtils.exp_transform(return_value)
            self.assertAlmostEqual(np.log(exp_return), return_value, places=5)

    def test_compute_lookback_fraction(self):
        """
        Test that the compute lookback fraction works as expected
        """
        # check the first position manually
        position = self.mock_positions[0]

        evaluation_time = 1710523564446
        time_delta = ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_MS

        close_time = evaluation_time - time_delta + int(time_delta * 0.3) # 30% of the way through the lookback period
        open_time = evaluation_time - time_delta + int(time_delta * 0.1) # 10% of the way through the lookback period

        lookback_fraction = PositionUtils.compute_lookback_fraction(open_time, close_time, evaluation_time)
        self.assertAlmostEqual(lookback_fraction, 0.7, places=5)

        for position in self.mock_positions[1:]:
            open_time = position.open_ms
            close_time = position.close_ms
            evaluation_time = self.end_time
            lookback_fraction = PositionUtils.compute_lookback_fraction(open_time, close_time, evaluation_time)
            self.assertGreaterEqual(lookback_fraction, 0)
            self.assertLessEqual(lookback_fraction, 1)

    def test_historical_decay_return(self):
        """
        Test that the historical decay return works as expected
        """
        return_value = 1.4
        time_fraction = 1
        decayed_return = HistoricalScoring.historical_decay_return(return_value, time_fraction)
        self.assertAlmostEqual(decayed_return, 0, places=5)

    def test_historical_decay_return_zero(self):
        """
        Test that the historical decay return works as expected when the time fraction is zero
        """
        return_value = 1.4
        time_fraction = 0
        decayed_return = HistoricalScoring.historical_decay_return(return_value, time_fraction)
        self.assertAlmostEqual(decayed_return, return_value, places=5)

    def test_historical_decay_return_gt_one(self):
        """
        Test that the historical decay return works as expected when the return value is greater than one
        """
        return_value = 1.4
        time_fraction = 1.1
        decayed_return = HistoricalScoring.historical_decay_return(return_value, time_fraction)
        self.assertAlmostEqual(decayed_return, 0)

    def test_historical_decay_return_lt_zero(self):
        """
        Test that the historical decay return works as expected when the return value is less than zero
        """
        return_value = 1.4
        time_fraction = -0.2
        decayed_return = HistoricalScoring.historical_decay_return(return_value, time_fraction)
        self.assertAlmostEqual(decayed_return, return_value)

    def test_permute_time_intensity(self):
        """
        Test that the permute time intensity works as expected
        """
        time_pertinence = 1.0
        permuted_intensity = HistoricalScoring.permute_time_intensity(time_pertinence)
        self.assertAlmostEqual(permuted_intensity, 1.0, places=5)

        time_pertinence = 0.0
        permuted_intensity = HistoricalScoring.permute_time_intensity(time_pertinence)
        self.assertAlmostEqual(permuted_intensity, 0.0, places=5)

        # test of monotonicity
        pertinence_array = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
        for c, intensity in enumerate(pertinence_array[:-1]):
            permuted_intensity_current = HistoricalScoring.permute_time_intensity(intensity)
            permuted_intensity_next = HistoricalScoring.permute_time_intensity(pertinence_array[c+1])
            self.assertGreaterEqual(permuted_intensity_next, permuted_intensity_current)

    def test_dampen_return(self):
        """
        Test that the dampen return works as expected
        """
        ## want to hand check the first position
        logged_returns = [ PositionUtils.log_transform(x) for x in self.returns_at_close ]
        for c,position in enumerate(self.mock_positions):
            open_time = position.open_ms
            close_time = position.close_ms
            evaluation_time = self.end_time
            return_value = logged_returns[c]
            dampened_return = PositionUtils.dampen_return(
                return_value, 
                open_time, 
                close_time, 
                evaluation_time
            )
            self.assertLessEqual(abs(dampened_return), abs(return_value))