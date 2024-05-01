# developer: trdougherty
# Copyright © 2024 Taoshi Inc
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.scoring.scoring import Scoring
from vali_objects.position import Position
import pickle
import uuid
import hashlib
import math
import copy
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType
from vali_config import ValiConfig, TradePair
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.scoring.historical_scoring import HistoricalScoring

import numpy as np
import random

from tests.shared_objects.test_utilities import get_time_in_range, hash_object, order_generator, position_generator, ledger_generator, checkpoint_generator

class TestWeights(TestBase):
    def setUp(self):
        super().setUp()

        possible_tradepairs = list(TradePair.__members__)    

        ## seeding
        np.random.seed(0)
        random.seed(0)

        n_positions = 50
        self.n_positions = n_positions

        self.start_time = 1710521764446
        self.end_time = self.start_time + ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_MS

        self.start_times = sorted([ get_time_in_range(x, self.start_time, self.end_time) for x in np.random.rand(n_positions) ])
        self.end_times = sorted([ get_time_in_range(x, self.start_times[c], self.end_time) for c,x in enumerate(np.random.rand(n_positions)) ])

        self.returns_at_close = [ 1 + (np.random.rand() - 0.5) * 0.1 for _ in range(n_positions) ]
        self.log_returns_at_close = [ math.log(x) for x in self.returns_at_close ]

        self.gains = [ x if x > 0 else 0 for x in self.returns_at_close ]
        self.losses = [ x if x < 0 else 0 for x in self.returns_at_close ]

        ## Standard checkpoint list
        self.standard_checkpoints = [
            checkpoint_generator(
                last_update_ms=self.start_times[c],
                open_ms=self.end_times[c]-self.start_times[c],
                n_updates=2,
                gain=self.gains[c],
                loss=self.losses[c],
            ) for c in range(n_positions)
        ]

        ## positions in the last 80% of the time range
        imbalanced_checkpoint_close_times = sorted([
            get_time_in_range(
                x, 
                self.start_time + ((self.end_time - self.start_time) * 0.8), 
                self.end_time) 
            for x in np.random.rand(n_positions)
        ])

        ## this should score much lower
        self.imbalanced_checkpoints = [
            checkpoint_generator(
                last_update_ms=imbalanced_checkpoint_close_times[c],
                open_ms=self.start_times[c] - imbalanced_checkpoint_close_times[c],
                n_updates=2,
                gain=self.gains[c],
                loss=self.losses[c],
            ) for c in range(n_positions)
        ]

        self.empty_checkpoints = []

    def test_compute_consistency_penalty(self):
        """
        Test that the consistency penalty works as expected
        """
        evaluation_time = self.end_time

        penalty = PositionUtils.compute_consistency_penalty_cps(
            self.standard_checkpoints, 
            evaluation_time
        )
        self.assertGreaterEqual(penalty, 0)
        self.assertLessEqual(penalty, 1)

    def test_compute_consistency_penalty_empty(self):
        """
        Test that the consistency penalty works as expected when the close times are empty
        """
        evaluation_time = self.end_time
        penalty = PositionUtils.compute_consistency_penalty_cps(
            self.empty_checkpoints, 
            evaluation_time
        )
        self.assertEqual(penalty, 0) # if no score then we multiply everything by zero

    def test_compute_consistency_penalty_single(self):
        """
        Test that the consistency penalty works as expected when there is only one position
        """
        evaluation_time = self.end_time
        penalty = PositionUtils.compute_consistency_penalty_cps(
            [self.standard_checkpoints[0]], 
            evaluation_time
        )

        self.assertGreater(penalty, 0)
        self.assertLessEqual(penalty, 0.8)

    def test_compute_consistency_penalty_known(self):
        """
        Test that the consistency penalty works as expected when there is only one position
        """
        evaluation_time = self.end_time

        imbalanced_penalty = PositionUtils.compute_consistency_penalty_cps(
            self.imbalanced_checkpoints, 
            evaluation_time
        )

        balanced_penalty = PositionUtils.compute_consistency_penalty_cps(
            self.standard_checkpoints, 
            evaluation_time
        )

        # want to make sure that the imbalanced penalty will multiply returns with a lower value
        self.assertLess(imbalanced_penalty, balanced_penalty)

    def test_compute_consistency_hand_check(self):
        """
        Test that the consistency penalty works as expected when there is only one position
        """
        n_positions = 100
        balanced_checkpoint_measured = np.ones(n_positions, dtype=bool)

        imbalanced_checkpoint_measured = copy.deepcopy(balanced_checkpoint_measured)
        imbalanced_checkpoint_measured[0:70] = False # Miner did not record duration for the first 70% of checkpoints

        balanced_gains = np.ones(n_positions) * 0.05
        balanced_losses = np.ones(n_positions) * -0.04
        balanced_times = np.ones(n_positions, dtype=int) * 1000

        imbalanced_gains = balanced_gains * imbalanced_checkpoint_measured
        imbalanced_losses = balanced_losses * imbalanced_checkpoint_measured
        imbalanced_times = balanced_times * imbalanced_checkpoint_measured

        update_times = np.linspace(self.start_time, self.end_time, n_positions, dtype=int)
        evaluation_time = self.end_time

        balanced_checkpoints = [
            checkpoint_generator(
                last_update_ms=update_times[c],
                open_ms=balanced_times[c],
                n_updates=2,
                gain=balanced_gains[c],
                loss=balanced_losses[c],
            ) for c in range(n_positions)
        ]

        imbalanced_checkpoints = [
            checkpoint_generator(
                last_update_ms=update_times[c],
                open_ms=imbalanced_times[c],
                n_updates=2,
                gain=imbalanced_gains[c],
                loss=imbalanced_losses[c],
            ) for c in range(n_positions)
        ]

        standard_penalty = PositionUtils.compute_consistency_penalty_cps(
            balanced_checkpoints, 
            evaluation_time
        )

        imbalanced_penalty = PositionUtils.compute_consistency_penalty_cps(
            imbalanced_checkpoints, 
            evaluation_time
        )

        # want to make sure that the imbalanced penalty will multiply returns with a lower value
        self.assertLess(imbalanced_penalty, standard_penalty)

    def test_compute_lookback_fraction(self):
        """
        Test that the compute lookback fraction works as expected
        """
        # check the first position manually
        checkpoint = self.standard_checkpoints[0]

        evaluation_time = 1710523564446
        time_delta = ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_MS

        close_time = evaluation_time - time_delta + int(time_delta * 0.3) # 30% of the way through the lookback period
        open_time = evaluation_time - time_delta + int(time_delta * 0.1) # 10% of the way through the lookback period

        lookback_fraction = PositionUtils.compute_lookback_fraction(open_time, close_time, evaluation_time)
        self.assertAlmostEqual(lookback_fraction, 0.7, places=5)

        for checkpoint in self.standard_checkpoints[1:]:
            open_time = 0
            close_time = checkpoint.last_update_ms
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
        for c,checkpoint in enumerate(self.standard_checkpoints):
            open_time = 0
            close_time = checkpoint.last_update_ms
            evaluation_time = self.end_time
            return_value = self.log_returns_at_close[c]
            dampened_return = PositionUtils.dampen_return(
                return_value, 
                open_time, 
                close_time, 
                evaluation_time
            )
            self.assertLessEqual(abs(dampened_return), abs(return_value))