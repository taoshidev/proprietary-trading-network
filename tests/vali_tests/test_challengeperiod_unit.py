# developer: trdougherty
import copy
from copy import deepcopy
import numpy as np

from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.test_utilities import generate_ledger

from vali_objects.vali_config import TradePair
from vali_objects.position import Position
from vali_objects.vali_config import ValiConfig

from tests.shared_objects.mock_classes import (
    MockMetagraph, MockChallengePeriodManager, MockPositionManager, MockPerfLedgerManager, MockCacheController
)

from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.scoring.scoring import Scoring
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType


class TestChallengePeriodUnit(TestBase):

    def setUp(self):
        super().setUp()

        # For the positions and ledger creation
        self.START_TIME = 0
        self.END_TIME = ValiConfig.CHALLENGE_PERIOD_MS - 1

        # For time management
        self.CURRENTLY_IN_CHALLENGE = ValiConfig.CHALLENGE_PERIOD_MS  # Evaluation time when inside the challenge period
        self.OUTSIDE_OF_CHALLENGE = ValiConfig.CHALLENGE_PERIOD_MS + 1  # Evaluation time when the challenge period is over

        # Number of positions
        self.N_POSITIONS_BOUNDS = 20 + 1
        self.N_POSITIONS = self.N_POSITIONS_BOUNDS - 1

        self.EVEN_TIME_DISTRIBUTION = [self.START_TIME + (self.END_TIME - self.START_TIME) * i / self.N_POSITIONS_BOUNDS
                                       for i
                                       in range(self.N_POSITIONS_BOUNDS)]

        self.MINER_NAMES = [f"miner{i}" for i in range(self.N_POSITIONS)]
        self.SUCCESS_MINER_NAMES = [f"miner{i}" for i in range(1, 5)]
        self.DEFAULT_POSITION = Position(
            miner_hotkey="miner",
            position_uuid="miner",
            orders=[Order(price=60000, processed_ms=self.START_TIME, order_uuid="initial_order", trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],
            net_leverage=0.0,
            open_ms=self.START_TIME,
            close_ms=self.END_TIME,
            trade_pair=TradePair.BTCUSD,
            is_closed_position=True,
            return_at_close=1.0
        )

        # Generate a positions list with N_POSITIONS positions
        self.DEFAULT_POSITIONS = []
        for i in range(self.N_POSITIONS):
            position = deepcopy(self.DEFAULT_POSITION)
            position.open_ms = self.EVEN_TIME_DISTRIBUTION[i]
            position.close_ms = self.EVEN_TIME_DISTRIBUTION[i + 1]
            position.is_closed_position = True
            position.return_at_close = 1.0
            position.orders[0] = Order(price=60000, processed_ms=int(position.open_ms), order_uuid="order" + str(i), trade_pair=TradePair.BTCUSD,  order_type=OrderType.LONG, leverage=0.1)
            self.DEFAULT_POSITIONS.append(position)

        self.DEFAULT_LEDGER = generate_ledger(
            start_time=self.START_TIME,
            end_time=self.END_TIME,
            gain=0.1,
            loss=-0.08,
            mdd=0.99
        )

        self.TOP_SCORE = 1.0
        self.MIN_SCORE = 0.0

        # Set up successful scores for 4 miners
        self.success_scores_dict = {"metrics": {}}

        raw_scores = np.linspace(self.TOP_SCORE, self.MIN_SCORE, len(self.SUCCESS_MINER_NAMES))
        success_scores = list(zip(self.SUCCESS_MINER_NAMES, raw_scores))

        for config_name, config in Scoring.scoring_config.items():
            self.success_scores_dict["metrics"][config_name] = {'scores': copy.deepcopy(success_scores),
                                                     'weight': config['weight']
            }
        raw_penalties = [1 for _ in self.SUCCESS_MINER_NAMES]
        success_penalties = dict(zip(self.SUCCESS_MINER_NAMES, raw_penalties))

        self.success_scores_dict["penalties"] = copy.deepcopy(success_penalties)

        # Initialize system components
        self.mock_metagraph = MockMetagraph(self.MINER_NAMES)
        self.challengeperiod_manager = MockChallengePeriodManager(self.mock_metagraph)
        self.position_manager = MockPositionManager(self.mock_metagraph)
        self.ledger_manager = MockPerfLedgerManager(self.mock_metagraph)
        self.cache_controller = MockCacheController(self.mock_metagraph)
    
    def get_trial_scores(self, high_performing=True, score=None):
        """
        Args:
            high_performing: true means trial miner should be passing, false means they should be failing
            score: specific score to use
        """
        trial_scores_dict = {"metrics": {}}
        trial_metrics = trial_scores_dict["metrics"]
        if score is not None:
            for config_name, config in Scoring.scoring_config.items():
                trial_metrics[config_name] = {'scores': [("miner", score)],
                                              'weight': config['weight']
            }
        elif high_performing:
            for config_name, config in Scoring.scoring_config.items():
                trial_metrics[config_name] = {'scores': [("miner", self.TOP_SCORE)],
                                              'weight': config['weight']
            }
        else:
            for config_name, config in Scoring.scoring_config.items():
                trial_metrics[config_name] = {'scores': [("miner", self.MIN_SCORE)],
                                              'weight': config['weight']
                                                  }
        trial_scores_dict["penalties"] = {"miner": 1}
        return trial_scores_dict


    def test_screen_drawdown(self):
        """Test that a high drawdown miner is screened"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        # Returns are strong and consistent
        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.1

        # Drawdown is high - 50% drawdown on the first period
        base_ledger.cps[0].mdd = 0.5

        # Drawdown criteria
        max_drawdown = LedgerUtils.recent_drawdown(base_ledger.cps, restricted=False)
        max_drawdown_percentage = LedgerUtils.drawdown_percentage(max_drawdown)
        self.assertGreater(max_drawdown_percentage, ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE)

        # Check that the miner is successfully screened as failing
        screening_logic, _ = self.challengeperiod_manager.screen_failing_criteria(ledger_element=base_ledger)
        self.assertTrue(screening_logic)


    # ------ Time Constrained Tests (Inspect) ------
    def test_failing_remaining_time(self):
        """Miner is not passing, but there is time remaining"""
        trial_scoring_dict = self.get_trial_scores(high_performing=False)
        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            success_scores_dict=self.success_scores_dict,
            inspection_scores_dict=trial_scoring_dict
        )
        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", failing)
    
    def test_failing_no_remaining_time(self):
        """Miner is not passing, and there is no time remaining"""
        
        trial_scoring_dict = self.get_trial_scores(high_performing=False)

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            success_scores_dict=self.success_scores_dict,
            inspection_scores_dict=trial_scoring_dict
        )

        self.assertNotIn("miner", passing)
        self.assertIn("miner", failing)

    def test_passing_remaining_time(self):
        """Miner is passing and there is remaining time - they should be promoted"""

        trial_scoring_dict = self.get_trial_scores(high_performing=True)

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.CURRENTLY_IN_CHALLENGE

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            success_scores_dict=self.success_scores_dict,
            inspection_scores_dict=trial_scoring_dict
        )

        self.assertIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_passing_no_remaining_time(self):
        """Redemption, if they pass right before the challenge period ends and before the next evaluation cycle"""
        trial_scoring_dict = self.get_trial_scores(high_performing=True)

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            success_scores_dict=self.success_scores_dict,
            inspection_scores_dict=trial_scoring_dict
        )

        self.assertListEqual(passing, ["miner"])
        self.assertListEqual(failing, [])

        self.assertIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_lingering_no_positions(self):
        """Test the scenario where the miner has no positions and has been in the system for a while"""
        base_positions = []

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as testing still
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time
        )

        self.assertNotIn("miner", passing)
        self.assertIn("miner", failing)

    def test_recently_re_registered_miner(self):
        """
        Test the scenario where the miner is eliminated and registers again. Simulate this with a stale perf ledger
        The positions begin after the perf ledger start therefore the ledger is stale.
        """

        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        base_position = deepcopy(self.DEFAULT_POSITION)
        base_position.orders[0].processed_ms = base_ledger.start_time_ms + 1
        base_positions = [base_position]

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as testing still
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=self.SUCCESS_MINER_NAMES,
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            success_scores_dict=self.success_scores_dict
        )

        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_lingering_with_positions(self):
        """Test the scenario where the miner has positions and has been in the system for a while"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_positions = [base_positions[0]]  # Only one position

        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as testing still
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=self.SUCCESS_MINER_NAMES,
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            success_scores_dict=self.success_scores_dict
        )

        self.assertNotIn("miner", passing)
        self.assertIn("miner", failing)
    def test_just_above_threshold(self):

        # Miner performing 80th percentile should pass
        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}

        trial_scoring_dict = self.get_trial_scores(score=0.75)

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            success_scores_dict=self.success_scores_dict,
            inspection_scores_dict=trial_scoring_dict
        )
        self.assertIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_just_below_threshold(self):

        # Miner performing 60th percentile should fail, but continue testing
        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}

        trial_scoring_dict = self.get_trial_scores(score=0.5)

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            success_scores_dict=self.success_scores_dict,
            inspection_scores_dict=trial_scoring_dict
        )
        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_at_threshold(self):
        # Miner performing exactly at 75th percentile should pass
        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}

        trial_scoring_dict = self.get_trial_scores(score=0.75)

        success_scores_dict = {"metrics": {}}
        success_miner_names = self.SUCCESS_MINER_NAMES[1:]
        raw_scores = np.linspace(self.TOP_SCORE, self.MIN_SCORE, len(success_miner_names))
        success_scores = list(zip(success_miner_names, raw_scores))

        for config_name, config in Scoring.scoring_config.items():
            success_scores_dict["metrics"][config_name] = {'scores': copy.deepcopy(success_scores),
                                                           'weight': config['weight']
                                                          }
        raw_penalties = [1 for _ in success_miner_names]
        success_penalties = dict(zip(success_miner_names, raw_penalties))

        success_scores_dict["penalties"] = copy.deepcopy(success_penalties)

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            success_scores_dict=success_scores_dict,
            inspection_scores_dict=trial_scoring_dict
        )

        self.assertIn("miner", passing)
        self.assertNotIn("miner", failing)


