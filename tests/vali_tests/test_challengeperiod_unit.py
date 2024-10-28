# developer: trdougherty
from copy import deepcopy
import math

from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.test_utilities import generate_ledger

from vali_objects.vali_config import TradePair
from vali_objects.position import Position
from vali_objects.vali_config import ValiConfig

from tests.shared_objects.mock_classes import (
    MockMetagraph, MockChallengePeriodManager, MockPositionManager, MockPerfLedgerManager, MockCacheController
)

from vali_objects.utils.position_filtering import PositionFiltering
from vali_objects.utils.position_penalties import PositionPenalties
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
        self.N_POSITIONS_BOUNDS = ValiConfig.CHALLENGE_PERIOD_MIN_POSITIONS + 1
        self.N_POSITIONS = self.N_POSITIONS_BOUNDS - 1

        self.EVEN_TIME_DISTRIBUTION = [self.START_TIME + (self.END_TIME - self.START_TIME) * i / self.N_POSITIONS_BOUNDS
                                       for i
                                       in range(self.N_POSITIONS_BOUNDS)]

        self.MINER_NAMES = [f"miner{i}" for i in range(self.N_POSITIONS)]
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

        # Initialize system components
        self.mock_metagraph = MockMetagraph(self.MINER_NAMES)
        self.challengeperiod_manager = MockChallengePeriodManager(self.mock_metagraph)
        self.position_manager = MockPositionManager(self.mock_metagraph)
        self.ledger_manager = MockPerfLedgerManager(self.mock_metagraph)
        self.cache_controller = MockCacheController(self.mock_metagraph)

    def test_fail_return_open_positions(self):
        """Test the scenario where the position is closed with a positive return and positions are still open with negative returns"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS) + deepcopy(self.DEFAULT_POSITIONS)

        # Positive positions - 30% of the positions
        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.05
            base_positions[i].is_closed_position = True

        # Negative positions - 70% of the positions
        for i in range(self.N_POSITIONS, self.N_POSITIONS + self.N_POSITIONS // 2):
            base_positions[i].return_at_close = 0.7
            base_positions[i].close_ms = None
            base_positions[i].is_closed_position = False

        # Check that the miner fails the returns criteria
        filtered_positions = PositionFiltering.filter_single_miner(
            base_positions,
            evaluation_time_ms=self.END_TIME,
            lookback_time_ms=ValiConfig.CHALLENGE_PERIOD_MS
        )

        # Returns criteria
        base_return = Scoring.base_return(filtered_positions)
        self.assertLessEqual(base_return, ValiConfig.CHALLENGE_PERIOD_RETURN)

        # Check that the miner is screened as failing
        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=base_positions,
            ledger_element=self.DEFAULT_LEDGER,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertFalse(screening_logic)

    def test_fail_return_number_of_closed_positions(self):
        """Test the scenario where the position is closed with a positive return and positions are still open with negative returns"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)

        # strong returns
        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.05

        # half of the positions are still open
        for i in range(self.N_POSITIONS // 2, self.N_POSITIONS):
            base_positions[i].close_ms = None
            base_positions[i].is_closed_position = False

        # Check that the miner is screened as failing
        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=base_positions,
            ledger_element=self.DEFAULT_LEDGER,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertFalse(screening_logic)

    def test_pass_return_open_positions(self):
        """Test the scenario where the position is closed with a positive return and positions are still open with negative returns"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        # Positive positions - 50% of the positions
        for i in range(self.N_POSITIONS // 2):
            base_positions[i].return_at_close = 1.3
            base_positions[i].is_closed_position = True

        # Negative positions - 50% of the positions
        for i in range(self.N_POSITIONS // 2, self.N_POSITIONS):
            base_positions[i].return_at_close = 0.99
            base_positions[i].close_ms = self.END_TIME
            base_positions[i].is_closed_position = True

        # Now inject a big loss that's still open
        big_loss = deepcopy(self.DEFAULT_POSITION)
        big_loss.return_at_close = 0.5
        big_loss.close_ms = None
        big_loss.is_closed_position = False

        base_positions.append(big_loss)

        # Check that the miner fails the returns criteria
        filtered_positions = PositionFiltering.filter_single_miner(
            base_positions,
            evaluation_time_ms=self.CURRENTLY_IN_CHALLENGE,
            lookback_time_ms=ValiConfig.CHALLENGE_PERIOD_MS
        )

        # Returns criteria
        base_return = Scoring.base_return(filtered_positions)
        base_return_percentage = (math.exp(base_return) - 1) * 100

        self.assertGreaterEqual(base_return, ValiConfig.CHALLENGE_PERIOD_RETURN_LOG)
        self.assertGreaterEqual(base_return_percentage, ValiConfig.CHALLENGE_PERIOD_RETURN_PERCENTAGE)

        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=filtered_positions,
            ledger_element=base_ledger,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertTrue(screening_logic)

    def test_fail_n_closed_positions(self):
        """Test the scenario where the position is closed with a positive return and positions are still open with negative returns"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)

        # Positive positions - 30% of the positions
        for i in range(self.N_POSITIONS // 3):
            base_positions[i].return_at_close = 1.3
            base_positions[i].is_closed_position = True

        base_positions = base_positions[:self.N_POSITIONS // 3]

        # Check that the miner fails the returns criteria
        filtered_positions = PositionFiltering.filter_single_miner(
            base_positions,
            evaluation_time_ms=self.CURRENTLY_IN_CHALLENGE,
            lookback_time_ms=ValiConfig.CHALLENGE_PERIOD_MS
        )

        # Returns criteria
        self.assertLess(len(filtered_positions), ValiConfig.CHALLENGE_PERIOD_MIN_POSITIONS)

        # Check that the miner is screened as failing
        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=base_positions,
            ledger_element=self.DEFAULT_LEDGER,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertFalse(screening_logic)

    def test_fail_returns_ratio(self):
        """Test the scenario where the position is closed with a positive return and positions are still open with negative returns"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_positions[0].return_at_close = 1.3

        # Positive positions - 30% of the positions
        for i in range(1, self.N_POSITIONS):
            base_positions[i].return_at_close = 1.0

        # Check that the miner fails the returns criteria
        filtered_positions = PositionFiltering.filter_single_miner(
            base_positions,
            evaluation_time_ms=self.CURRENTLY_IN_CHALLENGE,
            lookback_time_ms=ValiConfig.CHALLENGE_PERIOD_MS
        )

        # Returns ratio
        return_ratio = PositionPenalties.returns_ratio(filtered_positions)

        # Higher returns ratio is worse, this is a failing test
        self.assertGreater(return_ratio, ValiConfig.CHALLENGE_PERIOD_MAX_POSITIONAL_RETURNS_RATIO)

        # Check that the miner is screened as failing
        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=base_positions,
            ledger_element=self.DEFAULT_LEDGER,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertFalse(screening_logic)

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
        self.assertGreater(max_drawdown_percentage, ValiConfig.CHALLENGE_PERIOD_MAX_DRAWDOWN_PERCENT)

        # Check that the miner is successfully screened as failing
        screening_logic, _ = self.challengeperiod_manager.screen_failing_criteria(ledger_element=base_ledger)
        self.assertTrue(screening_logic)

    def test_pass_returns_ratio(self):
        """Test that a high returns ratio miner is screened"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        # Returns are strong and consistent
        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.01

        # Returns ratio is high
        return_ratio = PositionPenalties.returns_ratio(base_positions)
        self.assertLess(return_ratio, ValiConfig.CHALLENGE_PERIOD_MAX_POSITIONAL_RETURNS_RATIO)

        # Check that the miner is screened as failing
        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=base_positions,
            ledger_element=base_ledger,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertTrue(screening_logic)

    def test_fail_screen_returns(self):
        """Test that low returns are screened"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        # Returns are weak and consistent
        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.0

        # Check that the miner fails the returns criteria
        filtered_positions = PositionFiltering.filter_single_miner(
            base_positions,
            evaluation_time_ms=self.CURRENTLY_IN_CHALLENGE,
            lookback_time_ms=ValiConfig.CHALLENGE_PERIOD_MS
        )

        # Returns criteria
        base_return = Scoring.base_return(filtered_positions)
        self.assertLessEqual(base_return, ValiConfig.CHALLENGE_PERIOD_RETURN)

        # Check that the miner is screened as failing
        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=base_positions,
            ledger_element=base_ledger,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertFalse(screening_logic)

    def test_pass_screen_returns(self):
        """Test that high returns are screened"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        # Returns are strong and consistent
        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.1

        # Check that the miner passes the returns criteria
        filtered_positions = PositionFiltering.filter_single_miner(
            base_positions,
            evaluation_time_ms=self.CURRENTLY_IN_CHALLENGE,
            lookback_time_ms=ValiConfig.CHALLENGE_PERIOD_MS
        )

        # Returns criteria
        base_return = Scoring.base_return(filtered_positions)
        self.assertGreater(base_return, ValiConfig.CHALLENGE_PERIOD_RETURN_LOG)

        # Check that the miner is screened as failing
        screening_logic = self.challengeperiod_manager.screen_passing_criteria(
            position_elements=base_positions,
            ledger_element=base_ledger,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertTrue(screening_logic)

    # ------ Time Constrained Tests (Inspect) ------
    def test_failing_remaining_time(self):
        """Miner is not passing, but there is time remaining"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_positions = base_positions[:self.N_POSITIONS // 2]  # Half of the required positions

        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.CURRENTLY_IN_CHALLENGE

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time
        )

        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_failing_no_remaining_time(self):
        """Miner is not passing, and there is no time remaining"""
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
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time
        )

        self.assertNotIn("miner", passing)
        self.assertIn("miner", failing)

    def test_passing_remaining_time(self):
        """Miner is passing and there is remaining time - they should be promoted"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)

        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.1
            base_positions[i].is_closed_position = True

        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.CURRENTLY_IN_CHALLENGE

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time
        )

        self.assertIn("miner", passing)
        self.assertNotIn("miner", failing)

    def test_passing_no_remaining_time(self):
        """Redemption, if they pass right before the challenge period ends and before the next evaluation cycle"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)

        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.1
            base_positions[i].is_closed_position = True

        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions = {"miner": base_positions}
        inspection_ledger = {"miner": base_ledger}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as failing
        passing, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time
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
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time
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
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time
        )

        self.assertNotIn("miner", passing)
        self.assertIn("miner", failing)
