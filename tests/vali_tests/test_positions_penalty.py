from copy import deepcopy
from tests.shared_objects.mock_classes import MockMetagraph
from tests.shared_objects.test_utilities import add_orders_to_position
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.position_penalties import PositionPenalties


class TestPositionsPenalty(TestBase):
    """
    This class will only test penalties that apply to positions
    """

    def setUp(self):
        super().setUp()

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_CLOSE_MS = 2000
        self.DEFAULT_ORDER_MS = 1000
        self.MS_IN_DAY = 86400000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_position = Position(
            position_type=OrderType.LONG,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.position_manager.clear_all_miner_positions()

    def tearDown(self):
        self.position_manager.clear_all_miner_positions()
        super().tearDown()

    def test_martingale_penalty_general(self):

        # Base case
        self.assertAlmostEqual(PositionPenalties.martingale_penalty([]), 1, places=1)

        # Test Martingale penalty with LONG position
        position1 = deepcopy(self.default_position)

        leverages = [0.1, 0.1, 0.1]
        prices = [100, 50, 50]
        times = [(i + 1) * self.DEFAULT_OPEN_MS for i in range(len(leverages))]

        add_orders_to_position(
            position=position1,
            order_type=OrderType.LONG,
            trade_pair=TradePair.BTCUSD,
            leverages=leverages,
            prices=prices,
            times=times)

        position1.return_at_close = 1.05
        self.assertLess(PositionPenalties.martingale_penalty([position1], evaluation_time_ms=3 * self.DEFAULT_OPEN_MS), 0.5)

        # SHORT positions should also count towards martingale
        position2 = deepcopy(self.default_position)
        position2.position_type = OrderType.SHORT
        leverages = [-0.1, -0.1, -0.1]
        prices = [100, 110, 110]
        times = [(i + 1) * self.DEFAULT_OPEN_MS for i in range(len(leverages))]

        add_orders_to_position(
            position=position2,
            order_type=OrderType.SHORT,
            trade_pair=TradePair.BTCUSD,
            leverages=leverages,
            prices=prices,
            times=times)

        position2.return_at_close = 1.05

        # Should be max penalty
        self.assertAlmostEqual(PositionPenalties.martingale_penalty([position2]), 0.0)
        penalty_two_martingale = PositionPenalties.martingale_penalty([position1, position2])

        self.assertAlmostEqual(penalty_two_martingale, 0.0)

        # Add a third position that is non-martingale which should decrease the penalty
        position3 = deepcopy(self.default_position)
        position3.position_type = OrderType.SHORT

        # Although leverage is increasing past the maximum, order is in a winning order, so not a martinagale
        leverages = [-0.1, -0.1, -0.1]
        prices = [100, 90, 110]
        times = [(i + 1) * self.DEFAULT_OPEN_MS for i in range(len(leverages))]

        add_orders_to_position(
            position=position3,
            order_type=OrderType.SHORT,
            trade_pair=TradePair.BTCUSD,
            leverages=leverages,
            prices=prices,
            times=times)

        position3.return_at_close = 1.05

        # Penalty with only non-martingale position should be 1
        self.assertAlmostEqual(PositionPenalties.martingale_penalty([position3]), 1.0, places=1)

        # Penalty with 2 martingale and 1 non-martingale should be less than the penalty for a miner with only 2 martingale positions

        # Reminder that a return value of 1 => no penalty and return value of 0 => max penalty
        self.assertGreater(PositionPenalties.martingale_penalty([position1, position2, position3]), penalty_two_martingale)