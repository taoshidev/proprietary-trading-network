from copy import deepcopy
from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.position_penalties import PositionPenalties
import numpy as np
import random


class TestPositionsPenalty(TestBase):
    """
    This class will only test the positions and the consistency metrics associated with positions.
    """

    def setUp(self):
        super().setUp()
        # seeding
        np.random.seed(0)
        random.seed(0)

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
        self.position_manager.init_cache_files()
        self.position_manager.clear_all_miner_positions_from_disk()

    def tearDown(self):
        self.position_manager.clear_all_miner_positions_from_disk()
        super().tearDown()

    def test_max_positional_return_ratio(self):
        position1 = deepcopy(self.default_position)
        position2 = deepcopy(self.default_position)
        position3 = deepcopy(self.default_position)

        o1 = Order(
            order_type=OrderType.LONG,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="1000"
        )

        o2 = Order(
            order_type=OrderType.FLAT,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS + self.DEFAULT_ORDER_MS,
            order_uuid="1001"
        )

        o3 = Order(
            order_type=OrderType.FLAT,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS + self.DEFAULT_ORDER_MS + self.MS_IN_DAY * 3,
            order_uuid="1002"
        )

        o4 = Order(
            order_type=OrderType.FLAT,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS + self.DEFAULT_ORDER_MS + self.MS_IN_DAY + self.DEFAULT_ORDER_MS * 6,
            order_uuid="1003"
        )

        position1.add_order(o1)
        position1.add_order(o2)
        position1.return_at_close = 1.05

        position2.add_order(o1)
        position2.add_order(o3)
        position2.return_at_close = 1.05

        position3.add_order(o1)
        position3.add_order(o4)
        position3.return_at_close = 1.05

        # One position scenario
        self.assertAlmostEqual(PositionPenalties.returns_ratio([position1]), 1.0)
        self.assertAlmostEqual(PositionPenalties.returns_ratio_penalty([position1]), 0.0)

        # Two positions scenario
        self.assertAlmostEqual(PositionPenalties.returns_ratio([position1, position2]), 0.5)
        self.assertGreater(PositionPenalties.returns_ratio_penalty([position1, position2]), 0.0)

        # Three positions scenario
        self.assertAlmostEqual(PositionPenalties.returns_ratio([position1, position2, position3]), 1.0 / 3.0)
        self.assertGreater(PositionPenalties.returns_ratio_penalty([position1, position2, position3]), 0.0)
        self.assertGreater(
            PositionPenalties.returns_ratio_penalty([position1, position2, position3]),
            PositionPenalties.returns_ratio_penalty([position1, position2])
        )

    def test_positional_time(self):
        position1 = deepcopy(self.default_position)
        position2 = deepcopy(self.default_position)
        position3 = deepcopy(self.default_position)

        o1 = Order(
            order_type=OrderType.LONG,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="1000"
        )

        o2 = Order(
            order_type=OrderType.FLAT,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS + self.DEFAULT_ORDER_MS,
            order_uuid="1001"
        )

        o3 = Order(
            order_type=OrderType.LONG,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS + self.MS_IN_DAY * 7 + self.DEFAULT_ORDER_MS,
            order_uuid="1002"
        )

        o4 = Order(
            order_type=OrderType.FLAT,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=self.DEFAULT_OPEN_MS + self.MS_IN_DAY * 14 + self.DEFAULT_ORDER_MS,
            order_uuid="1003"
        )

        position1.add_order(o1)
        position1.add_order(o2)
        position1.return_at_close = 1.05

        position2.add_order(o1)
        position2.add_order(o2)
        position2.return_at_close = 1.05

        position3.add_order(o1)
        position3.add_order(o3)
        position3.add_order(o4)
        position3.return_at_close = 1.10

        # One position scenario
        self.assertAlmostEqual(PositionPenalties.time_consistency_ratio([position1]), 1.0)
        self.assertAlmostEqual(PositionPenalties.time_consistency_penalty([position1]), 0.0)

        # Two positions scenario - closed at the same time, should still get the full penalty
        self.assertAlmostEqual(PositionPenalties.time_consistency_ratio([position1, position2]), 1.0)
        self.assertAlmostEqual(PositionPenalties.time_consistency_penalty([position1, position2]), 0.0)

        # Two positions scenario - closed at different times
        self.assertLessEqual(PositionPenalties.time_consistency_ratio([position1, position3]), 1.0)
        self.assertGreater(PositionPenalties.time_consistency_ratio([position1, position3]), 0.0)

        self.assertGreater(PositionPenalties.time_consistency_penalty([position1, position3]), 0.0)

        # Three positions should have a lower penalty than two positions
        self.assertGreater(
            PositionPenalties.time_consistency_penalty([position1, position3]),
            PositionPenalties.time_consistency_penalty([position1, position2])
        )

