# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
from copy import deepcopy

from data_generator.twelvedata_service import TwelveDataService
from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order

class TestPositions(TestBase):

    def setUp(self):
        super().setUp()
        secrets = ValiUtils.get_secrets()
        secrets["twelvedata_apikey"] = secrets["twelvedata_apikey2"]
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets)
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.position_manager.init_cache_files()
        self.position_manager.clear_all_miner_positions_from_disk()

    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order)
        self.position_manager.save_miner_position_to_disk(position)

    def _find_disk_position_from_memory_position(self, position):
        for disk_position in self.position_manager.get_all_miner_positions(position.miner_hotkey):
            if disk_position.position_uuid == position.position_uuid:
                return disk_position
        raise ValueError(f"Could not find position {position.position_uuid} in disk")

    def validate_intermediate_position_state(self, in_memory_position, expected_state):
        disk_position = self._find_disk_position_from_memory_position(in_memory_position)
        success, reason = PositionManager.positions_are_the_same(in_memory_position, expected_state)
        self.assertTrue(success, "In memory position is not as expected. " + reason)
        success, reason = PositionManager.positions_are_the_same(disk_position, expected_state)
        self.assertTrue(success, "Disc position is not as expected. " + reason)

    def test_simple_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=110,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0978,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_simple_long_position_with_implicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'close_ms': None,
            'return_at_close': 0.998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.996,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_simple_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=90,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0978,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_liquidated_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 10.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.98,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 10.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
    def test_liquidated_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': .998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_liquidated_short_position_with_no_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=.1,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.LONG,
                   leverage=.1,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': .998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        # Orders post-liquidation are ignored
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_liquidated_long_position_with_no_FLAT(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=.1,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=.1,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 10,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.98,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 10,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        # Orders post-liquidation are ignored
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 10,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_simple_short_position_with_implicit_FLAT(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=2.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': .998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.4969999999999999,
            'current_return': 1.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_invalid_leverage_order(self):
        position = deepcopy(self.default_position)
        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                          leverage=0.0,
                                          price=100,
                                          trade_pair=TradePair.BTCUSD,
                                          processed_ms=1000,
                                          order_uuid="1000"))
        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                          leverage=TradePair.BTCUSD.max_leverage + 1,
                                          price=100,
                                          trade_pair=TradePair.BTCUSD,
                                          processed_ms=1000,
                                          order_uuid="1000"))
        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.SHORT,
                                          leverage=TradePair.BTCUSD.max_leverage + 1,
                                          price=100,
                                          trade_pair=TradePair.BTCUSD,
                                          processed_ms=1000,
                                          order_uuid="1000"))
        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                     leverage=-1.0,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"))
        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                     leverage=TradePair.BTCUSD.min_leverage / 2.0,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"))


    def test_invalid_prices_zero(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=0,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        with self.assertRaises(ValueError):
            position.add_order(o1)


    def test_invalid_prices_negative(self):
        with self.assertRaises(ValueError):
            o1 = Order(order_type=OrderType.LONG,
                       leverage=1.0,
                       price=-1,
                       trade_pair=TradePair.BTCUSD,
                       processed_ms=1000,
                       order_uuid="1000")

    def test_three_orders_with_longs_no_drawdown(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                leverage=0.1,
                price=2000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")
        o3 = Order(order_type=OrderType.FLAT,
                leverage=0.0,
                price=2000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=5000,
                order_uuid="5000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': .998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.1,
            'initial_entry_price': 1000,
            'average_entry_price': 1090.9090909090908,
            'close_ms': None,
            'return_at_close': 1.9956,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1090.9090909090908,
            'close_ms': 5000,
            'return_at_close': 1.9956,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_two_orders_with_a_loss(self):
        o1 = Order(order_type=OrderType.LONG,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                leverage=0.0,
                price=500,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': .997,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': 2000,
            'return_at_close': 0.4985,
            'current_return': 0.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_three_orders_with_a_loss_and_then_a_gain(self):
            o1 = Order(order_type=OrderType.LONG,
                    leverage=1.0,
                    price=1000,
                    trade_pair=TradePair.BTCUSD,
                    processed_ms=1000,
                    order_uuid="1000")
            o2 = Order(order_type=OrderType.LONG,
                    leverage=0.1,
                    price=500,
                    trade_pair=TradePair.BTCUSD,
                    processed_ms=2000,
                    order_uuid="2000")
            o3 = Order(order_type=OrderType.SHORT,
                    leverage=0.1,
                    price=1000,
                    trade_pair=TradePair.BTCUSD,
                    processed_ms=5000,
                    order_uuid="5000")

            position = deepcopy(self.default_position)
            self.add_order_to_position_and_save_to_disk(position, o1)
            self.validate_intermediate_position_state(position, {
                'orders': [o1],
                'position_type': OrderType.LONG,
                'is_closed_position': False,
                'net_leverage': 1.0,
                'initial_entry_price': 1000,
                'average_entry_price': 1000,
                'close_ms': None,
                'return_at_close': .998,
                'current_return': 1.0,
                'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
                'open_ms': self.DEFAULT_OPEN_MS,
                'trade_pair': self.DEFAULT_TRADE_PAIR,
                'position_uuid': self.DEFAULT_POSITION_UUID
            })

            self.add_order_to_position_and_save_to_disk(position, o2)
            self.validate_intermediate_position_state(position, {
                'orders': [o1, o2],
                'position_type': OrderType.LONG,
                'is_closed_position': False,
                'net_leverage': 1.1,
                'initial_entry_price': 1000,
                'average_entry_price': 954.5454545454545,
                'close_ms': None,
                'return_at_close': 0.4989,
                'current_return': 0.5,
                'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
                'open_ms': self.DEFAULT_OPEN_MS,
                'trade_pair': self.DEFAULT_TRADE_PAIR,
                'position_uuid': self.DEFAULT_POSITION_UUID
            })

            self.add_order_to_position_and_save_to_disk(position, o3)
            self.validate_intermediate_position_state(position, {
                'orders': [o1, o2, o3],
                'position_type': OrderType.LONG,
                'is_closed_position': False,
                'net_leverage': 1.0,
                'initial_entry_price': 1000,
                'average_entry_price': 950.0,
                'close_ms': None,
                'return_at_close': 1.0479,
                'current_return': 1.05,
                'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
                'open_ms': self.DEFAULT_OPEN_MS,
                'trade_pair': self.DEFAULT_TRADE_PAIR,
                'position_uuid': self.DEFAULT_POSITION_UUID
            })

    def test_returns_on_large_price_increase(self):
        o1 = Order(order_type=OrderType.LONG,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                leverage=0.1,
                price=2000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")
        o3 = Order(order_type=OrderType.LONG,
                leverage=.01,
                price=39000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=3000,
                order_uuid="3000")
        o4 = Order(order_type=OrderType.LONG,
                leverage=.01,
                price=40000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=4000,
                order_uuid="4000")
        o5 = Order(order_type=OrderType.FLAT,
                leverage=0.0,
                price=40000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=5000,
                order_uuid="5000")
        position = deepcopy(self.default_position)

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.add_order_to_position_and_save_to_disk(position, o2)
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.add_order_to_position_and_save_to_disk(position, o4)
        self.add_order_to_position_and_save_to_disk(position, o5)


        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3, o4, o5],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1776.7857142857142,
            'close_ms': 5000,
            'return_at_close': 43.7118656,
            'current_return': 43.81,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })


    def test_two_orders_with_a_loss(self):
        o1 = Order(order_type=OrderType.LONG,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                leverage=0.0,
                price=500,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': .998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': 2000,
            'return_at_close': 0.499,
            'current_return': 0.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_error_adding_mismatched_trade_pair(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.EURNZD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=3.0,
                   price=500,
                   trade_pair=TradePair.CADCHF,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.FLAT,
                   leverage=5.0,
                   price=500,
                   trade_pair=TradePair.SPX,
                   processed_ms=3000,
                   order_uuid="3000")

        for order in [o1, o2, o3]:
            with self.assertRaises(ValueError):
                position.add_order(order)

    def test_two_positions_no_collisions(self):
        trade_pair1 = TradePair.SPX
        hotkey1 = self.DEFAULT_MINER_HOTKEY
        position1 = Position(
            miner_hotkey=hotkey1,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=trade_pair1
        )
        trade_pair2 = TradePair.EURJPY
        hotkey2 = self.DEFAULT_MINER_HOTKEY + '_2'
        position2 = Position(
            miner_hotkey=hotkey2,
            position_uuid=self.DEFAULT_POSITION_UUID + '_2',
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=trade_pair2
        )

        o1 = Order(order_type=OrderType.SHORT,
                leverage=0.4,
                price=1000,
                trade_pair=trade_pair1,
                processed_ms=1000,
                order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                leverage=0.4,
                price=500,
                trade_pair=trade_pair2,
                processed_ms=2000,
                order_uuid="2000")


        self.add_order_to_position_and_save_to_disk(position1, o1)
        self.validate_intermediate_position_state(position1, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -0.4,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': 0.999964,
            'current_return': 1.0,
            'miner_hotkey': position1.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': trade_pair1,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position2, o2)
        self.validate_intermediate_position_state(position2, {
            'orders': [o2],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -0.4,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'close_ms': None,
            'return_at_close': 0.999972,
            'current_return': 1.0,
            'miner_hotkey': position2.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': trade_pair2,
            'position_uuid': self.DEFAULT_POSITION_UUID + '_2'
        })
    def test_leverage_clamping_long(self):
        position = deepcopy(self.default_position)
        live_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=11.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        o2_clamped = deepcopy(o2)
        o2_clamped.leverage = TradePair.BTCUSD.max_leverage - o1.leverage


        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2_clamped],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })


    def test_leverage_clamping_skip_long_order(self):
        position = deepcopy(self.default_position)
        live_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")


        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_leverage_clamping_short(self):
        position = deepcopy(self.default_position)
        live_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-10.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=-11.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        o2_clamped = deepcopy(o2)
        o2_clamped.leverage = -1.0 * (TradePair.BTCUSD.max_leverage - abs(o1.leverage))


        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2_clamped],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1.0 * TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
    def test_leverage_clamping_skip_short_order(self):
        position = deepcopy(self.default_position)
        live_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-self.DEFAULT_TRADE_PAIR.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")


        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_position_json(self):
        position = deepcopy(self.default_position)
        live_price = 100000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=5.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)


        #self.assertEqual(position_json, {})
        dict_repr = position.to_dict()  # Make sure no side effects in the recreated object...
        for x in dict_repr['orders']:
            self.assertFalse('trade_pair' in x, dict_repr)

        position_json = position.to_json_string()
        recreated_object = Position(**json.loads(position_json))
        for x in recreated_object.orders:
            self.assertTrue(hasattr(x, 'trade_pair'), recreated_object)

        recreated_object_json = json.loads(position_json)
        for x in recreated_object_json['orders']:
            self.assertFalse('trade_pair' in x, recreated_object_json)

        #print(f"position json: {position_json}")
        dict_repr = position.to_dict() # Make sure no side effects in the recreated object...

        recreated_object = Position.parse_raw(position_json)#Position(**json.loads(position_json))
        #print(f"recreated object str repr: {recreated_object}")
        #print("recreated object:", recreated_object)
        self.assertTrue(PositionManager.positions_are_the_same(position, recreated_object))
        for x in dict_repr['orders']:
            self.assertFalse('trade_pair' in x, dict_repr)

        for x in recreated_object.orders:
            self.assertTrue(hasattr(x, 'trade_pair'), recreated_object)




if __name__ == '__main__':
    import unittest
    unittest.main()