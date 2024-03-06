# developer: jbonilla
# Copyright © 2023 Taoshi Inc

from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order


class TestPositions(TestBase):

    def setUp(self):
        super().setUp()
        self.MINER_HOTKEY = "test_miner"
        self.TEST_POSITION_UUID = "test_position"
        self.OPEN_MS = 1000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.position = Position(
            miner_hotkey=self.MINER_HOTKEY,
            position_uuid=self.TEST_POSITION_UUID,
            open_ms=self.OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )

    def validate_intermediate_position_state(self, expected_state):
        for attr in ['orders', 'position_type', 'is_closed_position', '_net_leverage', 
                     '_initial_entry_price', '_average_entry_price', 'max_drawdown', 
                     'close_ms', 'return_at_close', 'current_return', 'miner_hotkey', 
                     'open_ms', 'trade_pair']:
            expected_value = expected_state.get(attr)
            actual_value = getattr(self.position, attr, None)
            self.assertEqual(actual_value, expected_value,
                             f"Expected {attr} to be {expected_value}, got {actual_value}")


    def test_simple_long_position_with_explicit_FLAT(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0,
                   price=110,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.position.add_order(o1)
        self.validate_intermediate_position_state({
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            '_net_leverage': 1,
            '_initial_entry_price': 100,
            '_average_entry_price': 100,
            'max_drawdown': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

        self.position.add_order(o2)
        self.validate_intermediate_position_state({
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            '_net_leverage': 0,
            '_initial_entry_price': 100,
            '_average_entry_price': 100,
            'max_drawdown': 0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0967,
            'current_return': 1.1,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

    def test_simple_long_position_with_implicit_FLAT(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=-1,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.position.add_order(o1)
        self.validate_intermediate_position_state({
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            '_net_leverage': 1,
            '_initial_entry_price': 500,
            '_average_entry_price': 500,
            'max_drawdown': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

        self.position.add_order(o2)
        self.validate_intermediate_position_state({
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            '_net_leverage': 0,
            '_initial_entry_price': 500,
            '_average_entry_price': 500,
            'max_drawdown': 0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.994,
            'current_return': 2.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

    def test_simple_short_position_with_explicit_FLAT(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-1,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0,
                   price=90,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.position.add_order(o1)
        self.validate_intermediate_position_state({
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            '_net_leverage': -1,
            '_initial_entry_price': 100,
            '_average_entry_price': 100,
            'max_drawdown': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

        self.position.add_order(o2)
        self.validate_intermediate_position_state({
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            '_net_leverage': 0,
            '_initial_entry_price': 100,
            '_average_entry_price': 100,
            'max_drawdown': 0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0967,
            'current_return': 1.1,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

    def test_simple_short_position_with_implicit_FLAT(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-1,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=2,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.position.add_order(o1)
        self.validate_intermediate_position_state({
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            '_net_leverage': -1,
            '_initial_entry_price': 1000,
            '_average_entry_price': 1000,
            'max_drawdown': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

        self.position.add_order(o2)
        self.validate_intermediate_position_state({
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            '_net_leverage': 0,
            '_initial_entry_price': 1000,
            '_average_entry_price': 1000,
            'max_drawdown': 0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.4955,
            'current_return': 1.5,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

    def test_zero_leverage_order(self):
        with self.assertRaises(ValueError):
            self.position.add_order(Order(order_type=OrderType.LONG,
                                          leverage=0,
                                          price=100,
                                          trade_pair=TradePair.BTCUSD,
                                          processed_ms=1000,
                                          order_uuid="1000"))

        self.validate_intermediate_position_state({
            'orders': [],
            'position_type': None,
            'is_closed_position': False,
            '_net_leverage': 0,
            '_initial_entry_price': 0,
            '_average_entry_price': 0,
            'max_drawdown': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

    def test_invalid_prices_zero(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1,
                   price=0,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        with self.assertRaises(ValueError):
            self.position.add_order(o1)

        self.validate_intermediate_position_state({
            'orders': [o1],
            'position_type': None,
            'is_closed_position': False,
            '_net_leverage': 0,
            '_initial_entry_price': 0,
            '_average_entry_price': 0,
            'max_drawdown': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })

    def test_invalid_prices_negative(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1,
                   price=-1,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        with self.assertRaises(ValueError):
            self.position.add_order(o1)

        self.validate_intermediate_position_state({
            'orders': [o1],
            'position_type': None,
            'is_closed_position': False,
            '_net_leverage': 0,
            '_initial_entry_price': -1,
            '_average_entry_price': 0,
            'max_drawdown': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': self.MINER_HOTKEY,
            'open_ms': self.OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR
        })


if __name__ == '__main__':
    import unittest
    unittest.main()