import unittest
from typing import List, Dict

from vali_objects.dataclasses.order import Order
from vali_objects.dataclasses.test_position import Position
from vali_objects.enums.order_type_enum import OrderTypeEnum
from vali_objects.utils.position_utils import ExchangeUtils


class TestExchangeUtils(unittest.TestCase):

    @staticmethod
    def return_test_position(order_details: List[List[float | OrderTypeEnum | float]],
                             closing_price):

        orders = [Order(
                leverage=order_detail[0],
                order_type=order_detail[1],
                order_uuid="123",
                price=order_detail[2],
                processed_ms=123,
                trade_pair="BTC/USD"
                ) for order_detail in order_details]
        return Position(
            miner_hotkey="123",
            open_ms=123,
            orders=orders,
            position_uuid="test",
            trade_pair="BTC/USD",
            close_price=closing_price
        )

    def test_calculate_position_return_long_gain(self):
        # test calculating long only in profit
        # with an order that was flat in return
        _position = TestExchangeUtils.return_test_position([[1.0, OrderTypeEnum.LONG, 100.0],
                                                            [1.0, OrderTypeEnum.LONG, 110.0]], 110)

        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=110)
        self.assertEqual(1.0934099, positional_return)

        # test calculating long only at a loss
        _position = TestExchangeUtils.return_test_position([[1.0, OrderTypeEnum.LONG, 100.0],
                                                            [1.0, OrderTypeEnum.LONG, 110.0]], 90)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=90)
        self.assertAlmostEqual(0.7319520818181818, positional_return)

        # test calculating short in profit
        # with an order thats flat
        _position = TestExchangeUtils.return_test_position([[1.0, OrderTypeEnum.SHORT, 100.0],
                                                            [1.0, OrderTypeEnum.SHORT, 90.0]], 90)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=90)
        self.assertEqual(1.0934099, positional_return)

        # test calculating short in profit
        _position = TestExchangeUtils.return_test_position([[1.0, OrderTypeEnum.SHORT, 100.0],
                                                            [1.0, OrderTypeEnum.SHORT, 90.0]], 110)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=110)
        self.assertAlmostEqual(0.6958063, positional_return)

        # having a long position that then begins to take profit
        _position = TestExchangeUtils.return_test_position([[1.0, OrderTypeEnum.LONG, 100.0],
                                                            [0.5, OrderTypeEnum.SHORT, 110.0],
                                                            [2.0, OrderTypeEnum.FLAT, 120.0]], 120)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=120)
        self.assertAlmostEqual(1.1385921272727273, positional_return)
        # 1.0967
        # 1.045318181818182




