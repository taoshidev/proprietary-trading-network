import unittest

from vali_objects.dataclasses.order import Order
from vali_objects.dataclasses.position import Position
from vali_objects.enums.order_type_enum import OrderTypeEnum
from vali_objects.utils.exchange_utils import ExchangeUtils


class TestExchangeUtils(unittest.TestCase):

    @staticmethod
    def return_test_position(order_price1,
                             order_price2,
                             order_type1,
                             order_type2,
                             closing_price):
        return Position(
            miner_hotkey="123",
            open_ms=123,
            orders=[Order(
                leverage=1.0,
                order_type=order_type1,
                order_uuid="123",
                price=order_price1,
                processed_ms=123,
                trade_pair="BTC/USD"
                ),
                Order(
                    leverage=1.0,
                    order_type=order_type2,
                    order_uuid="123",
                    price=order_price2,
                    processed_ms=123,
                    trade_pair="BTC/USD"
                ),
            ],
            position_uuid="test",
            trade_pair="BTC/USD",
            close_price=closing_price
        )

    def test_calculate_position_return_long_gain(self):
        # test calculating long only in profit
        # with an order that was flat in return
        _position = TestExchangeUtils.return_test_position(100.0,
                                                           110.0,
                                                           OrderTypeEnum.LONG,
                                                           OrderTypeEnum.LONG,
                                                           110)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=110)
        self.assertEqual(1.0934099, positional_return)

        # test calculating long only at a loss
        _position = TestExchangeUtils.return_test_position(100.0,
                                                           110.0,
                                                           OrderTypeEnum.LONG,
                                                           OrderTypeEnum.LONG,
                                                           90)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=90)
        self.assertEqual(0.7319520818181818, positional_return)

        # test calculating short in profit
        # with an order thats flat
        _position = TestExchangeUtils.return_test_position(100.0,
                                                           90.0,
                                                           OrderTypeEnum.SHORT,
                                                           OrderTypeEnum.SHORT,
                                                           90)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=90)
        self.assertEqual(1.0934099, positional_return)

        # test calculating short in profit
        # with an order thats flat
        # 0.909090909090909

        _position = TestExchangeUtils.return_test_position(100.0,
                                                           90.0,
                                                           OrderTypeEnum.SHORT,
                                                           OrderTypeEnum.SHORT,
                                                           110)
        positional_return = ExchangeUtils.calculate_position_return(position=_position,
                                                closing_price=110)
        self.assertEqual(0.6958062999999999, positional_return)




