from vali_config import ValiConfig
from vali_objects.dataclasses.order import Order
from vali_objects.dataclasses.position import Position
from vali_objects.enums.order_type_enum import OrderTypeEnum


class ExchangeUtils:

    @staticmethod
    def is_closed_position(position: Position) -> bool:
        orders = position.orders

        # first check to see if the sum of longs and shorts is the same
        long_sum = sum([order.leverage
                        for order in orders
                        if order.order_type == OrderTypeEnum.LONG])
        short_sum = sum([order.leverage
                         for order in orders
                         if order.order_type == OrderTypeEnum.SHORT])
        if long_sum == short_sum:
            return True

        # get the last order and see if it's FLAT
        if orders[len(orders) - 1].order_type == OrderTypeEnum.FLAT:
            return True

        return False

    @staticmethod
    def calculate_position_return(position: Position, closing_price: float) -> float:
        def _order_return(_order: Order, _closing_price: float) -> float:
            return _closing_price / _order.price * _order.leverage

        def _add_fee(_order_return: float):
            return _order_return * (1 - ValiConfig.TRADE_PAIR_FEES[position.trade_pair])

        positional_return = 1.0
        for order in position.orders:
            if order.order_type == OrderTypeEnum.LONG:
                order_return = _add_fee((_order_return(order, closing_price)))
            elif order.order_type == OrderTypeEnum.SHORT:
                order_return = _add_fee(2 - _order_return(order, closing_price))
            else:
                order_return = 1
            positional_return = positional_return * order_return

        return positional_return
