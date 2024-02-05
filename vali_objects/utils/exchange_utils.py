from vali_objects.dataclasses.position import Position
from vali_objects.enums.order_type_enum import OrderTypeEnum


class ExchangeUtils:

    @staticmethod
    def is_closed_position(position: Position):
        orders = position.orders

        # first check to see if the sum of longs and shorts is the same
        long_sum = sum([order.leverage for order in orders if order.order_type == OrderTypeEnum.LONG])
        short_sum = sum([order.leverage for order in orders if order.order_type == OrderTypeEnum.SHORT])
        if long_sum == short_sum:
            return True

        # get the last order and see if its FLAT
        if orders[len(orders)-1].order_type == OrderTypeEnum.FLAT:
            return True

        return False


    @staticmethod
    def calculate_position_return(position: Position):

