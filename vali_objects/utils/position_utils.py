from typing import List

from vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderTypeEnum
from vali_objects.position import Position


class PositionUtils:
	@staticmethod
	def get_return_per_closed_position(positions: List[Position]) -> List[float]:
		closed_position_returns = [position.return_at_close
		                           for position in positions
		                           if position.is_closed_position]
		cumulative_return = 1
		per_position_return = []

		# calculate the return over time at each position close
		for value in closed_position_returns:
			cumulative_return *= value
			per_position_return.append(cumulative_return)
		return per_position_return
#
#     @staticmethod
#     def is_closed_position(position: Position) -> bool:
#         orders = position.orders
#
#         # first check to see if the sum of longs and shorts is the same
#         long_sum = sum([order.leverage
#                         for order in orders
#                         if order.order_type == OrderTypeEnum.LONG])
#         short_sum = sum([order.leverage
#                          for order in orders
#                          if order.order_type == OrderTypeEnum.SHORT])
#         if long_sum == short_sum:
#             return True
#
#         # get the last order and see if it's FLAT
#         if orders[len(orders) - 1].order_type == OrderTypeEnum.FLAT:
#             return True
#
#         return False
#
#     @staticmethod
#     def calculate_position_return(position: Position, closing_price: float) -> float:
#         net_leverage = 0
#         average_entry_price = 0
#
#         for order in position.orders:
#             leverage = order.leverage
#             if order == OrderTypeEnum.SHORT:
#                 leverage = 0 - order.leverage
#             elif order == OrderTypeEnum.FLAT:
#                 leverage = 0
#
#             total_leverage = net_leverage + leverage
#             if total_leverage != 0:
#                 average_entry_price = (average_entry_price * net_leverage + order.price * leverage) / total_leverage
#             else:
#                 average_entry_price = 0  # Reset if the position is flat
#
#             # Update the net leverage
#             net_leverage = total_leverage
#
#         positional_return = (((position.close_price - average_entry_price) * net_leverage) /
#                              * (1 - ValiConfig.TRADE_PAIR_FEES[position.trade_pair]))
#
#
#
#     @staticmethod
#     def calculate_position_return(position: Position, closing_price: float) -> float:
#         def _order_return(_order: Order, _closing_price: float) -> float:
#             return ((_closing_price - _order.price) / _order.price) * _order.leverage
#
#         def _add_fee(_order_return: float):
#             return _order_return * (1 - ValiConfig.TRADE_PAIR_FEES[position.trade_pair])
#
#         positional_return = 1.0
#         current_position_type = ""
#         positional_leverage = 0.0
#
#         current_positional_return = 1.0
#
#         for order in position.orders:
#             # initial setting of position type and leverage
#             if current_position_type == "":
#                 current_position_type = order.order_type
#
#             if current_position_type == OrderTypeEnum.LONG:
#                 positional_leverage += order.leverage
#             elif current_position_type == OrderTypeEnum.SHORT:
#                 positional_leverage -= order.leverage
#
#             if order.order_type == OrderTypeEnum.LONG:
#                 order_return = _add_fee((1 + _order_return(order, closing_price)))
#             elif order.order_type == OrderTypeEnum.SHORT:
#                 order_return = _add_fee(1 - _order_return(order, closing_price))
#             else:
#                 order_return = 1
#             positional_return = positional_return * order_return
#
#         return positional_return
