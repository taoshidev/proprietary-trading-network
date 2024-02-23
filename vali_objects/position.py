from typing import Optional, List

from vali_config import ValiConfig, TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderTypeEnum


class Position:
    _FLAT_ACCOUNT = 0.00000001

    def __init__(self,
                 miner_hotkey: str,
                 position_uuid: str,
                 open_ms: int,
                 trade_pair: TradePair,
                 orders: List[Order] = None,
                 current_return: Optional[float] = 0,
                 max_drawdown: Optional[float] = 0,
                 close_ms: Optional[int] = None,
                 return_at_close: Optional[float] = None,
                 close_price: Optional[float] = None):

        if orders is None:
            orders = []

        self.miner_hotkey = miner_hotkey
        self.position_uuid = position_uuid
        self.open_ms = open_ms
        self.trade_pair = trade_pair
        self.orders = orders
        self.current_return = current_return
        self.max_drawdown = max_drawdown
        self.close_ms = close_ms
        self.return_at_close = return_at_close
        self.close_price = close_price

        self._net_leverage = 0  # Positive for net long, negative for net short
        self._average_entry_price = 0
        self._initial_entry_price = 0

        self.position_type = None
        self.is_closed_position = False

    def set_current_return(self, current_price):
        self.current_return = self.calculate_unrealized_pnl(current_price)

    def add_order(self, order: Order):
        if (self.position_type is not None
                and order.order_type != self.position_type):
            raise ValueError(f"order type [{order.order_type}] "
                             f"does not match position type [{self.position_type}]")
        self.orders.append(order)

    def update_position(self):
        for order in self.orders:
            if self.position_type is not OrderTypeEnum.FLAT:
                # set the position type
                if self.position_type is None:
                    if order.leverage > 0:
                        self.position_type = OrderTypeEnum.LONG
                    elif order.leverage < 0:
                        self.position_type = OrderTypeEnum.SHORT
                    else:
                        raise ValueError("leverage of 0 provided as initial order.")

                # add logic to set leverage reset to 0 leverage and close position
                # if the position switches side
                # check position status
                if (self.position_type == OrderTypeEnum.LONG
                        and self._net_leverage + order.leverage <= 0):
                    adjusted_leverage = 0 - self._net_leverage + self._FLAT_ACCOUNT
                elif (self.position_type == OrderTypeEnum.SHORT
                        and self._net_leverage + order.leverage >= 0):
                    adjusted_leverage = abs(self._net_leverage) - self._FLAT_ACCOUNT
                elif (order.order_type == OrderTypeEnum.FLAT
                      and self.position_type == OrderTypeEnum.SHORT):
                    adjusted_leverage = self._FLAT_ACCOUNT
                elif (order.order_type == OrderTypeEnum.FLAT
                      and self.position_type == OrderTypeEnum.LONG):
                    adjusted_leverage = 0 - self._FLAT_ACCOUNT
                else:
                    adjusted_leverage = order.leverage

                # Calculate the new average entry price based on the leverage
                total_leverage = self._net_leverage + adjusted_leverage

                if total_leverage == self._FLAT_ACCOUNT:
                    self.position_type = OrderTypeEnum.FLAT

                self._average_entry_price = ((self._average_entry_price *
                                             self._net_leverage + order.price * adjusted_leverage)
                                             / total_leverage)

                # Update the net leverage
                self._net_leverage = total_leverage

                if self._initial_entry_price == 0:
                    self._initial_entry_price = order.price

            if (order.order_type is OrderTypeEnum.FLAT
                    or self.position_type == OrderTypeEnum.FLAT):
                self.is_closed_position = True
                self.close_price = order.price
                self.close_ms = order.processed_ms

                curr_return = self.calculate_unrealized_pnl(order.price)
                self.current_return = curr_return
                self.return_at_close = curr_return
                self.set_return_at_close_with_fees()

    def set_return_at_close_with_fees(self):
        self.return_at_close = (self.return_at_close *
                                (1 - ValiConfig.TRADE_PAIR_FEES[position.trade_pair]))

    def calculate_unrealized_pnl(self, current_price):
        # Calculate the unrealized profit or loss based on the current price
        return (1 + (current_price - self._average_entry_price)
                * self._net_leverage / self._initial_entry_price)



if __name__ == "__main__":
    # Example usage:
    position = Position(
        miner_hotkey="test",
        position_uuid="test",
        open_ms=123,
        trade_pair=TradePair.BTCUSD,
    )

    position.add_order(Order(order_type=OrderTypeEnum.LONG,
                             leverage=1.0,
                             price=80,
                             trade_pair=TradePair.BTCUSD,
                             processed_ms=123,
                             order_uuid="123"))
    position.add_order(Order(order_type=OrderTypeEnum.SHORT,
                             leverage=-0.99999,
                             price=90,
                             trade_pair=TradePair.BTCUSD,
                             processed_ms=123,
                             order_uuid="123"))
    position.add_order(Order(order_type=OrderTypeEnum.LONG,
                             leverage=1.0,
                             price=90,
                             trade_pair=TradePair.BTCUSD,
                             processed_ms=123,
                             order_uuid="123"))
    position.add_order(Order(order_type=OrderTypeEnum.SHORT,
                             leverage=-0.99999,
                             price=80,
                             trade_pair=TradePair.BTCUSD,
                             processed_ms=123,
                             order_uuid="123"))
    position.add_order(Order(order_type=OrderTypeEnum.FLAT,
                             leverage=0,
                             price=80,
                             trade_pair=TradePair.BTCUSD,
                             processed_ms=123,
                             order_uuid="123"))
    position.add_order(Order(order_type=OrderTypeEnum.LONG,
                             leverage=0,
                             price=80,
                             trade_pair=TradePair.BTCUSD,
                             processed_ms=123,
                             order_uuid="123"))
    position.add_order(Order(order_type=OrderTypeEnum.LONG,
                             leverage=0,
                             price=90,
                             trade_pair=TradePair.BTCUSD,
                             processed_ms=123,
                             order_uuid="123"))

    position.update_position()
    position.set_current_return(current_price=90)
    print(position.position_type)
    print(position.current_return)
    print(position.return_at_close)
