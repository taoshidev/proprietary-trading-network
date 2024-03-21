import logging
from typing import Optional, List

from vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType

import bittensor as bt


class Position:
    """Represents a position in a trading system.

    As a miner, you need to send in signals to the validators, who will keep track
    of your closed and open positions based on your signals. Miners are judged based
    on a 30-day rolling window of return with time decay, so they must continuously perform.

    A signal contains the following information:
    - Trade Pair: The trade pair you want to trade (e.g., major indexes, forex, BTC, ETH).
    - Order Type: SHORT, LONG, or FLAT.
    - Leverage: The amount of leverage for the order type.

    On the validator's side, signals are converted into orders. The validator specifies
    the price at which they fulfilled a signal, which is then used for the order.
    Positions are composed of orders.

    Rules:
    - Please refer to README.md for the rules of the trading system.
    """

    def __init__(
        self,
        miner_hotkey: str,
        # hotkeys are used to sign for a coldkey. They're what registers with subnets and how we identify miners.
        position_uuid: str,
        open_ms: int,
        trade_pair: TradePair,
        orders: List[Order] = None,
        current_return: Optional[float] = 1,
        max_drawdown: Optional[float] = 0,
        close_ms: Optional[int] = None,
        return_at_close: Optional[float] = 1,
        net_leverage: Optional[float] = 0,
        average_entry_price: Optional[float] = 0,
        initial_entry_price: Optional[float] = 0,
        position_type: Optional[OrderType] = None,
        is_closed_position: Optional[bool] = False
    ):
        if orders is None:
            orders = []

        self.miner_hotkey = miner_hotkey
        self.position_uuid = position_uuid
        self.open_ms = open_ms
        self.trade_pair = trade_pair
        self.orders = orders
        self.current_return = current_return
        self.close_ms = close_ms
        self.return_at_close = return_at_close

        self.net_leverage = net_leverage
        self.average_entry_price = average_entry_price
        self.initial_entry_price = initial_entry_price

        self.position_type = position_type
        self.is_closed_position = is_closed_position

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self):
        return {
                # args
                "miner_hotkey": self.miner_hotkey,
                "position_uuid": self.position_uuid,
                "open_ms": self.open_ms,
                "trade_pair": str(self.trade_pair.trade_pair_id),
                "orders": [str(order) for order in self.orders],
                "current_return": self.current_return,
                "close_ms": self.close_ms,
                # additional important data
                "return_at_close": self.return_at_close,
                "net_leverage": self.net_leverage,
                "average_entry_price": self.average_entry_price,
                "initial_entry_price": self.initial_entry_price,
                "position_type": str(self.position_type),
                "is_closed_position": str(self.is_closed_position),
            }

    @staticmethod
    def from_dict(position_dict):
        orders = [Order.from_dict(order) for order in position_dict["orders"]]
        position_dict["orders"] = orders
        position_dict["trade_pair"] = TradePair.from_trade_pair_id(
            position_dict["trade_pair"]["trade_pair_id"]
        )
        position_dict["is_closed_position"] = (
            True if position_dict["is_closed_position"].lower() == "true" else False
        )
        return Position(**position_dict)

    @staticmethod
    def _position_log(message):
        bt.logging.info("Position Notification - " + message)

    def get_net_leverage(self):
        return self.net_leverage

    def log_position_status(self):
        bt.logging.debug(
            f"position details: "
            f"close_ms [{self.close_ms}] "
            f"initial entry price [{self.initial_entry_price}] "
            f"net leverage [{self.net_leverage}] "
            f"average entry price [{self.average_entry_price}] "
            f"return_at_close [{self.return_at_close}]"
        )
        order_info = [
            {
                "order type": order.order_type.value,
                "leverage": order.leverage,
                "price": order,
            }
            for order in self.orders
        ]
        bt.logging.debug(f"position order details: " f"close_ms [{order_info}] ")

    def add_order(self, order: Order):
        if self.is_closed_position:
            logging.warning(
                "Miner attempted to add order to a closed/liquidated position. Ignoring."
            )
            return
        if order.trade_pair != self.trade_pair:
            raise ValueError(
                f"Order trade pair [{order.trade_pair}] does not match position trade pair [{self.trade_pair}]"
            )

        if self._clamp_leverage(order):
            if order.leverage == 0:
                # This order's leverage got clamped to zero.
                # Skip it since we don't want to consider this a FLAT position and we don't want to allow bad actors
                # to send in a bunch of spam orders.
                logging.warning(
                    f"Miner attempted to add exceed max leverage for trade pair {self.trade_pair.trade_pair_id}. "
                    f"Clamping to max leverage {self.trade_pair.max_leverage}"
                )
                return
        self.orders.append(order)
        self._update_position()

    def calculate_unrealized_pnl(self, current_price):
        if self.initial_entry_price == 0 or self.average_entry_price is None:
            return 1

        bt.logging.info(
            f"trade_pair: {self.trade_pair.trade_pair_id} current price: {current_price},"
            f" average entry price: {self.average_entry_price}, net leverage: {self.net_leverage}, "
            f"initial entry price: {self.initial_entry_price}"
        )
        gain = (
            (current_price - self.average_entry_price)
            * self.net_leverage
            / self.initial_entry_price
        )
        # Check if liquidated
        if gain <= -1.0:
            return 0
        net_return = 1 + gain
        return net_return

    def _handle_liquidation(self, order):
        self._position_log("position liquidated")
        self.close_out_position(order.processed_ms)

    def set_returns(self, realtime_price, net_leverage):
        self.current_return = self.calculate_unrealized_pnl(realtime_price)
        self.return_at_close = self.current_return * (
            1 - self.trade_pair.fees * abs(net_leverage)
        )

    def update_position_state_for_new_order(self, order, delta_leverage):
        """
        Must be called after every order to maintain accurate internal state. The variable average_entry_price has
        a name that can be a little confusing. Although it claims to be the average price, it really isn't.
        For example, it can take a negative value. A more accurate name for this variable is the weighted average
        entry price.
        """
        realtime_price = order.price
        assert self.initial_entry_price > 0, self.initial_entry_price
        new_net_leverage = self.net_leverage + delta_leverage

        self.set_returns(realtime_price, new_net_leverage)

        if self.current_return < 0:
            raise ValueError(f"current return must be positive {self.current_return}")

        if self.current_return == 0:
            self._handle_liquidation(order)
            return
        self._position_log(f"closed position total w/o fees [{self.current_return}]")
        self._position_log(f"closed return with fees [{self.return_at_close}]")

        if self.position_type == OrderType.FLAT:
            self.net_leverage = 0.0
        else:
            self.average_entry_price = (
                self.average_entry_price * self.net_leverage
                + realtime_price * delta_leverage
            ) / new_net_leverage
            self.net_leverage = new_net_leverage

    def initialize_position_from_first_order(self, order):
        self.initial_entry_price = order.price
        if self.initial_entry_price <= 0:
            raise ValueError("Initial entry price must be > 0")
        # Initialize the position type. It will stay the same until the position is closed.
        if order.leverage > 0:
            self._position_log("setting new position type as LONG")
            self.position_type = OrderType.LONG
        elif order.leverage < 0:
            self._position_log("setting new position type as SHORT")
            self.position_type = OrderType.SHORT
        else:
            raise ValueError("leverage of 0 provided as initial order.")

    def close_out_position(self, close_ms):
        self.position_type = OrderType.FLAT
        self.is_closed_position = True
        self.close_ms = close_ms

    def _clamp_leverage(self, order):
        proposed_leverage = self.net_leverage + order.leverage
        if self.position_type == OrderType.LONG and proposed_leverage > self.trade_pair.max_leverage:
            order.leverage = self.trade_pair.max_leverage - self.net_leverage
            return True
        elif self.position_type == OrderType.SHORT and proposed_leverage < -self.trade_pair.max_leverage:
            order.leverage = -self.trade_pair.max_leverage - self.net_leverage
            return True

        return False

    def _update_position(self):
        self.net_leverage = 0.0
        bt.logging.info(f"Updating position with n orders: {len(self.orders)}")
        for order in self.orders:
            if self.position_type is None:
                self.initialize_position_from_first_order(order)

            # Check if the new order flattens the position, explicitly or implicitly
            if (
                (
                    self.position_type == OrderType.LONG
                    and self.net_leverage + order.leverage <= 0
                )
                or (
                    self.position_type == OrderType.SHORT
                    and self.net_leverage + order.leverage >= 0
                )
                or order.order_type == OrderType.FLAT
            ):
                #self._position_log(
                #    f"Flattening {self.position_type.value} position from order {order}"
                #)
                self.close_out_position(order.processed_ms)

            # Reflect the current order in the current position's return.
            adjusted_leverage = (
                0.0 if self.position_type == OrderType.FLAT else order.leverage
            )
            #bt.logging.info(
            #    f"Updating position state for new order {order} with adjusted leverage {adjusted_leverage}"
            #)
            self.update_position_state_for_new_order(order, adjusted_leverage)

            # If the position is already closed, we don't need to process any more orders. break in case there are more orders.
            if self.position_type == OrderType.FLAT:
                break
