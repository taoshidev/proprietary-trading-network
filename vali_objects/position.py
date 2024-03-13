import logging
from typing import Optional, List

from vali_config import ValiConfig, TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType

import bittensor as bt


class Position:
    """Represents a position in a trading system.

    As a miner, you need to send in signals to the validators, who will keep track
    of your closed and open positions based on your signals. Miners are judged based
    on a 30-day rolling window of return, so they must continuously perform.

    A signal contains the following information:
    - Trade Pair: The trade pair you want to trade (e.g., major indexes, forex, BTC, ETH).
    - Order Type: SHORT, LONG, or FLAT.
    - Leverage: The amount of leverage for the order type.

    On the validator's side, signals are converted into orders. The validator specifies
    the price at which they fulfilled a signal, which is then used for the order.
    Positions are composed of orders.

    Rules:
    - Returns are calculated as a multiplier of the original portfolio value which for simplicity can be considered 1.
    - LONG signal's leverage should be positive.
    - SHORT signal's leverage should be negative.
    - You can only open 1 position per trade pair at a time.
    - Positions are uni-directional. If a position starts LONG (the first order it receives
      is LONG), it can't flip to SHORT. If you try to flip it to SHORT (using more leverage
      SHORT than exists LONG), it will close out the position. You'll then need to open a
      second position which is SHORT with the difference.
    - You can take profit on an open position using LONG and SHORT. For example, if you have
      an open LONG position with 0.75x leverage and you want to start taking profit, you
      would send in SHORT signals to reduce the size of the position. Ex: Sending a short at 
      -.25 leverage. This functions very similarly to dYdX.
    - You can close out a position by sending in a FLAT signal.
    - Max drawdown is determined every minute. If you go beyond 5% max drawdown on daily
      close, or 10% at any point in time, you're eliminated. Eliminated miners won't
      necessarily be immediately eliminated; they'll need to wait to be deregistered based
      on the immunity period.
    - If a miner copies another miner's order repeatedly, they will be eliminated. There is
      core logic to catch and remove miners who provide signals that are similar to another
      miner.
    - There is a fee per trade pair: Crypto has 0.3% per position, forex has 0.03%, and
      indexes have 0.05%.
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
    ):
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

        self._net_leverage = 0  # Positive for net long, negative for net short
        self._average_entry_price = 0
        self._initial_entry_price = 0

        self.position_type = None
        self.is_closed_position = False

    def __str__(self) -> str:
        return str({
            'position_uuid': self.position_uuid,
            'net_leverage': self._net_leverage,
            'average_entry_price': self._average_entry_price,
            'position_type': str(self.position_type),
            'return_at_close': self.return_at_close,
            'current_return': self.current_return,
            'orders': [str(order) for order in self.orders],
            'close_ms': self.close_ms
        })

    @staticmethod
    def _position_log(message):
        bt.logging.info("Position Notification - " + message)

    def get_net_leverage(self): return self._net_leverage

    def log_position_status(self):
        bt.logging.debug(
            f"position details: "
            f"close_ms [{self.close_ms}] "
            f"initial entry price [{self._initial_entry_price}] "
            f"net leverage [{self._net_leverage}] "
            f"average entry price [{self._average_entry_price}] "
            f"return_at_close [{self.return_at_close}]"
        )
        order_info = [
            {"order type": order.order_type.value, "leverage": order.leverage, "price": order}
            for order in self.orders
        ]
        bt.logging.debug(f"position order details: " f"close_ms [{order_info}] ")

    def add_order(self, order: Order):
        if self.is_closed_position:
            logging.warning("Miner attempted to add order to a closed/liquidated position. Ignoring.")
            return
        self.orders.append(order)
        self._update_position()

    def calculate_unrealized_pnl(self, current_price):
        if self._initial_entry_price == 0 or self._average_entry_price is None:
            return 1

        bt.logging.info(
            f"current price: {current_price}, average entry price: {self._average_entry_price}, net leverage: {self._net_leverage}, initial entry price: {self._initial_entry_price}")
        gain = (current_price - self._average_entry_price) * self._net_leverage / self._initial_entry_price
        # Check if liquidated
        if gain <= -1.0:
            return 0
        net_return = 1 + gain
        return net_return

    def _handle_liquidation(self, order):
        self._position_log("position liquidated")
        self._close_out_position(order)

    def set_returns(self, realtime_price, net_leverage):
        self.current_return = self.calculate_unrealized_pnl(realtime_price)
        self.return_at_close = self.current_return * (
                1 - ValiConfig.TRADE_PAIR_FEES[self.trade_pair] * abs(net_leverage)
        )

    def update_position_state_for_new_order(self, order, delta_leverage):
        """
        Must be called after every order to maintain accurate internal state. The variable _average_entry_price has
        a name that can be a little confusing. Although it claims to be the average price, it is really isn't. For example
        it can take a negative value. A more accurate name for this variable is the weighted average entry price.
        """
        realtime_price = order.price
        assert self._initial_entry_price > 0, self._initial_entry_price
        new_net_leverage = self._net_leverage + delta_leverage

        self.set_returns(realtime_price, new_net_leverage)

        if self.current_return < 0:
            raise ValueError(f"current return must be positive {self.current_return}")

        if self.current_return == 0:
            self._handle_liquidation(order)
            return
        self._position_log(f"closed position total w/o fees [{self.current_return}]")
        self._position_log(f"closed return with fees [{self.return_at_close}]")

        if self.position_type == OrderType.FLAT:
            self._net_leverage = 0.0
        else:
            self._average_entry_price = \
                ((self._average_entry_price * self._net_leverage + realtime_price * delta_leverage) /
                 new_net_leverage)
            self._net_leverage = new_net_leverage

    def initialize_position_from_first_order(self, order):
        self._initial_entry_price = order.price
        if self._initial_entry_price <= 0:
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

    def _close_out_position(self, order):
        self.position_type = OrderType.FLAT
        self.is_closed_position = True
        self.close_ms = order.processed_ms

    def _update_position(self):
        self._net_leverage = 0.0
        bt.logging.info(f"Updating position with n orders: {len(self.orders)}")
        for order in self.orders:
            if self.position_type is None:
                self.initialize_position_from_first_order(order)

            # Check if the new order flattens the position, explicitly or implicitly
            if ((self.position_type == OrderType.LONG and self._net_leverage + order.leverage <= 0) or
                    (self.position_type == OrderType.SHORT and self._net_leverage + order.leverage >= 0) or
                    order.order_type == OrderType.FLAT):
                self._position_log(f"Flattening {self.position_type.value} position from order {order}")
                self._close_out_position(order)

            # Reflect the current order in the current position's return. 
            adjusted_leverage = 0.0 if self.position_type == OrderType.FLAT else order.leverage
            bt.logging.info(f"Updating position state for new order {order} with adjusted leverage {adjusted_leverage}")
            self.update_position_state_for_new_order(order, adjusted_leverage)

            # If the position is already closed, we don't need to process any more orders. break in case there are more orders.
            if self.position_type == OrderType.FLAT:
                break
