import json
import logging
from copy import deepcopy
from typing import Optional, List
from pydantic import model_validator, BaseModel, Field

from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType

import bittensor as bt


class Position(BaseModel):
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

    miner_hotkey: str
    position_uuid: str
    open_ms: int
    trade_pair: TradePair
    orders: List[Order] = Field(default_factory=list)
    current_return: float = 1.0
    close_ms: Optional[int] = None
    return_at_close: float = 1.0
    net_leverage: float = 0.0
    average_entry_price: float = 0.0
    initial_entry_price: float = 0.0
    position_type: Optional[OrderType] = None
    is_closed_position: bool = False

    @model_validator(mode="before")
    @classmethod
    def add_trade_pair_to_orders(cls, values):
        if isinstance(values['trade_pair'], TradePair):
            trade_pair_id = values['trade_pair'].trade_pair_id
        else:
            trade_pair_id = values['trade_pair'][0]
        trade_pair = TradePair.get_latest_trade_pair_from_trade_pair_id(trade_pair_id)
        orders = values.get('orders', [])

        # Add the position-level trade_pair to each order
        updated_orders = []
        for order in orders:
            if not isinstance(order, Order):
                order['trade_pair'] = trade_pair
                
            updated_orders.append(order)
        values['orders'] = updated_orders
        values['trade_pair'] = trade_pair
        return values

    def __hash__(self):
        # Include specified fields in the hash, assuming trade_pair is accessible and immutable
        return hash((self.miner_hotkey, self.position_uuid, self.open_ms, self.current_return,
                     self.net_leverage, self.initial_entry_price, self.trade_pair.trade_pair))

    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.miner_hotkey == other.miner_hotkey and
                self.position_uuid == other.position_uuid and
                self.open_ms == other.open_ms and
                self.current_return == other.current_return and
                self.net_leverage == other.net_leverage and
                self.initial_entry_price == other.initial_entry_price and
                self.trade_pair.trade_pair == other.trade_pair.trade_pair)

    def _handle_trade_pair_encoding(self, d):
        # Remove trade_pair from orders
        if 'orders' in d:
            for order in d['orders']:
                if 'trade_pair' in order:
                    del order['trade_pair']
        # Write the trade_pair in the legacy tuple format as to not break generate_request_outputs. This is temporary
        # code until generate_request_outputs is updated to have the new TradePair decoding logic. If BTC or ETH, put
        # the legacy fee value so that pydantic can validate the JSON with the original decoding logic
        if isinstance(d['trade_pair'], TradePair):
            tp = d['trade_pair']
            fee = .003 if tp.is_crypto else tp.fees
            d['trade_pair'] = [tp.trade_pair_id, tp.trade_pair, fee, tp.min_leverage, tp.max_leverage]
        else:
            d['trade_pair'] = d['trade_pair'][:5]
            if d['trade_pair'][0] in (TradePair.BTCUSD.trade_pair_id, TradePair.ETHUSD.trade_pair_id):
                d['trade_pair'][2] = 0.003
        return d

    def to_dict(self):
        d = deepcopy(self.dict())
        return self._handle_trade_pair_encoding(d)

    @property
    def is_open_position(self):
        return not self.is_closed_position

    @property
    def newest_order_age_ms(self):
        return TimeUtil.now_in_millis() - self.orders[-1].processed_ms

    def __str__(self):
        return self.to_json_string()

    def to_json_string(self) -> str:
        # Using pydantic's json method with built-in validation
        json_str = self.json()
        # Unfortunately, we can't tell pydantic v1 to strip certain fields so we do that here
        json_loaded = json.loads(json_str)
        json_compressed = self._handle_trade_pair_encoding(json_loaded)
        return json.dumps(json_compressed)

    @classmethod
    def from_dict(cls, position_dict):
        # Assuming 'orders' and 'trade_pair' need to be parsed from dict representations
        # Adjust as necessary based on the actual structure and types of Order and TradePair
        if 'orders' in position_dict:
            position_dict['orders'] = [Order.parse_obj(order) for order in position_dict['orders']]
        if 'trade_pair' in position_dict and isinstance(position_dict['trade_pair'], dict):
            # This line assumes TradePair can be initialized directly from a dict or has a similar parsing method
            position_dict['trade_pair'] = TradePair.from_trade_pair_id(position_dict['trade_pair']['trade_pair_id'])

        # Convert is_closed_position to bool if necessary
        # (assuming this conversion logic is no longer needed if input is properly formatted for Pydantic)

        return cls(**position_dict)

    @staticmethod
    def _position_log(message):
        bt.logging.trace("Position Notification - " + message)

    def get_net_leverage(self):
        return self.net_leverage

    def rebuild_position_with_updated_orders(self):
        self.current_return = 1.0
        self.close_ms = None
        self.return_at_close = 1.0
        self.net_leverage = 0.0
        self.average_entry_price = 0.0
        self.initial_entry_price = 0.0
        self.position_type = None
        self.is_closed_position = False
        self.position_type = None

        self._update_position()

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

        bt.logging.trace(
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

    def _leverage_flipped(self, prev_leverage, cur_leverage):
        return prev_leverage * cur_leverage < 0
    def max_leverage_seen(self):
        max_leverage = 0
        current_leverage = 0
        for order in self.orders:
            # Explicit flat
            if order.order_type == OrderType.FLAT:
                break
            prev_leverage = current_leverage
            current_leverage += order.leverage
            if current_leverage > self.trade_pair.max_leverage:
                current_leverage = self.trade_pair.max_leverage
            elif current_leverage < -self.trade_pair.max_leverage:
                current_leverage = -self.trade_pair.max_leverage
            # Implicit FLAT
            if current_leverage == 0 or self._leverage_flipped(prev_leverage, current_leverage):
                break

            if abs(current_leverage) > max_leverage:
                max_leverage = abs(current_leverage)
        return max_leverage

    def cumulative_leverage(self):
        current_leverage = 0.0
        cumulative_leverage = 0.0
        for order in self.orders:
            # Explicit flat
            if order.order_type == OrderType.FLAT:
                cumulative_leverage += abs(current_leverage)
                break

            prev_leverage = current_leverage

            # Clamp
            if current_leverage + order.leverage > self.trade_pair.max_leverage:
                current_leverage = self.trade_pair.max_leverage
            elif current_leverage + order.leverage < -self.trade_pair.max_leverage:
                current_leverage = -self.trade_pair.max_leverage
            else:
                current_leverage += order.leverage

            # Implicit FLAT
            if current_leverage == 0.0 or self._leverage_flipped(prev_leverage, current_leverage):
                cumulative_leverage += abs(prev_leverage)
                break
            else:
                cumulative_leverage += abs(current_leverage - prev_leverage)

        return cumulative_leverage

    def _handle_liquidation(self, time_ms):
        self._position_log("position liquidated. Trade pair: " + str(self.trade_pair.trade_pair_id))

        self.close_out_position(time_ms)

    def calculate_return_with_fees(self, current_return_no_fees, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()
        # Note: Closed positions will have static returns. This method is only called for open positions.
        # V2 fee calculation. Crypto fee lowered from .003 to .002. Multiply fee by leverage for crypto pairs.
        # V3 calculation. All fees scaled by leverage. Updated forex and indices fees.
        # V4 calculation. Fees are now based on cumulative leverage
        # V5 Crypto fees cut in half
        if timestamp_ms < 1713198680000:  # V4 PR merged
            fee = self.trade_pair.fees * self.max_leverage_seen()
        else:
            fee = self.trade_pair.fees * self.cumulative_leverage() / 2.0
        return current_return_no_fees * (1.0 - fee)

    def get_open_position_return_with_fees(self, realtime_price, time_ms):
        current_return = self.calculate_unrealized_pnl(realtime_price)
        return self.calculate_return_with_fees(current_return, timestamp_ms=time_ms)

    def set_returns(self, realtime_price, time_ms=None):
        if time_ms is None:
            time_ms = TimeUtil.now_in_millis()
        # We used to multiple trade_pair.fees by net_leverage. Eventually we will
        # Update this calculation to approximate actual exchange fees.
        self.current_return = self.calculate_unrealized_pnl(realtime_price)
        self.return_at_close = self.calculate_return_with_fees(self.current_return, timestamp_ms=time_ms)

        if self.current_return < 0:
            raise ValueError(f"current return must be positive {self.current_return}")

        if self.current_return == 0:
            self._handle_liquidation(time_ms)

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

        self.set_returns(realtime_price, time_ms=order.processed_ms)

        # Liquidated
        if self.current_return == 0:
            return
        self._position_log(f"closed position total w/o fees [{self.current_return}]. Trade pair: {self.trade_pair.trade_pair_id}")
        self._position_log(f"closed return with fees [{self.return_at_close}]. Trade pair: {self.trade_pair.trade_pair_id}")

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
            self._position_log("setting new position type as LONG. Trade pair: " + str(self.trade_pair.trade_pair_id))
            self.position_type = OrderType.LONG
        elif order.leverage < 0:
            self._position_log("setting new position type as SHORT. Trade pair: " + str(self.trade_pair.trade_pair_id))
            self.position_type = OrderType.SHORT
        else:
            raise ValueError("leverage of 0 provided as initial order.")

    def close_out_position(self, close_ms):
        self.position_type = OrderType.FLAT
        self.is_closed_position = True
        self.close_ms = close_ms

    def reopen_position(self):
        self.position_type = self.orders[0].order_type
        self.is_closed_position = False
        self.close_ms = None

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
        bt.logging.trace(f"Updating position {self.trade_pair.trade_pair_id} with n orders: {len(self.orders)}")
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
