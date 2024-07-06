import json
import logging
from copy import deepcopy
from typing import Optional, List
from pydantic import model_validator, BaseModel, Field, model_serializer, root_validator

from time_util.time_util import TimeUtil, MS_IN_8_HOURS
from vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_type_enum import OrderType

import bittensor as bt
import math

CRYPTO_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - 0.08) / (365.0*3.0))  # 8% per year for 1x leverage. Each interval is 8 hrs
FOREX_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - .0175) / 365.0)  # 1.75% per year for 1x leverage. Each interval is 24 hrs
INDICES_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - .03) / 365.0)  # 3% per year for 1x leverage. Each interval is 24 hrs


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
    position_type: Optional[OrderType] = None
    is_closed_position: bool = False

    #@model_serializer
    #def custom_serializer(self):
    #    # Manually construct the dictionary without excluded fields
    #    data = {field: value for field, value in self.__dict__.items() if field[0] != '_'}
    #    return data

    @model_validator(mode='before')
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
            else:
                order = order.copy(update={'trade_pair': trade_pair})

            updated_orders.append(order)
        values['orders'] = updated_orders
        values['trade_pair'] = trade_pair
        return values

    def get_cumulative_leverage(self) -> float:
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


    def get_spread_fee(self) -> float:
        return 1.0 - (self.get_cumulative_leverage() * self.trade_pair.fees * 0.5)

    def crypto_carry_fee(self, current_time_ms: int) -> (float, int):
        # Fees every 8 hrs. 4 UTC, 12 UTC, 20 UTC
        n_intervals_elapsed, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_crypto(self.open_ms, current_time_ms)
        fee_product = 1.0
        start_ms = self.open_ms
        end_ms = start_ms + time_until_next_interval_ms
        for n in range(n_intervals_elapsed):
            if n != 0:
                start_ms = end_ms
                end_ms = start_ms + MS_IN_8_HOURS
            max_lev = self.max_leverage_seen_in_interval(start_ms, end_ms)
            fee_product *= CRYPTO_CARRY_FEE_PER_INTERVAL ** max_lev

        final_fee = fee_product
        #ct_formatted = TimeUtil.millis_to_formatted_date_str(current_time_ms)
        #start_formatted = TimeUtil.millis_to_formatted_date_str(self.open_ms)
        #print(f"start time {start_formatted}, end time {ct_formatted}, delta (days) {(current_time_ms - self.open_ms) / (1000 * 60 * 60 * 24)} final fee {final_fee}")
        return final_fee, current_time_ms + time_until_next_interval_ms

    def forex_indices_carry_fee(self, current_time_ms: int) -> tuple[float, int]:
        # Fees M-F where W gets triple fee.
        n_intervals_elapsed, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_forex_indices(self.open_ms,
                                                                                               current_time_ms)
        fee_product = 1.0
        start_ms = self.open_ms
        end_ms = start_ms + time_until_next_interval_ms
        for n in range(n_intervals_elapsed):
            if n != 0:
                start_ms = end_ms
                end_ms = start_ms + MS_IN_8_HOURS
            # Monday == 0...Sunday == 6
            day_of_week_index = TimeUtil.get_day_of_week_from_timestamp(end_ms)
            assert day_of_week_index in range(7)
            if day_of_week_index in (5, 6):
                continue  # no fees on Saturday, Sunday
            else:
                fee = 1.0
                max_lev = self.max_leverage_seen_in_interval(start_ms, end_ms)
                if self.trade_pair.is_forex:
                    fee *= FOREX_CARRY_FEE_PER_INTERVAL ** max_lev
                elif self.trade_pair.is_indices:
                    fee *= INDICES_CARRY_FEE_PER_INTERVAL ** max_lev
                else:
                    raise ValueError(f"Unexpected trade pair: {self.trade_pair.trade_pair_id}")
                if day_of_week_index == 2:
                    fee = fee ** 3  # triple fee on Wednesday

            fee_product *= fee

        return fee_product, current_time_ms + time_until_next_interval_ms

    def get_carry_fee(self, current_time_ms) -> (float, int):
        # Calculate the number of times a new day occurred (UTC). If a position is opened at 23:59:58 and this function is
        # called at 00:00:02, the carry fee will be calculated as if a day has passed. Another example: if a position is
        # opened at 23:59:58 and this function is called at 23:59:59, the carry fee will be calculated as 0 days have passed
        # Recalculate and update cache
        assert current_time_ms
        if self.is_closed_position and current_time_ms > self.close_ms:
            current_time_ms = self.close_ms
        if self.trade_pair.is_crypto:
            carry_fee, next_update_time_ms = self.crypto_carry_fee(current_time_ms)
        elif self.trade_pair.is_forex or self.trade_pair.is_indices:
            carry_fee, next_update_time_ms = self.forex_indices_carry_fee(current_time_ms)
        else:
            raise Exception(f'Unexpected trade pair: {self.trade_pair.trade_pair_id}')

        return carry_fee, next_update_time_ms


    @property
    def initial_entry_price(self) -> float:
        if not self.orders or len(self.orders) == 0:
            return 0.0
        return self.orders[0].price

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
        return prev_leverage * cur_leverage < 0 or prev_leverage != 0 and cur_leverage == 0

    def max_leverage_seen_in_interval(self, start_ms: int, end_ms: int) -> float:
        """
        Returns the max leverage seen in the interval [start_ms, end_ms] (inclusive). If no orders are in the interval,
        raise an exception
        """
        # check valid bounds and throw ValueError if bad data
        if start_ms > end_ms:
            raise ValueError(f"start_ms [{start_ms}] is greater than end_ms [{end_ms}]")
        if end_ms < self.open_ms:
            raise ValueError(f"end_ms [{end_ms}] is less than open_ms [{self.open_ms}]")
        if end_ms < 0 or start_ms < 0:
            raise ValueError(f"start_ms [{start_ms}] or end_ms [{end_ms}] is less than 0")
        if len(self.orders) == 0:
            raise ValueError("No orders in position")
        if self.orders[0].processed_ms > end_ms:
            raise ValueError(f"First order processed_ms [{self.orders[0].processed_ms}] is greater than end_ms [{end_ms}]")
        if self.is_closed_position and start_ms > self.close_ms:
            raise ValueError(f"Position closed before interval start_ms [{start_ms}]")


        interval_data = {'start_ms': start_ms, 'end_ms': end_ms, 'max_leverage': -float('inf')}
        self.max_leverage_seen(interval_data=interval_data)

        if interval_data['max_leverage'] == -float('inf'):
            raise ValueError('Unable to find max leverage in interval')
        assert interval_data['max_leverage'] > 0, interval_data['max_leverage']
        return interval_data['max_leverage']

    def max_leverage_seen(self, interval_data=None):
        max_leverage = 0
        current_leverage = 0
        stop_signaled = False
        for idx, order in enumerate(self.orders):
            if stop_signaled:
                break

            prev_leverage = current_leverage
            # Explicit flat
            if order.order_type == OrderType.FLAT:
                stop_signaled = True
                current_leverage = 0
            else:
                current_leverage += order.leverage
                if current_leverage > self.trade_pair.max_leverage:
                    current_leverage = self.trade_pair.max_leverage
                elif current_leverage < -self.trade_pair.max_leverage:
                    current_leverage = -self.trade_pair.max_leverage
                # Implicit FLAT
                if self._leverage_flipped(prev_leverage, current_leverage):
                    stop_signaled = True
                    current_leverage = 0

            if abs(current_leverage) > max_leverage:
                max_leverage = abs(current_leverage)

            if interval_data:
                if order.processed_ms < interval_data['start_ms']:
                    pass
                elif order.processed_ms <= interval_data['end_ms']:
                    interval_data['max_leverage'] = max(abs(current_leverage), interval_data['max_leverage'])
                # An order passes the interval for the first time
                elif order.processed_ms > interval_data['end_ms']:
                    interval_data['max_leverage'] = max(abs(prev_leverage), interval_data['max_leverage'])
                    stop_signaled = True

        # The position's last order is way before the interval start. Use the last known position leverage
        if interval_data and interval_data['max_leverage'] == -float('inf'):
            interval_data['max_leverage'] = abs(current_leverage)
        return max_leverage

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
        # V6 introduce "carry fee"
        if timestamp_ms < 1713198680000:  # V4 PR merged
            fee = 1.0 - self.trade_pair.fees * self.max_leverage_seen()
        else: # TODO UPDATE TO TIME OF PR
            fee = self.get_spread_fee()
        #else:
        #    fee = self.get_carry_fee(timestamp_ms)[0] * self.get_spread_fee()
        return current_return_no_fees * fee

    def get_open_position_return_with_fees(self, realtime_price, time_ms):
        current_return = self.calculate_unrealized_pnl(realtime_price)
        return self.calculate_return_with_fees(current_return, timestamp_ms=time_ms)

    def set_returns(self, realtime_price, time_ms=None, total_fees=None):
        # We used to multiple trade_pair.fees by net_leverage. Eventually we will
        # Update this calculation to approximate actual exchange fees.
        self.current_return = self.calculate_unrealized_pnl(realtime_price)
        if total_fees is None:
            self.return_at_close = self.calculate_return_with_fees(self.current_return,
                               timestamp_ms=TimeUtil.now_in_millis() if time_ms is None else time_ms)
        else:
            self.return_at_close = self.current_return * total_fees

        if self.current_return < 0:
            raise ValueError(f"current return must be positive {self.current_return}")

        if self.current_return == 0:
            self._handle_liquidation(TimeUtil.now_in_millis() if time_ms is None else time_ms)

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
        self.open_ms = order.processed_ms
        if order.price <= 0:
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
