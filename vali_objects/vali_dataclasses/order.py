# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

from typing import Optional as OptionalType
import uuid

from time_util.time_util import TimeUtil
from pydantic import model_validator

from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.vali_dataclasses.order_signal import OrderFields
from vali_objects.vali_dataclasses.price_source import PriceSource

# Execution types whose size fields are not sign-normalized against order_type
_UNSIZED_EXECUTION_TYPES = (ExecutionType.LIMIT_EDIT, ExecutionType.LIMIT_CANCEL, ExecutionType.FLAT_ALL)
# Execution types whose legacy top-level stop_loss/take_profit/trailing_stop are folded into bracket_orders
_BRACKETABLE_EXECUTION_TYPES = (ExecutionType.MARKET, ExecutionType.LIMIT, ExecutionType.STOP_LIMIT, ExecutionType.LIMIT_EDIT)


class Order(OrderFields):
    """
    A stored order record: a filled order, or a pending limit/stop-limit/bracket order.

    Request rules (size fields, price rules, trailing stop ranges) are validated on Signal at the
    API boundary, not here, because Orders are rebuilt from disk, checkpoints and RPC far more
    often than they are created. The two validators below only canonicalize legacy shapes and fill
    derived defaults; both are a few dict lookups for already-canonical data.
    """
    price: float                # Quote currency
    bid: float = 0              # Quote currency
    ask: float = 0              # Quote currency
    slippage: float = 0
    quote_usd_rate: float = 0.0 # Conversion rate from quote currency to USD
    usd_base_rate: float = 0.0  # Conversion rate from usd to base currency
    processed_ms: int
    order_uuid: str
    price_sources: list = []
    src: int = OrderSource.ORGANIC
    realized_pnl: float = 0
    margin_loan: float = 0.0
    # Liquidity flag. None=unknown (priced as taker), True=taker, False=maker.
    is_hl_taker: OptionalType[bool] = None

    @model_validator(mode='before')
    @classmethod
    def normalize(cls, values):
        """Canonicalize input shapes in a single pass (one Python call per order instead of one per field)."""
        if not isinstance(values, dict):
            return values

        # trade_pair: accept trade_pair_id, or a str / {'trade_pair_id': ...} / [id, name, ...] form
        if 'trade_pair_id' in values and 'trade_pair' not in values:
            values = {k: v for k, v in values.items() if k != 'trade_pair_id'} | {'trade_pair': values['trade_pair_id']}
        tp = values.get('trade_pair')
        if tp is not None and not isinstance(tp, TradePair):
            if isinstance(tp, str):
                values['trade_pair'] = TradePair.from_trade_pair_id(tp)
            elif isinstance(tp, dict):
                if 'trade_pair_id' in tp:
                    values['trade_pair'] = TradePair.from_trade_pair_id(tp['trade_pair_id'])
            elif isinstance(tp, list) and tp:
                values['trade_pair'] = TradePair.from_trade_pair_id(tp[0])

        # Enums arrive as their upper-case values from disk; tolerate legacy lower-case input
        execution_type = values.get('execution_type')
        if isinstance(execution_type, str) and not execution_type.isupper():
            values['execution_type'] = ExecutionType.from_string(execution_type)
        stop_condition = values.get('stop_condition')
        if isinstance(stop_condition, str) and not stop_condition.isupper():
            values['stop_condition'] = StopCondition.from_string(stop_condition)

        price = values.get('price')
        if price is not None and price < 0:
            raise ValueError("Price must be greater than 0")
        processed_ms = values.get('processed_ms')
        if processed_ms is not None and processed_ms < 0:
            raise ValueError("processed_ms must be greater than 0")

        if 'order_uuid' in values and not isinstance(values['order_uuid'], str):
            values['order_uuid'] = str(values['order_uuid'])

        price_sources = values.get('price_sources')
        if price_sources and isinstance(price_sources, list):
            values['price_sources'] = [PriceSource(**ps) if isinstance(ps, dict) else ps for ps in price_sources]

        # Size sign follows direction: SHORT sizes are negative, LONG sizes must not be
        order_type = values.get('order_type')
        if (order_type is OrderType.SHORT or order_type is OrderType.LONG) and execution_type not in _UNSIZED_EXECUTION_TYPES:
            for field in ('leverage', 'value', 'quantity'):
                size = values.get(field)
                if size is not None:
                    if order_type is OrderType.LONG and size < 0:
                        raise ValueError(f"{field} must be positive for LONG orders.")
                    elif order_type is OrderType.SHORT:
                        values[field] = -1.0 * abs(size)

        # Legacy top-level stop_loss/take_profit/trailing_stop become a bracket_orders entry
        bracket_orders = values.get('bracket_orders')
        bracketable = values.get('execution_type', ExecutionType.MARKET) in _BRACKETABLE_EXECUTION_TYPES
        if bracketable and not bracket_orders:
            stop_loss = values.get('stop_loss')
            take_profit = values.get('take_profit')
            trailing_stop = values.get('trailing_stop')
            if stop_loss is not None or take_profit is not None or trailing_stop is not None:
                bracket_entry = {'stop_loss': stop_loss, 'take_profit': take_profit, 'order_uuid': str(uuid.uuid4())}
                if trailing_stop is not None:
                    if 'trailing_percent' in trailing_stop:
                        bracket_entry['trailing_percent'] = trailing_stop['trailing_percent']
                    if 'trailing_value' in trailing_stop:
                        bracket_entry['trailing_value'] = trailing_stop['trailing_value']
                values['bracket_orders'] = [bracket_entry]
        elif bracket_orders and isinstance(bracket_orders, list):
            for sub in bracket_orders:
                if isinstance(sub, dict) and ('order_uuid' not in sub or (bracketable and not sub['order_uuid'])):
                    sub['order_uuid'] = str(uuid.uuid4())

        return values

    @model_validator(mode="after")
    def set_conversion_defaults(self):
        """
        Initializes quote_usd_rate and usd_base_rate based on the trade pair.
        Only sets values if they were left at the default 0.
        """
        if self.quote_usd_rate != 0 and self.usd_base_rate != 0:
            return self

        # Skip for FLAT_ALL or other cases where trade_pair is None
        if self.trade_pair is None:
            return self

        price = self.price
        if price == 0:
            return self

        base = self.trade_pair.base  # e.g. BTC in BTCUSDC
        quote = self.trade_pair.quote  # e.g. USDC in BTCUSDC

        if self.quote_usd_rate == 0:
            if "USD" in quote:
                self.quote_usd_rate = 1.0
            elif base == "USD":
                self.quote_usd_rate = 1.0 / price

        if self.usd_base_rate == 0:
            if base == "USD":
                self.usd_base_rate = 1.0
            elif "USD" in quote:
                self.usd_base_rate = 1.0 / price

        return self

    @classmethod
    def from_dict(cls, order_dict):
        """
        Create Order from dict. Pydantic validators handle all conversions:
        - trade_pair_id (str) -> trade_pair (TradePair)
        - order_type (str) -> order_type (OrderType)
        """
        return cls(**order_dict)

    def get_order_age(self, order):
        return TimeUtil.now_in_millis() - order.processed_ms

    def to_python_dict(self):
        trade_pair_id = None
        trade_pair_name = None
        if self.trade_pair is not None:
            trade_pair_id = self.trade_pair.trade_pair_id if hasattr(self.trade_pair, 'trade_pair_id') else 'unknown'
            trade_pair_name = self.trade_pair.trade_pair if hasattr(self.trade_pair, 'trade_pair') else 'unknown'

        return {
            'trade_pair_id': trade_pair_id,
            'trade_pair': [trade_pair_id, trade_pair_name],
            'order_type': self.order_type.name,
            'leverage': self.leverage,
            'value': self.value,
            'quantity': self.quantity,
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'slippage': self.slippage,
            'quote_usd_rate': self.quote_usd_rate,
            'usd_base_rate': self.usd_base_rate,
            'processed_ms': self.processed_ms,
            'price_sources': self.price_sources,
            'order_uuid': self.order_uuid,
            'src': self.src,
            'is_hl_taker': self.is_hl_taker,
            'execution_type': self.execution_type.name if self.execution_type else None,
            'realized_pnl': self.realized_pnl,
            'limit_price': self.limit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'margin_loan': self.margin_loan,
            'trailing_stop': self.trailing_stop,
            'bracket_orders': self.bracket_orders,
            'stop_price': self.stop_price,
            'stop_condition': str(self.stop_condition) if self.stop_condition else None,
            'bracket_pct': self.bracket_pct,
        }

    def to_dashboard(self, include_trade_pair: bool = False) -> dict:
        results = {
            "t": self.order_type.name,
            "v": self.value,
            "e": self.execution_type.name,
            "p": self.processed_ms,
        }

        if self.leverage is not None:
            results["l"] = self.leverage

        if self.quantity is not None:
            results["q"] = self.quantity

        if self.bracket_pct is not None:
            results["bpct"] = self.bracket_pct

        if self.price:
            results["pr"] = self.price

        if self.slippage:
            results["s"] = self.slippage

        if include_trade_pair and self.trade_pair is not None:
            results["tp"] = self.trade_pair.trade_pair

        if self.limit_price is not None:
            results["lp"] = self.limit_price

        if self.stop_loss is not None:
            results["sl"] = self.stop_loss

        if self.trailing_stop is not None:
            tsl_compact = {}
            if 'trailing_percent' in self.trailing_stop:
                tsl_compact["pct"] = self.trailing_stop["trailing_percent"]
            if 'trailing_value' in self.trailing_stop:
                tsl_compact["val"] = self.trailing_stop["trailing_value"]
            results["tsl"] = tsl_compact

        if self.take_profit is not None:
            results["tk"] = self.take_profit

        if self.bracket_orders is not None:
            unfilled_orders = {}
            for sub_order in self.bracket_orders:
                sub_bracket_orders = {}
                if sl := sub_order.get("stop_loss"):
                    sub_bracket_orders["sl"] = sl
                if tp := sub_order.get("take_profit"):
                    sub_bracket_orders["tp"] = tp
                if val := sub_order.get("value"):
                    sub_bracket_orders["v"] = val
                if qty := sub_order.get("quantity"):
                    sub_bracket_orders["q"] = qty
                if bpct := sub_order.get("bracket_pct"):
                    sub_bracket_orders["bpct"] = bpct

                tsl_compact = {}
                if 'trailing_percent' in sub_order:
                    tsl_compact["pct"] = sub_order["trailing_percent"]
                if 'trailing_value' in sub_order:
                    tsl_compact["val"] = sub_order["trailing_value"]

                if tsl_compact:
                    sub_bracket_orders["tsl"] = tsl_compact

                unfilled_orders[sub_order["order_uuid"]] = sub_bracket_orders
            results["uo"] = unfilled_orders  # index as uo to match positions

        if self.execution_type == ExecutionType.STOP_LIMIT:
            results["sp"] = self.stop_price
            results["cond"] = self.stop_condition.name

        return results

    def __str__(self):
        # Ensuring the `trade_pair.trade_pair_id` is accessible for the string representation
        # This assumes that trade_pair_id is a valid attribute of trade_pair
        d = self.to_python_dict()
        return str(d)
