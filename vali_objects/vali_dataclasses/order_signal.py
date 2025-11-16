# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
from typing import Optional
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from pydantic import BaseModel, field_validator, model_validator
from vali_objects.vali_config import ValiConfig

class Signal(BaseModel):
    trade_pair: TradePair
    order_type: OrderType
    leverage: float

    execution_type: ExecutionType = ExecutionType.MARKET
    limit_price: Optional[float] = None

    @field_validator('leverage', mode='before')
    def set_leverage(cls, leverage, info):
        order_type = info.data.get('order_type')
        if order_type == OrderType.LONG and leverage < 0:
            raise ValueError("Leverage must be positive for LONG orders.")
        elif order_type == OrderType.SHORT:
            leverage = -1.0 * abs(leverage)
        return leverage

    @model_validator(mode='before')
    def validate_trade_pair_and_leverage(cls, values):
        trade_pair = values['trade_pair']
        order_type = values['order_type']
        lev = values['leverage']
        is_flat_order = order_type == OrderType.FLAT or order_type == 'FLAT'
        if not is_flat_order and trade_pair and not (ValiConfig.ORDER_MIN_LEVERAGE <= abs(lev) <= ValiConfig.ORDER_MAX_LEVERAGE):
            raise ValueError(
                f"Order leverage must be between {ValiConfig.ORDER_MIN_LEVERAGE} and {ValiConfig.ORDER_MAX_LEVERAGE}, provided - lev [{lev}] and order_type [{order_type}] ({type(order_type)})")
        return values

    @staticmethod
    def parse_trade_pair_from_signal(signal) -> TradePair | None:
        if not signal or not isinstance(signal, dict):
            return None
        if 'trade_pair' not in signal:
            return None
        temp = signal["trade_pair"]
        if 'trade_pair_id' not in temp:
            return None
        string_trade_pair = signal["trade_pair"]["trade_pair_id"]
        trade_pair = TradePair.from_trade_pair_id(string_trade_pair)
        return trade_pair

    def __str__(self):
        base = {
            'trade_pair': str(self.trade_pair),
            'order_type': str(self.order_type),
            'leverage': self.leverage,
            'execution_type': self.execution_type
        }
        if self.execution_type == ExecutionType.MARKET:
            return str(base)

        elif self.execution_type == ExecutionType.LIMIT:
            base.update({
                'limit_price': self.limit_price
            })
            return str(base)

        elif self.execution_type == ExecutionType.LIMIT_CANCEL:
            # No extra fields needed - order_uuid comes from synapse.miner_order_uuid
            return str(base)

        return str({**base, 'Error': 'Unknown execution type'})

