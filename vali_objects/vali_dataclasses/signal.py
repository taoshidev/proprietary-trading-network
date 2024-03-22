# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from pydantic import BaseModel, validator

class Signal(BaseModel):
    trade_pair: TradePair
    order_type: OrderType
    leverage: float

    @validator('leverage')
    def set_leverage(cls, leverage, values):
        if 'order_type' in values:
            order_type = values['order_type']
            if order_type == OrderType.LONG and leverage < 0:
                raise ValueError("Leverage must be positive for LONG orders.")
            elif order_type == OrderType.SHORT:
                leverage = -1.0 * abs(leverage)
        return leverage

    @validator('trade_pair', 'order_type', 'leverage')
    def validate_trade_pair_and_leverage(cls, v, field, values):
        if field.name == 'leverage':
            trade_pair = values.get('trade_pair')
            order_type = values.get('order_type')
            is_flat_order = order_type == OrderType.FLAT
            if (not is_flat_order) and trade_pair and (not trade_pair.min_leverage <= abs(v) <= trade_pair.max_leverage):
                raise ValueError(
                    f"Leverage must be between {trade_pair.min_leverage} and {trade_pair.max_leverage}, provided - [{v}]")
        return v

    def __str__(self):
        return str({'trade_pair': str(self.trade_pair),
                    'order_type': str(self.order_type),
                    'leverage': self.leverage})
