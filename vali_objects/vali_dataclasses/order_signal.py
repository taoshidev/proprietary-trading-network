# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
from typing import Optional

from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from pydantic import BaseModel, model_validator

class Signal(BaseModel):
    trade_pair: TradePair
    order_type: OrderType
    leverage: Optional[float] = None
    value: Optional[float] = None

    @model_validator(mode='before')
    def check_exclusive_fields(cls, values):
        """
        Ensure that only ONE of leverage or value is filled
        """
        fields = ['leverage', 'value']
        filled = [f for f in fields if values.get(f) is not None]
        if len(filled) != 1:
            raise ValueError(f"Exactly one of {fields} must be provided, got {filled}")
        return values

    @model_validator(mode='before')
    def set_size(cls, values):
        """
        Ensure that long orders have positive size, and short orders have negative size,
        applied to all non-None of leverage or value.
        """
        order_type = values['order_type']

        for field in ['leverage', 'value']:
            size = values.get(field)
            if size is not None:
                if order_type == OrderType.LONG and size < 0:
                    raise ValueError(f"{field} must be positive for LONG orders.")
                elif order_type == OrderType.SHORT:
                    values[field] = -1.0 * abs(size)
        return values

    def __str__(self):
        return str({'trade_pair': str(self.trade_pair),
                    'order_type': str(self.order_type),
                    'leverage': self.leverage,
                    'value': self.value
                    })
