# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

from dataclasses import dataclass

from time_util.time_util import TimeUtil
from vali_config import TradePair
from pydantic import validator
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.signal import Signal
from enum import Enum, auto

class Order(Signal):
    price: float
    processed_ms: int
    order_uuid: str
    price_sources: list[PriceSource] = []

    @validator('price', 'processed_ms', 'leverage', pre=True, each_item=False)
    def validate_values(cls, v, values, field):
        if field.name == 'price' and v < 0:
            raise ValueError("Price must be greater than 0")
        if field.name == 'processed_ms' and v < 0:
            raise ValueError("processed_ms must be greater than 0")
        if field.name == 'leverage':
            order_type = values.get('order_type')
            if order_type == OrderType.LONG and v < 0:
                raise ValueError("Leverage must be positive for LONG orders.")
        return v

    # Using Pydantic's constructor instead of a custom from_dict method
    @classmethod
    def from_dict(cls, order_dict):
        # This method is now simplified as Pydantic can automatically
        # handle the conversion from dict to model instance
        return cls(**order_dict)

    def get_order_age(self, order):
        return TimeUtil.now_in_millis() - order.processed_ms

    def __str__(self):
        # Ensuring the `trade_pair.trade_pair_id` is accessible for the string representation
        # This assumes that trade_pair_id is a valid attribute of trade_pair
        trade_pair_id = self.trade_pair.trade_pair_id if hasattr(self.trade_pair, 'trade_pair_id') else 'unknown'
        return str({'trade_pair': trade_pair_id,
                    'order_type': self.order_type.name,
                    'leverage': self.leverage,
                    'price': self.price,
                    'processed_ms': self.processed_ms,
                    'price_sources': self.price_sources,
                    'order_uuid': self.order_uuid})

class OrderStatus(Enum):
    OPEN = auto()
    CLOSED = auto()
    ALL = auto()  # Represents both or neither, depending on your logic

