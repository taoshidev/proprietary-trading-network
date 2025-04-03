# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

from time_util.time_util import TimeUtil
from pydantic import field_validator

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.vali_dataclasses.price_source import PriceSource
from enum import Enum, auto

ORDER_SRC_ORGANIC = 0               # order generated from a miner's signal
ORDER_SRC_ELIMINATION_FLAT = 1      # order inserted when a miner is eliminated
ORDER_SRC_DEPRECATION_FLAT = 2      # order inserted when a trade pair is removed

class Order(Signal):
    price: float
    bid: float = 0
    ask: float = 0
    slippage: float = 0
    processed_ms: int
    order_uuid: str
    price_sources: list = []
    src: int = ORDER_SRC_ORGANIC

    @field_validator('price', 'processed_ms', 'leverage', mode='before')
    def validate_values(cls, v, info):
        if info.field_name == 'price' and v < 0:
            raise ValueError("Price must be greater than 0")
        if info.field_name == 'processed_ms' and v < 0:
            raise ValueError("processed_ms must be greater than 0")
        if info.field_name == 'leverage':
            order_type = info.data.get('order_type')
            if order_type == OrderType.LONG and v < 0:
                raise ValueError("Leverage must be positive for LONG orders.")
        return v

    @field_validator('order_uuid', mode='before')
    def ensure_order_uuid_is_string(cls, v):
        if not isinstance(v, str):
            v = str(v)
        return v

    @field_validator('price_sources', mode='before')
    def validate_price_sources(cls, v):
        if isinstance(v, list):
            return [PriceSource(**ps) if isinstance(ps, dict) else ps for ps in v]
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
                    'bid': self.bid,
                    'ask': self.ask,
                    'slippage': self.slippage,
                    'processed_ms': self.processed_ms,
                    'price_sources': self.price_sources,
                    'order_uuid': self.order_uuid,
                    'src': self.src})

class OrderStatus(Enum):
    OPEN = auto()
    CLOSED = auto()
    ALL = auto()  # Represents both or neither, depending on your logic

