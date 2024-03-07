# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass
from vali_objects.vali_dataclasses.signal import Signal


@dataclass
class Order(Signal):
    price: float
    processed_ms: int
    order_uuid: str
        
    def __str__(self):
        return str({'trade_pair': str(self.trade_pair), 'order_type': str(self.order_type), 'leverage': self.leverage, 
                'price': self.price, 'processed_ms': self.processed_ms, 'order_uuid': self.order_uuid})
