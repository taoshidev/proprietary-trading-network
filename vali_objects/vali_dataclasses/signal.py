# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass

from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType


@dataclass
class Signal:
    trade_pair: TradePair
    order_type: OrderType
    leverage: float


    def __post_init__(self):
        if isinstance(self.leverage, int):
            self.leverage = float(self.leverage)

        if not isinstance(self.order_type, OrderType):
            raise ValueError(f"Order type value received is not of type trade pair [{self.order_type}].")

        if not isinstance(self.trade_pair, TradePair):
            raise ValueError(f"Trade pair value received is not of type trade pair [{self.trade_pair}].")

        if not isinstance(self.leverage, float):
            raise ValueError(f"Leverage value received is not of type float [{self.leverage}].")

        if (self.order_type == OrderType.SHORT and self.leverage > 0) or \
           (self.order_type == OrderType.LONG and self.leverage < 0):
            raise ValueError("Leverage must be negative for SHORT orders and positive for LONG orders.")

        if (self.order_type != OrderType.FLAT
                and ((self.trade_pair.min_leverage < abs(self.leverage) < self.trade_pair.max_leverage) is False)):
            raise ValueError(f"Leverage must be greater than [{self.trade_pair.min_leverage}] and"
                             f" less than [{self.trade_pair.max_leverage}]."
                             f"Leverage provided - [{self.leverage}]")

    def __str__(self):
        return str({'trade_pair': str(self.trade_pair),
                    'order_type': str(self.order_type),
                    'leverage': self.leverage})
