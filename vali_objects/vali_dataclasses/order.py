# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderTypeEnum
from dataclasses import dataclass


@dataclass
class Order:
    trade_pair: TradePair
    order_type: OrderTypeEnum
    leverage: float
    price: float
    processed_ms: int
    order_uuid: str

    def __post_init__(self):
        if (self.order_type == OrderTypeEnum.SHORT and self.leverage > 0) or \
           (self.order_type == OrderTypeEnum.LONG and self.leverage < 0):
            raise ValueError("Leverage must be negative for SHORT orders and positive for LONG orders.")
        if abs(self.leverage) < ValiConfig.MIN_LEVERAGE and self.order_type != OrderTypeEnum.FLAT:
            raise ValueError(f"Leverage must be greater than [{ValiConfig.MIN_LEVERAGE}]."
                             f"Leverage provided - [{self.leverage}]")

        trade_pair_map = [trade_pair.trade_pair_id for trade_pair in TradePair]
        if self.trade_pair not in trade_pair_map:
            raise ValueError(f"Trade pair passed isn't an option for this subnet [{self.trade_pair}].")
