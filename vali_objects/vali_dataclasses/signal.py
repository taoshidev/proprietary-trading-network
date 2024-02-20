# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass

from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderTypeEnum


@dataclass
class Signal:
    trade_pair: TradePair
    order_type: OrderTypeEnum
    leverage: float

