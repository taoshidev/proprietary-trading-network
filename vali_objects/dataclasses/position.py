# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass
from typing import Optional, List

from vali_objects.dataclasses.base_objects.base_dataclass import BaseDataClass
from vali_objects.dataclasses.order import Order


@dataclass
class Position:
    miner_hotkey: str
    position_uuid: str
    open_ms: int
    trade_pair: str
    orders: list[Order]
    current_return: Optional[float] = 0
    max_drawdown: Optional[float] = 0
    close_ms: Optional[int] = None
    return_at_close: Optional[float] = None
    close_price: Optional[float] = None
