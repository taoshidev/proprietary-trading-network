# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass

from vali_objects.dataclasses.base_objects.base_dataclass import BaseDataClass
from vali_objects.enums.order_type_enum import OrderTypeEnum


@dataclass
class Order(BaseDataClass):
    trade_pair: str
    order_type: OrderTypeEnum
    leverage: float
    price: float
    processed_ms: int
    order_uuid: str

    def __eq__(self, other):
        return self.equal_base_class_check(other)
