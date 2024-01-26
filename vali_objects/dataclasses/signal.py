# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from dataclasses import dataclass

from vali_objects.dataclasses.base_objects.base_dataclass import BaseDataClass


@dataclass
class Signal(BaseDataClass):
    leverage: float
    order_type: str

    def __eq__(self, other):
        return self.equal_base_class_check(other)
