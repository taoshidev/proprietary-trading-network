from enum import Enum


class OrderTypeEnum(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


    def __str__(self):
        return self.value
