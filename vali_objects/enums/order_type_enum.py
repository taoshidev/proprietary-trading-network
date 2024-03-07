from enum import Enum


class OrderType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    def __str__(self):
        return self.value

    @staticmethod
    def order_type_map():
        return {ote.value: ote for ote in OrderType}

    @staticmethod
    def get_order_type(order_type_value: str):
        otm = OrderType.order_type_map()
        return otm[order_type_value]