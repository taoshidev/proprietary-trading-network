import json
from dataclasses import dataclass
from enum import Enum

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


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


class TradePair(Enum):
    BTCUSD = ("BTCUSD", "BTC/USD", 0.003, 0.0001, 20)
    ETHUSD = ("ETHUSD", "ETH/USD", 0.003, 0.0001, 20)
    EURUSD = ("EURUSD", "EUR/USD", 0.0003, 0.0001, 100)
    SPX = ("SPX", "SPX", 0.0005, 0.0001, 100)

    def __init__(
        self,
        trade_pair_id: str,
        trade_pair: str,
        fees: float,
        min_leverage: float,
        max_leverage: float,
    ):
        self.trade_pair_id = trade_pair_id
        self.trade_pair = trade_pair
        self.fees = fees
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage

    @staticmethod
    def to_dict():
        # Convert TradePair Enum to a dictionary
        return {
            member.name: {
                "stream_id": member.trade_pair_id,
                "trade_pair": member.trade_pair,
                "fees": member.fees,
                "min_leverage": member.min_leverage,
                "max_leverage": member.max_leverage,
            }
            for member in TradePair
        }

    @staticmethod
    def to_enum(stream_id):
        m_map = {member.name: member for member in TradePair}
        return m_map[stream_id]

    @staticmethod
    def pair_map():
        return {pair.trade_pair_id: pair for pair in TradePair}

    @staticmethod
    def get_trade_pair(trade_pair_id: str):
        tp_map = TradePair.pair_map()
        return tp_map[trade_pair_id]

    def __str__(self):
        return str(
            {
                "trade_pair_id": self.trade_pair_id,
                "trade_pair": self.trade_pair,
                "fees": self.fees,
                "min_leverage": self.min_leverage,
                "max_leverage": self.max_leverage,
            }
        )


@dataclass
class Signal:
    trade_pair: TradePair
    order_type: OrderType
    leverage: float

    def __post_init__(self):
        if not isinstance(self.order_type, OrderType):
            raise ValueError(
                f"Order type value received is not of type trade pair [{self.order_type}]."
            )

        if not isinstance(self.trade_pair, TradePair):
            raise ValueError(
                f"Trade pair value received is not of type trade pair [{self.trade_pair}]."
            )

        if not isinstance(self.leverage, float):
            raise ValueError(
                f"Leverage value received is not of type float [{self.leverage}]."
            )

        if (self.order_type == OrderType.SHORT and self.leverage > 0) or (
            self.order_type == OrderType.LONG and self.leverage < 0
        ):
            raise ValueError(
                "Leverage must be negative for SHORT orders and positive for LONG orders."
            )

        if self.order_type != OrderType.FLAT and (
            (
                self.trade_pair.min_leverage
                < abs(self.leverage)
                < self.trade_pair.max_leverage
            )
            is False
        ):
            raise ValueError(
                f"Leverage must be greater than [{self.trade_pair.min_leverage}] and"
                f" less than [{self.trade_pair.max_leverage}]."
                f"Leverage provided - [{self.leverage}]"
            )

    def __str__(self):
        return str(
            {
                "trade_pair": str(self.trade_pair),
                "order_type": str(self.order_type),
                "leverage": self.leverage,
            }
        )


class CustomJSONDecoder(json.JSONDecoder):
    def decode(self, s, *args, **kwargs):
        # Replace single quotes with double quotes
        s = s.replace("'", '"')

        # Unescape inner JSON string
        s = s.replace('\\"', '"')

        # Remove extra double quotes at the beginning and end of the string
        s = s.strip('"')

        # Unescape inner JSON string
        s = s.replace('"{', "{").replace('}"', "}")

        # Decode the string to a JSON object
        return super().decode(s, *args, **kwargs)


if __name__ == "__main__":
    received_signals = ValiBkpUtils.get_all_files_in_dir("mining/received_signals/")

    test1 = ValiBkpUtils.get_file(received_signals[0])
    json_obj = json.loads(test1, cls=CustomJSONDecoder)

    # Replace single quotes with double quotes
    # json_string = test1.replace("'", '"')
    #
    # # Unescape inner JSON string
    # json_string = json_string.replace('\\"', '"')
    #
    # # Remove extra double quotes at the beginning and end of the string
    # input_string = json_string.strip('"')
    #
    # # Unescape inner JSON string
    # input_string = input_string.replace('"{', "{").replace('}"', "}")

    # Convert the string to a JSON object
    # json_obj = json.loads(input_string)

    print(json_obj)

    # print(test1)
    # test1 = json.loads(ValiBkpUtils.get_file(received_signals[0])

    # test = str(signal)
    # print(test.replace('"', "'"))

    # test1 = "{'trade_pair': "{'trade_pair_id': 'BTCUSD', 'trade_pair': 'BTC/USD', 'fees': 0.003, 'min_leverage': 0.0001, 'max_leverage': 20}", 'order_type': 'LONG', 'leverage': 0.5}"
    # test1.replace('"', "'")
