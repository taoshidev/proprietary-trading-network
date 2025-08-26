import json

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.utils.live_price_fetcher import LivePriceFetcher

positions_to_snap = [
     #Added FLAT order - Exited position on  2025-07-18 13:24:00 (Price - 148.2150)
     {
        "miner_hotkey": "5HeNFBvP8MP1RrZMQwg5Pk3fxpecahRHxzMovRHbz7EufHdF",
        "position_uuid": "3562356a-a178-4fbc-b97a-d131ca95f634",
        "open_ms": 1752763757701,
        "trade_pair": [
            "USDJPY",
            "USD/JPY",
            0.00007,
            0.1,
            5
        ],
        "orders": [
            {
                "order_type": "SHORT",
                "leverage": -1,
                "price": 148.506,
                "bid": 148.5045,
                "ask": 148.5075,
                "slippage": 0.000012957052320443017,
                "processed_ms": 1752763757701,
                "order_uuid": "3562356a-a178-4fbc-b97a-d131ca95f634",
                "price_sources": [],
                "src": 0
            },
            {
                "order_type": "FLAT",
                "leverage": 1,
                "price": 148.2150,
                "bid": 148.2150,
                "ask": 148.2150,
                "slippage": 0.000012957052320443017,
                "processed_ms": 1752845040000,
                "order_uuid": "f0f70286-eeff-4db3-a625-1130f4a0975f",
                "price_sources": [],
                "src": 0
            }
        ],
        "current_return": 1.010815028418151,
        "close_ms": 0,
        "net_leverage": -1,
        "return_at_close": 1.0104776751399407,
        "average_entry_price": 148.5040757999881,
        "cumulative_entry_value": -148.5040757999881,
        "realized_pnl": 0,
        "position_type": "SHORT",
        "is_closed_position": True
    },

    #Updated FLAT order - Exited position on  2025-07-02 12:25:00 (Price - 107237.030)
    {
        "miner_hotkey": "5GCALDKDfzFTC5MLT45JK7KDZECxsonR67fKcxV1uPPAm4c8",
        "position_uuid": "970cf66b-75c5-429e-b0ae-25cd9e7622c5",
        "open_ms": 1751455487902,
        "trade_pair": [
            "BTCUSD",
            "BTC/USD",
            0.003,
            0.01,
            0.5
        ],
        "orders": [
            {
                "order_type": "SHORT",
                "leverage": -0.5,
                "price": 107798.46,
                "bid": 0,
                "ask": 0,
                "slippage": 0,
                "processed_ms": 1751455487902, # Wednesday, July 2, 2025 11:24:47.902 AM
                "order_uuid": "970cf66b-75c5-429e-b0ae-25cd9e7622c5",
                "price_sources": [],
                "src": 0
            },
            {
                "order_type": "FLAT",
                "leverage": 0.5,
                "price": 107237.030,
                "bid": 107237.030,
                "ask": 107237.030,
                "slippage": 0,
                "processed_ms": 1751459100000, # Wednesday, July 2, 2025 12:25:00 PM
                "order_uuid": "18d48d82-33eb-4b1b-99a1-3024b717cbd3",
                "price_sources": [],
                "src": 0
            }
    ],
        "current_return": 0.9854724733544431,
        "close_ms": 1752156439011,
        "net_leverage": 0,
        "return_at_close": 0.9841686860352171,
        "average_entry_price": 107798.46,
        "cumulative_entry_value": -53899.23,
        "realized_pnl": -1566.0449999999983,
        "position_type": "FLAT",
        "is_closed_position": True
    },

    #Updated FLAT order - Exited position on  2025-07-02 11:53:00 (Price - 2.1745)
    {
        "miner_hotkey": "5GCALDKDfzFTC5MLT45JK7KDZECxsonR67fKcxV1uPPAm4c8",
        "position_uuid": "e849229d-eb67-4c46-88cb-eb16a618a7b4",
        "open_ms": 1751455509159,
        "trade_pair": [
            "XRPUSD",
            "XRP/USD",
            0.001,
            0.01,
            0.5
        ],
        "orders": [
            {
                "order_type": "SHORT",
                "leverage": -0.5,
                "price": 2.1856,
                "bid": 0,
                "ask": 0,
                "slippage": 0,
                "processed_ms": 1751455509159,
                "order_uuid": "e849229d-eb67-4c46-88cb-eb16a618a7b4",
                "price_sources": [],
                "src": 0
            },
            {
                "order_type": "FLAT",
                "leverage": 0.5,
                "price": 2.1745,
                "bid": 2.1745,
                "ask": 2.1745,
                "slippage": 0,
                "processed_ms": 1751457180000,
                "order_uuid": "5eec2472-b73c-4f45-b4cb-f3e5b87147f0",
                "price_sources": [],
                "src": 0
            }
        ],
        "current_return": 0.9446833821376281,
        "close_ms": 1752156475324,
        "net_leverage": 0,
        "return_at_close": 0.9434335590856234,
        "average_entry_price": 2.1856,
        "cumulative_entry_value": -1.0928,
        "realized_pnl": -0.12090000000000001,
        "position_type": "FLAT",
        "is_closed_position": True
    },

    #Updated FLAT order - Exited position on  2025-06-26 20:33:00 (Price - 2.1380)
    {
        "miner_hotkey": "5GCALDKDfzFTC5MLT45JK7KDZECxsonR67fKcxV1uPPAm4c8",
        "position_uuid": "0993669f-1b89-478b-9d67-241f950a6f8a",
        "open_ms": 1750958986779,
        "trade_pair": [
            "XRPUSD",
            "XRP/USD",
            0.001,
            0.01,
            0.5
        ],
        "orders": [
            {
                "order_type": "SHORT",
                "leverage": -0.5,
                "price": 2.1183,
                "bid": 0,
                "ask": 0,
                "slippage": 0,
                "processed_ms": 1750958986779,
                "order_uuid": "0993669f-1b89-478b-9d67-241f950a6f8a",
                "price_sources": [],
                "src": 0
            },
            {
                "order_type": "FLAT",
                "leverage": 0.5,
                "price": 2.1380,
                "bid": 2.1380,
                "ask": 2.1380,
                "slippage": 0,
                "processed_ms": 1750969980000,
                "order_uuid": "7b013601-2d85-4f28-8546-27d3b7ff0b05",
                "price_sources": [],
                "src": 0
            }
        ],
        "current_return": 1.0052636548175424,
        "close_ms": 1751015755948,
        "net_leverage": 0,
        "return_at_close": 1.005157192314982,
        "average_entry_price": 2.1183,
        "cumulative_entry_value": -1.05915,
        "realized_pnl": 0.011149999999999993,
        "position_type": "FLAT",
        "is_closed_position": True
    },

    #Updated FLAT order - Exited position on 2025-06-27 15:40:00 (Price - 107230)
    {
        "miner_hotkey": "5GCALDKDfzFTC5MLT45JK7KDZECxsonR67fKcxV1uPPAm4c8",
        "position_uuid": "67de7bbe-2073-44e2-841e-aa7d7d52469c",
        "open_ms": 1751036438871,
        "trade_pair": [
            "BTCUSD",
            "BTC/USD",
            0.003,
            0.01,
            0.5
        ],
        "orders": [
            {
                "order_type": "SHORT",
                "leverage": -0.5,
                "price": 106632,
                "bid": 0,
                "ask": 0,
                "slippage": 0,
                "processed_ms": 1751036438871, # Friday, June 27, 2025 3:00:38.871 PM
                "order_uuid": "67de7bbe-2073-44e2-841e-aa7d7d52469c",
                "price_sources": [],
                "src": 0
            },
            {
                "order_type": "FLAT",
                "leverage": 0.5,
                "price": 107230,
                "bid": 107230,
                "ask": 107230,
                "slippage": 0,
                "processed_ms": 1751038800000, # Friday, June 27, 2025 3:40:00 PM
                "order_uuid": "0889361f-2457-4f38-9917-7f497d38a200",
                "price_sources": [],
                "src": 0
            }
        ],
        "current_return": 1.000150048765849,
        "close_ms": 1751036884037,
        "net_leverage": 0,
        "return_at_close": 1.000150048765849,
        "average_entry_price": 106632,
        "cumulative_entry_value": -53316,
        "realized_pnl": 16,
        "position_type": "FLAT",
        "is_closed_position": False
    },

    # NO CHANGE.
    {
        "miner_hotkey": "5GCALDKDfzFTC5MLT45JK7KDZECxsonR67fKcxV1uPPAm4c8",
        "position_uuid": "c1651959-ba63-489f-ab00-b0e9ffb1d6ca",
        "open_ms": 1750958670317,
        "trade_pair": ["BTCUSD", "BTC/USD", 0.003, 0.01, 0.5],
        "orders": [
            {
                "order_type": "SHORT", "leverage": -0.5, "price": 107334.88, "bid": 0.0, "ask": 0.0, "slippage": 0.0,
                "processed_ms": 1750958670317,  # Thursday, June 26, 2025 5:24:30.317 PM
                "order_uuid": "c1651959-ba63-489f-ab00-b0e9ffb1d6ca", "price_sources": [], "src": 0},
            {
                "order_type": "FLAT", "leverage": 0.5, "price": 106641.69, "bid": 0.0, "ask": 0.0, "slippage": 0.0,
                "processed_ms": 1751035621953,  # Friday, June 27, 2025 2:47:01.953 PM
                "order_uuid": "6c8f77b0-4a7b-45c3-a075-b31913318ecc", "price_sources": [], "src": 0
            }],
        "current_return": 1.0032290994316106, "close_ms": 1751035621953, "net_leverage": 0.0,
        "return_at_close": 1.003069733101931, "average_entry_price": 107334.88, "cumulative_entry_value": -53667.44,
        "realized_pnl": 346.59500000000116, "position_type": "FLAT", "is_closed_position": True},

    #Updated FLAT order - Exited position on  2025-07-02 16:09:00 (Price - 109623.540) CANCELED
    #{
    #    "miner_hotkey": "5GCALDKDfzFTC5MLT45JK7KDZECxsonR67fKcxV1uPPAm4c8",
    #    "position_uuid": "c1651959-ba63-489f-ab00-b0e9ffb1d6ca",
    #    "open_ms": 1750958670317, # Thursday, June 26, 2025 5:24:30.317 PM
    #    "trade_pair": [
    #        "BTCUSD",
    #        "BTC/USD",
    #        0.003,
    #        0.01,
    #        0.5
    #    ],
    #    "orders": [
    #        {
    #            "order_type": "SHORT",
    #            "leverage": -0.5,
    #            "price": 107334.88,
    #            "bid": 0,
    #            "ask": 0,
    #            "slippage": 0,
    #            "processed_ms": 1750958670317, #Thursday, June 26, 2025 5:24:30.317 PM
    #            "order_uuid": "c1651959-ba63-489f-ab00-b0e9ffb1d6ca",
    #            "price_sources": [],
    #            "src": 0
    #        },
    #        {
    #            "order_type": "FLAT",
    #            "leverage": 0.5,
    #            "price": 109623.540,
    #            "bid": 109623.5400,
    #            "ask": 109623.540,
    #            "slippage": 0,
    #            "processed_ms": 1751472540000, # Wednesday, July 2, 2025 4:09:00 PM
    #            "order_uuid": "6c8f77b0-4a7b-45c3-a075-b31913318ecc",
    #            "price_sources": [],
    #            "src": 0
    #        }
    #    ],
    #    "current_return": 1.0032290994316106,
    #    "close_ms": 1751035621953,
    #    "net_leverage": 0,
    #    "return_at_close": 1.003069733101931,
    #    "average_entry_price": 107334.88,
    #    "cumulative_entry_value": -53667.44,
    #    "realized_pnl": 346.59500000000116,
    #    "position_type": "FLAT",
    #    "is_closed_position": False
    #},

    #Updated FLAT order - Exited position on  2025-06-04 13:52:00 (Price - 19.1653)
    {
        "miner_hotkey": "5FCPYqbYEq2y7NwQTCLxNApP2UjUE86J8QnhdWTHFkzzFWL1",
        "position_uuid": "0d6f5217-a2f0-4083-a49b-c7ef75ea75a7",
        "open_ms": 1748998235683,
        "trade_pair": [
            "USDMXN",
            "USD/MXN",
            0.00007,
            0.1,
            5
        ],
        "orders": [
            {
                "order_type": "LONG",
                "leverage": 2,
                "price": 19.22629,
                "bid": 19.22629,
                "ask": 19.23276,
                "slippage": 0.0001579447751969581,
                "processed_ms": 1748998235683,
                "order_uuid": "0d6f5217-a2f0-4083-a49b-c7ef75ea75a7",
                "price_sources": [],
                "src": 0
            },
            {
                "order_type": "FLAT",
                "leverage": -2,
                "price": 19.1653,
                "bid": 19.1653,
                "ask": 19.1653,
                "slippage": 0.00012564130160352342,
                "processed_ms": 1749045120000,
                "order_uuid": "6f62a5da-64ca-4804-b796-6b8ccc2491f2",
                "price_sources": [],
                "src": 0
            }
        ],
        "current_return": 0.9814636672987944,
        "close_ms": 1749485306611,
        "net_leverage": 0,
        "return_at_close": 0.9796634536764902,
        "average_entry_price": 19.22932669205192,
        "cumulative_entry_value": 38.45865338410384,
        "realized_pnl": -0.3564411971840471,
        "position_type": "FLAT",
        "is_closed_position": False
    }

]
secrets = ValiUtils.get_secrets()
lpf = LivePriceFetcher(secrets, disable_ws=True)
for i, position_json in enumerate(positions_to_snap):
    # build the positions as the order edits did not propagate to position-level attributes.
    pos = Position(**position_json)
    pos.rebuild_position_with_updated_orders(lpf)
    positions_to_snap[i] = pos.model_dump()

if __name__ == "__main__":
    for position_json in positions_to_snap:
        pos = Position(**position_json)
        pos.rebuild_position_with_updated_orders()
        assert pos.is_closed_position
        #print(pos.to_copyable_str())
        str_to_write = json.dumps(pos, cls=CustomEncoder)

        print(pos.model_dump_json(), '\n', str_to_write)


