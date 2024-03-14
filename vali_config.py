# developer: Taoshi
# Copyright © 2023 Taoshi Inc
import json
import os
from enum import Enum

from time_util.time_util import TimeUtil


class TradePair(Enum):
    # crypto
    BTCUSD = ("BTCUSD", "BTC/USD", 0.003, 0.001, 20)
    ETHUSD = ("ETHUSD", "ETH/USD", 0.003, 0.001, 20)

    # forex
    AUDCAD = ("AUDCAD", "AUD/CAD", 0.0003, 0.001, 500)
    AUDUSD = ("AUDUSD", "AUD/USD", 0.0003, 0.001, 500)
    AUDJPY = ("AUDJPY", "AUD/JPY", 0.0003, 0.001, 500)

    CADCHF = ("CADCHF", "CAD/CHF", 0.0003, 0.001, 500)
    CADJPY = ("CADJPY", "CAD/JPY", 0.0003, 0.001, 500)

    CHFJPY = ("CHFJPY", "CHF/JPY", 0.0003, 0.001, 500)

    EURCAD = ("EURCAD", "EUR/CAD", 0.0003, 0.001, 500)
    EURUSD = ("EURUSD", "EUR/USD", 0.0003, 0.001, 500)
    EURCHF = ("EURCHF", "EUR/CHF", 0.0003, 0.001, 500)
    EURGBP = ("EURGBP", "EUR/GBP", 0.0003, 0.001, 500)
    EURJPY = ("EURJPY", "EUR/JPY", 0.0003, 0.001, 500)
    EURNZD = ("EURNZD", "EUR/NZD", 0.0003, 0.001, 500)

    NZDCAD = ("NZDCAD", "NZD/CAD", 0.0003, 0.001, 500)
    NZDJPY = ("NZDJPY", "NZD/JPY", 0.0003, 0.001, 500)

    GBPUSD = ("GBPUSD", "GBP/USD", 0.0003, 0.001, 500)
    GBPJPY = ("GBPJPY", "GBP/JPY", 0.0003, 0.001, 500)

    USDCAD = ("USDCAD", "USD/CAD", 0.0003, 0.001, 500)
    USDCHF = ("USDCHF", "USD/CHF", 0.0003, 0.001, 500)
    USDJPY = ("USDJPY", "USD/JPY", 0.0003, 0.001, 500)

    # indices
    SPX = ("SPX", "SPX", 0.0005, 0.001, 500)
    DJI = ("DJI", "DJI", 0.0005, 0.001, 500)
    FTSE = ("FTSE", "FTSE", 0.0005, 0.001, 500)
    GDAXI = ("GDAXI", "GDAXI", 0.0005, 0.001, 500)

    def __init__(
        self,
        trade_pair_id: str,
        trade_pair: str = None,
        fees: float = None,
        min_leverage: float = None,
        max_leverage: float = None,
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
                "trade_pair_id": member.trade_pair_id,
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

    def __json__(self):
        # Provide a dictionary representation for JSON serialization
        return {
            "trade_pair_id": self.trade_pair_id,
            "trade_pair": self.trade_pair,
            "fees": self.fees,
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
        }

    def __str__(self):
        return str(self.__json__())


class ValiConfig:
    # fees take into account exiting and entering a position, liquidity, and futures fees
    MDD_CHECK_REFRESH_TIME_S = 15  # 15 seconds
    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_OPEN_ORDERS_PER_HOTKEY = 200

    SET_WEIGHT_REFRESH_TIME_S = 60 * 30  # 30 minutes
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = 30

    SET_WEIGHT_MINIMUM_POSITIONS = 3

    ORDER_SIMILARITY_WINDOW_MS = TimeUtil.hours_in_millis(24)

    MINER_COPYING_WEIGHT = 0.01

    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

    METAGRAPH_UPDATE_REFRESH_TIME_S = 60 * 5  # 5 minutes

    ELIMINATION_CHECK_INTERVAL_S = 60 * 5  # 5 minutes
    ELIMINATION_FILE_DELETION_DELAY_S = 60 * 30  # 30 min
    MAX_MINER_PLAGIARISM_SCORE = 1.0

    TOP_MINER_BENEFIT = 0.8
    TOP_MINER_PERCENT = 0.2

