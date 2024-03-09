# developer: Taoshi
# Copyright © 2023 Taoshi Inc
import os
from enum import Enum

from time_util.time_util import TimeUtil


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


class ValiConfig:
    # fees take into account exiting and entering a position, liquidity, and futures fees
    TRADE_PAIR_FEES = {TradePair.BTCUSD: 0.003, TradePair.ETHUSD: 0.003}

    MDD_CHECK_REFRESH_TIME_S = 15 # 15 seconds
    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_ORDERS = 200

    SET_WEIGHT_REFRESH_TIME_S = 60 * 30  # 30 minutes
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = 30

    SET_WEIGHT_MINIMUM_POSITIONS = 3

    ORDER_SIMILARITY_WINDOW_MS = TimeUtil.hours_in_millis(24)

    MINER_COPYING_WEIGHT = 0.01

    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

    METAGRAPH_UPDATE_REFRESH_TIME_S = 60 * 5  # 5 minutes

    ELIMINATION_FILE_DELETION_DELAY_S = 60 * 30  # 30 min
