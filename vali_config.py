# developer: Taoshi
# Copyright © 2023 Taoshi Inc
import os
from enum import Enum

from time_util.time_util import TimeUtil


class TradePair(Enum):
    BTCUSD = "BTC/USD"
    ETHUSD = "ETH/USD"

    def __str__(self):
        return self.value


class ValiConfig:
    # fees take into account exiting and entering a position, liquidity, and futures fees
    TRADE_PAIR_FEES = {
        TradePair.BTCUSD: 0.003,
        TradePair.ETHUSD: 0.003
    }

    MIN_LEVERAGE = 0.001
    MAX_DAILY_DRAWDOWN = 0.95 # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9 # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_ORDERS = 200

    SET_WEIGHT_INTERVALS = [0, 30]
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = 30

    SET_WEIGHT_MINIMUM_POSITIONS = 3

    ORDER_SIMILARITY_WINDOW_MS = TimeUtil.hours_in_millis(24)

    MINER_COPYING_WEIGHT = 0.01

    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

