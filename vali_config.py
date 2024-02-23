# developer: Taoshi
# Copyright © 2023 Taoshi Inc
import os
from enum import Enum

from time_util.time_util import TimeUtil


class TradePair(Enum):
    BTCUSD = "BTC/USD"
    ETHUSD = "ETH/USD"


class ValiConfig:
    # fees take into account exiting and entering a position, liquidity, and futures fees
    TRADE_PAIR_FEES = {
        TradePair.BTCUSD: 0.003,
        TradePair.ETHUSD: 0.003
    }

    MIN_LEVERAGE = 0.001
    MAX_DAILY_DRAWDOWN = 0.05
    MAX_TOTAL_DRAWDOWN = 0.1
    MAX_ORDERS = 200

    SET_WEIGHT_INTERVALS = [0, 30]
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = 30

    SET_WEIGHT_MINIMUM_POSITIONS = 3

    ORDER_SIMILARITY_WINDOW_MS = TimeUtil.hours_in_millis(24)

    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

