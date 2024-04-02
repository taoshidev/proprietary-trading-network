# developer: Taoshi
# Copyright © 2024 Taoshi Inc
import json
import os
from enum import Enum

from time_util.time_util import TimeUtil

class TradePairCategory(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    INDICES = "indices"

class TradePair(Enum):
    # crypto
    BTCUSD = ["BTCUSD", "BTC/USD", 0.002, 0.001, 20, TradePairCategory.CRYPTO]
    ETHUSD = ["ETHUSD", "ETH/USD", 0.002, 0.001, 20, TradePairCategory.CRYPTO]

    # forex
    AUDCAD = ["AUDCAD", "AUD/CAD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    AUDUSD = ["AUDUSD", "AUD/USD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    AUDJPY = ["AUDJPY", "AUD/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    CADCHF = ["CADCHF", "CAD/CHF", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    CADJPY = ["CADJPY", "CAD/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    CHFJPY = ["CHFJPY", "CHF/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    EURCAD = ["EURCAD", "EUR/CAD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    EURUSD = ["EURUSD", "EUR/USD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    EURCHF = ["EURCHF", "EUR/CHF", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    EURGBP = ["EURGBP", "EUR/GBP", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    EURJPY = ["EURJPY", "EUR/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    EURNZD = ["EURNZD", "EUR/NZD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    NZDCAD = ["NZDCAD", "NZD/CAD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    NZDJPY = ["NZDJPY", "NZD/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]

    # indices
    SPX = ["SPX", "SPX", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    DJI = ["DJI", "DJI", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    FTSE = ["FTSE", "FTSE", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, 0.001, 500, TradePairCategory.INDICES]

    @property
    def trade_pair_id(self):
        return self.value[0]

    @property
    def trade_pair(self):
        return self.value[1]

    @property
    def fees(self):
        return self.value[2]

    @property
    def min_leverage(self):
        return self.value[3]

    @property
    def max_leverage(self):
        return self.value[4]

    @property
    def trade_pair_category(self):
        return self.value[5]

    @property
    def is_crypto(self):
        return self.trade_pair_category == TradePairCategory.CRYPTO

    @property
    def is_forex(self):
        return self.trade_pair_category == TradePairCategory.FOREX

    @property
    def is_indices(self):
        return self.trade_pair_category == TradePairCategory.INDICES

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
    def from_trade_pair_id(trade_pair_id: str):
        """
        Converts a trade_pair_id string into a TradePair object.

        Args:
            trade_pair_id (str): The ID of the trade pair to convert.

        Returns:
            TradePair: The corresponding TradePair object.

        Raises:
            ValueError: If no matching trade pair is found.
        """
        if trade_pair_id in TRADE_PAIR_ID_TO_TRADE_PAIR:
            return TRADE_PAIR_ID_TO_TRADE_PAIR[trade_pair_id]
        else:
            return None

    def __json__(self):
        # Provide a dictionary representation for JSON serialization
        return {
            "trade_pair_id": self.trade_pair_id,
            "trade_pair": self.trade_pair,
            "fees": self.fees,
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
            "trade_pair_category": self.trade_pair_category,
        }
    
    def __dict__(self):
        return self.__json__()

    @staticmethod
    def get_latest_trade_pair_from_trade_pair_id(trade_pair_id):
        return TRADE_PAIR_ID_TO_TRADE_PAIR[trade_pair_id]

    def __str__(self):
        return str(self.__json__())


TRADE_PAIR_ID_TO_TRADE_PAIR = {x.trade_pair_id: x for x in TradePair}


class ValiConfig:
    ## versioning
    VERSION = "2.1.0"

    # fees take into account exiting and entering a position, liquidity, and futures fees
    MDD_CHECK_REFRESH_TIME_MS = 15 * 1000  # 15 seconds
    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_OPEN_ORDERS_PER_HOTKEY = 200
    ORDER_COOLDOWN_MS = 60000  # 1 minute

    SET_WEIGHT_REFRESH_TIME_MS = 60 * 5 * 1000  # 5 minutes
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = 30
    SET_WEIGHT_LOOKBACK_RANGE_MS = SET_WEIGHT_LOOKBACK_RANGE_DAYS * 24 * 60 * 60 * 1000

    HISTORICAL_DECAY_TIME_INTENSITY_COEFFICIENT = 0.35
    ANNUAL_RISK_FREE_RATE = 0.02
    DAYS_IN_YEAR = 365.25
    LOOKBACK_RANGE_DAYS_RISK_FREE_RATE = (1+ANNUAL_RISK_FREE_RATE)**(SET_WEIGHT_LOOKBACK_RANGE_DAYS/DAYS_IN_YEAR) - 1

    PROBABILISTIC_SHARPE_RATIO_THRESHOLD = 0.0
    OMEGA_RATIO_THRESHOLD = 0.0 # in reality, this would be the risk free rate - but we just want the functionality to compare the magnitude of gains and losses internally for our scoring function
    OMEGA_MINIMUM_DENOMINATOR = 1e-6
    PROBABILISTIC_SHARPE_RATIO_MIN_STD_DEV = 1e-6

    SET_WEIGHT_MINIMUM_POSITIONS = 10 # mimumum number of positions over the lookback range
    SET_WEIGHT_MINIMUM_POSITION_DURATION_MS = 1 * 60 * 1000  # 1 minutes
    SET_WEIGHT_MINIMUM_POSITION_DURATION_TOTAL_MS = SET_WEIGHT_MINIMUM_POSITION_DURATION_MS * SET_WEIGHT_MINIMUM_POSITIONS

    ORDER_SIMILARITY_WINDOW_MS = TimeUtil.hours_in_millis(24)

    MINER_COPYING_WEIGHT = 0.01

    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

    METAGRAPH_UPDATE_REFRESH_TIME_MS = 60 * 5 * 1000  # 5 minutes

    ELIMINATION_CHECK_INTERVAL_MS = 60 * 5 * 1000  # 5 minutes
    ELIMINATION_FILE_DELETION_DELAY_MS = 60 * 30 * 1000  # 30 min

    MAX_MINER_PLAGIARISM_SCORE = 0.9 # want to make sure we're filtering out the bad actors
    TOP_MINER_BENEFIT = 0.9
    TOP_MINER_PERCENT = 0.1
