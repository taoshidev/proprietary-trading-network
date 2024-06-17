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
    BTCUSD = ["BTCUSD", "BTC/USD", 0.001, 0.001, 20, TradePairCategory.CRYPTO]
    ETHUSD = ["ETHUSD", "ETH/USD", 0.001, 0.001, 20, TradePairCategory.CRYPTO]

    # forex
    AUDCAD = ["AUDCAD", "AUD/CAD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    AUDUSD = ["AUDUSD", "AUD/USD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    AUDJPY = ["AUDJPY", "AUD/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    AUDNZD = ["AUDNZD", "AUD/NZD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
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
    NZDUSD = ["NZDUSD", "NZD/USD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, 0.001, 500, TradePairCategory.FOREX]
    USDMXN = ["USDMXN", "USD/MXN", 0.00007, 0.001, 500, TradePairCategory.FOREX]

    # indices
    SPX = ["SPX", "SPX", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    DJI = ["DJI", "DJI", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    FTSE = ["FTSE", "FTSE", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    NDX = ["NDX", "NDX", 0.00009, 0.001, 500, TradePairCategory.INDICES]
    VIX = ["VIX", "VIX", 0.00009, 0.001, 500, TradePairCategory.INDICES]

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

    def debug_dict(self):
        return {
            "trade_pair_id": self.trade_pair_id,
            "trade_pair": self.trade_pair,
            "fees": self.fees,
            "min_leverage": self.min_leverage,
            "max_leverage": self.max_leverage,
        }

    @staticmethod
    def get_latest_trade_pair_from_trade_pair_id(trade_pair_id):
        return TRADE_PAIR_ID_TO_TRADE_PAIR.get(trade_pair_id)

    @staticmethod
    def get_latest_tade_pair_from_trade_pair_str(trade_pair_str):
        return TRADE_PAIR_STR_TO_TRADE_PAIR.get(trade_pair_str)

    def __str__(self):
        return str(self.__json__())


TRADE_PAIR_ID_TO_TRADE_PAIR = {x.trade_pair_id: x for x in TradePair}
TRADE_PAIR_STR_TO_TRADE_PAIR = {x.trade_pair: x for x in TradePair}

class ValiConfig:
    ## versioning
    VERSION = "2.8.9"

    # fees take into account exiting and entering a position, liquidity, and futures fees
    PERF_LEDGER_REFRESH_TIME_MS = 1000 * 60 * 5  # minutes
    CHALLENGE_PERIOD_REFRESH_TIME_MS = 1000 * 60 * 5  # minutes
    MDD_CHECK_REFRESH_TIME_MS = 60 * 1000  # 60 seconds
    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_TOTAL_DRAWDOWN_V2 = 0.95
    MAX_OPEN_ORDERS_PER_HOTKEY = 200
    ORDER_COOLDOWN_MS = 10000  # 10 seconds

    SET_WEIGHT_REFRESH_TIME_MS = 60 * 5 * 1000  # 5 minutes
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = 30
    SET_WEIGHT_LOOKBACK_RANGE_MS = SET_WEIGHT_LOOKBACK_RANGE_DAYS * 24 * 60 * 60 * 1000
    SET_WEIGHT_MINIMUM_UPDATES = 5

    PLAGIARISM_MATCHING_TIME_RESOLUTION_MS = 1 * 60 * 1000  # 1 minute
    PLAGIARISM_THRESHOLD = 0.9
    PLAGIARISM_MAX_LAGS = 60
    PLAGIARISM_LOOKBACK_RANGE_MS = 5 * 24 * 60 * 60 * 1000  # 5 days
    PLAGIARISM_FOLLOWER_TIMELAG_THRESHOLD = 1.005
    PLAGIARISM_FOLLOWER_SIMILARITY_THRESHOLD = 0.8

    HISTORICAL_DECAY_TIME_INTENSITY_COEFFICIENT = 0.18
    ANNUAL_RISK_FREE_RATE = 0.02
    DAYS_IN_YEAR = 365.25
    LOOKBACK_RANGE_DAYS_RISK_FREE_RATE = (1+ANNUAL_RISK_FREE_RATE)**(SET_WEIGHT_LOOKBACK_RANGE_DAYS/DAYS_IN_YEAR) - 1
    HISTORICAL_PENALTY_WINDOW = .175
    HISTORICAL_PENALTY_STRIDE = 0.05

    PROBABILISTIC_LOG_SHARPE_RATIO_THRESHOLD = 0.0

    # omega ratio threshold = np.log( 1 + threshold_percentage ) -> so value for 3% would be np.log(1.03)
    OMEGA_RATIO_THRESHOLD = 0.0 # in reality, this would be the risk free rate - but we just want the functionality 
    # to compare the magnitude of gains and losses internally for our scoring function
    OMEGA_LOG_RATIO_THRESHOLD = 0.0 # np.log(1 + OMEGA_RATIO_THRESHOLD) -> 0

    BASELINE_ANNUAL_LOG_RETURN = 0.05
    BASELINE_ANNUAL_LOG_RETURN_MS = BASELINE_ANNUAL_LOG_RETURN / (DAYS_IN_YEAR * 24 * 60 * 60 * 1000)
    SORTINO_MIN_DENOMINATOR = 1e-6
    OMEGA_MINIMUM_DENOMINATOR = 1e-6
    PROBABILISTIC_SHARPE_RATIO_MIN_STD_DEV = 1e-6
    MIN_STD_DEV = 1e-6
    MIN_MEDIAN = 1e-6
    HISTORICAL_GAIN_LOSS_COEFFICIENT = 1.0
    HISTORICAL_DECAY_GAIN_COEFFICIENT = 1.0
    HISTORICAL_DECAY_LOSS_COEFFICIENT = 1.0
    HISTORICAL_DECAY_COEFFICIENT_RETURNS_SHORT = 0.18
    HISTORICAL_DECAY_COEFFICIENT_RETURNS_LONG = 3
    HISTORICAL_DECAY_COEFFICIENT_RISKMETRIC = 2.5

    SET_WEIGHT_MINIMUM_SINGLE_CHECKPOINT_DURATION_MINUTES = 15
    SET_WEIGHT_MINIMUM_SINGLE_CHECKPOINT_DURATION_MS = SET_WEIGHT_MINIMUM_SINGLE_CHECKPOINT_DURATION_MINUTES * 60 * 1000
    SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS = 5 * 60 * 60 * 1000  # 5 hours
    SET_WEIGHT_MINIMUM_POSITIONS = 10 # mimumum number of positions over the lookback range
    SET_WEIGHT_MINIMUM_POSITION_DURATION_MS = 1 * 60 * 1000  # 1 minutes
    SET_WEIGHT_MINIMUM_POSITION_DURATION_TOTAL_MS = SET_WEIGHT_MINIMUM_POSITION_DURATION_MS * SET_WEIGHT_MINIMUM_POSITIONS
    
    SET_WEIGHT_MINER_CHECKPOINT_CONSISTENCY_DISPLACEMENT = 22.5
    SET_WEIGHT_CHECKPOINT_CONSISTENCY_TAPER = 0.45
    SET_WEIGHT_CHECKPOINT_CONSISTENCY_LOWER_BOUND = 0.0
    TOP_PERCENT_CONSISTENCY = 0.1
    SET_WEIGHT_MINER_GRACE_PERIOD_MS = 30 * 24 * 60 * 60 * 1000  # 30 days
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT = 5.4e-06 # essentially nothing
    SET_WEIGHT_MINER_GRACE_PERIOD_EQUIVALENT_PERCENTILE = 2 # two pence for their troubles
    MIN_LEVERAGE_CONSITENCY_PENALTY = 0.01
    CHECKPOINT_LENGTH_THRESHOLD = 70 # 2.5 weeks
    CHECKPOINT_DURATION_THRESHOLD = 1.25e9

    ## Scoring weights
    SCORING_RETURN_CPS_SHORT_WEIGHT = 0.95
    SCORING_RETURN_CPS_LONG_WEIGHT = 0.5
    SCORING_OMEGA_CPS_WEIGHT = 0.15
    SCORING_SORTINO_CPS_WEIGHT = 0.00
    TOP_MINERS_CUTOFF = 25

    ## MDD penalty calculation
    DRAWDOWN_NTERMS = 4 * 4 * 7 # four weeks
    DRAWDOWN_UPPER_SCALING = 3.5
    DRAWDOWN_MAXVALUE = 0.95
    DRAWDOWN_MINVALUE = 0.985    
    DRAWDOWN_MAXVALUE_PERCENTAGE = 5
    DRAWDOWN_MINVALUE_PERCENTAGE = 0.2

    ## Challenge period for setting weights, depreciated in favor of the new challenge period
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_SHARPE = 1.10
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_RETURN = 1.02
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_NRETURNS = 10
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_MRAD = 20

    ## Challenge period for setting weights with new checkpoint system
    SET_WEIGHT_CHALLENGE_PERIOD_MS = 30 * 24 * 60 * 60 * 1000  # 30 days
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_TOTAL_POSITION_DURATION = 0 * 60 * 60 * 1000  # 0 hours
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_SORTINO_CPS = -3.0e-8
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_OMEGA_CPS = 1.0015
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_RETURN_CPS_PERCENT = 1.02
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_TOTAL_ACTIVITY = 0.01
    CHECKPOINT_VOLUME_THRESHOLD = 0.1
    SET_WEIGHT_MINER_CHALLENGE_PERIOD_VOLUME_CHECKPOINTS = 12

    ORDER_SIMILARITY_WINDOW_MS = TimeUtil.hours_in_millis(24)
    MINER_COPYING_WEIGHT = 0.01
    BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

    METAGRAPH_UPDATE_REFRESH_TIME_MS = 60 * 1000  # 1 minute

    ELIMINATION_CHECK_INTERVAL_MS = 60 * 5 * 1000  # 5 minutes
    ELIMINATION_FILE_DELETION_DELAY_MS = 60 * 30 * 1000  # 30 min

    MAX_MINER_PLAGIARISM_SCORE = 0.9 # want to make sure we're filtering out the bad actors
    TOP_MINER_BENEFIT = 0.90
    TOP_MINER_PERCENT = 0.40