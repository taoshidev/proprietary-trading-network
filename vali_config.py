# developer: Taoshi
import os
import math
from time_util.time_util import TimeUtil

from enum import Enum

from meta import load_version
BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))
meta_dict = load_version(os.path.join(base_directory, "meta", "meta.json"))
meta_version = meta_dict.get("subnet_version", "x.x.x")

class TradePairCategory(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    INDICES = "indices"


class ValiConfig:
    # versioning
    VERSION = meta_version

    # Time Configurations
    TARGET_CHECKPOINT_DURATION_MS = 1000 * 60 * 60 * 12  # 12 hours
    TARGET_LEDGER_WINDOW_MS = 1000 * 60 * 60 * 24 * 90  # 90 days
    TARGET_LEDGER_N_CHECKPOINTS = TARGET_LEDGER_WINDOW_MS // TARGET_CHECKPOINT_DURATION_MS

    SET_WEIGHT_REFRESH_TIME_MS = 60 * 5 * 1000  # 5 minutes
    SET_WEIGHT_LOOKBACK_RANGE_DAYS = int(TARGET_LEDGER_WINDOW_MS / (24 * 60 * 60 * 1000))

    # Fees take into account exiting and entering a position, liquidity, and futures fees
    PERF_LEDGER_REFRESH_TIME_MS = 1000 * 60 * 5  # minutes
    CHALLENGE_PERIOD_REFRESH_TIME_MS = 1000 * 60 * 1  # minutes
    MDD_CHECK_REFRESH_TIME_MS = 60 * 1000  # 60 seconds

    # Positional Leverage limits
    CRYPTO_MIN_LEVERAGE = 0.01
    CRYPTO_MAX_LEVERAGE = 0.5
    FOREX_MIN_LEVERAGE = 0.1
    FOREX_MAX_LEVERAGE = 5
    INDICES_MIN_LEVERAGE = 0.1
    INDICES_MAX_LEVERAGE = 5

    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_TOTAL_DRAWDOWN_V2 = 0.95
    MAX_OPEN_ORDERS_PER_HOTKEY = 200
    ORDER_COOLDOWN_MS = 10000  # 10 seconds
    ORDER_MIN_LEVERAGE = 0.001
    ORDER_MAX_LEVERAGE = 500

    PLAGIARISM_MATCHING_TIME_RESOLUTION_MS = 1 * 60 * 1000  # 1 minute
    PLAGIARISM_THRESHOLD = 0.9
    PLAGIARISM_MAX_LAGS = 60
    PLAGIARISM_LOOKBACK_RANGE_MS = 5 * 24 * 60 * 60 * 1000  # 5 days
    PLAGIARISM_FOLLOWER_TIMELAG_THRESHOLD = 1.005
    PLAGIARISM_FOLLOWER_SIMILARITY_THRESHOLD = 0.8

    HISTORICAL_DECAY_TIME_INTENSITY_COEFFICIENT = 0.18
    DAYS_IN_YEAR = 365.25

    EPSILON = 1e-6
    RETURN_SHORT_LOOKBACK_TIME_MS = 5 * 24 * 60 * 60 * 1000  # 5 days
    RETURN_SHORT_LOOKBACK_LEDGER_WINDOWS = RETURN_SHORT_LOOKBACK_TIME_MS // TARGET_CHECKPOINT_DURATION_MS

    # Consistency spreads for max positional return
    MAX_RETURN_SIGMOID_SHIFT = 0.25
    MAX_RETURN_SIGMOID_SPREAD = 50

    # Consistency spreads for positional metrics
    POSITIONAL_RETURN_TIME_SIGMOID_SPREAD = 50
    POSITIONAL_RETURN_TIME_SIGMOID_SHIFT = 0.45
    POSITIONAL_RETURN_TIME_WINDOW_MS = 1000 * 60 * 60 * 24 * 7  # 7 days

    SET_WEIGHT_MINER_CHECKPOINT_CONSISTENCY_DISPLACEMENT = 15
    SET_WEIGHT_CHECKPOINT_CONSISTENCY_TAPER = SET_WEIGHT_MINER_CHECKPOINT_CONSISTENCY_DISPLACEMENT * 1.25
    SET_WEIGHT_CHECKPOINT_CONSISTENCY_LOWER_BOUND = 0.0
    SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS = 5 * 60 * 60 * 1000  # 5 hours

    # Consistency windowing
    DAILY_MS = 1000 * 60 * 60 * 24  # 1 day
    DAILY_CHECKPOINTS = DAILY_MS // TARGET_CHECKPOINT_DURATION_MS
    DAILY_SIGMOID_SHIFT = 0.35
    DAILY_SIGMOID_SPREAD = 35

    BIWEEKLY_MS = 1000 * 60 * 60 * 24 * 14  # two weeks
    BIWEEKLY_CHECKPOINTS = BIWEEKLY_MS // TARGET_CHECKPOINT_DURATION_MS
    BIWEEKLY_SIGMOID_SHIFT = 0.5
    BIWEEKLY_SIGMOID_SPREAD = 35

    MINIMUM_POSITION_DURATION_MS = 1 * 60 * 1000  # 1 minutes

    CHECKPOINT_LENGTH_RATIO = 0.625
    CHECKPOINT_LENGTH_THRESHOLD = int(
        (TARGET_LEDGER_WINDOW_MS // TARGET_CHECKPOINT_DURATION_MS) * CHECKPOINT_LENGTH_RATIO)

    CHECKPOINT_DURATION_RATIO = 0.5
    CHECKPOINT_DURATION_THRESHOLD = int(TARGET_LEDGER_WINDOW_MS * CHECKPOINT_DURATION_RATIO)

    # Scoring weights
    SCORING_RETURN_LONG_LOOKBACK_WEIGHT = 1.0
    SCORING_RETURN_SHORT_LOOKBACK_WEIGHT = 0.25
    SCORING_SHARPE_WEIGHT = 0.25
    SCORING_OMEGA_WEIGHT = 0.25

    # Scoring hyperparameters
    OMEGA_LOSS_MINIMUM = 0.02  # Equivalent to 2% loss
    SHARPE_STDDEV_MINIMUM = 0.01  # Equivalent to 1% standard deviation

    # MDD penalty calculation
    APPROXIMATE_DRAWDOWN_PERCENTILE = 0.90
    DRAWDOWN_UPPER_SCALING = 5
    DRAWDOWN_MAXVALUE_PERCENTAGE = 10
    DRAWDOWN_MINVALUE_PERCENTAGE = 0.5

    # Challenge period
    CHALLENGE_PERIOD_WEIGHT = 5.4e-06  # essentially nothing
    CHALLENGE_PERIOD_MS = 60 * 24 * 60 * 60 * 1000  # 60 days
    CHALLENGE_PERIOD_RETURN = 1.02
    CHALLENGE_PERIOD_RETURN_LOG = math.log(CHALLENGE_PERIOD_RETURN)
    CHALLENGE_PERIOD_RETURN_PERCENTAGE = (CHALLENGE_PERIOD_RETURN - 1) * 100
    CHALLENGE_PERIOD_MAX_POSITIONAL_RETURNS_RATIO = 0.25  # one position shouldn't be more than 25% of the total returns
    CHALLENGE_PERIOD_MAX_UNREALIZED_RETURNS_RATIO = 0.20  # one day shouldn't be more than 20% of the total returns
    CHALLENGE_PERIOD_MAX_DRAWDOWN_PERCENT = 5  # beyond 5% drawdown, they don't pass challenge
    CHALLENGE_PERIOD_MIN_POSITIONS = 10  # need at least 10 positions to pass challenge

    # Plagiarism
    ORDER_SIMILARITY_WINDOW_MS = TimeUtil.hours_in_millis(24)
    MINER_COPYING_WEIGHT = 0.01
    MAX_MINER_PLAGIARISM_SCORE = 0.9  # want to make sure we're filtering out the bad actors

    BASE_DIR = base_directory = BASE_DIR

    METAGRAPH_UPDATE_REFRESH_TIME_MS = 60 * 1000  # 1 minute
    ELIMINATION_CHECK_INTERVAL_MS = 60 * 5 * 1000  # 5 minutes
    ELIMINATION_FILE_DELETION_DELAY_MS = 60 * 30 * 1000  # 30 min

    # Distributional statistics
    TOP_MINER_BENEFIT = 0.90
    TOP_MINER_PERCENT = 0.40
    # SOFTMAX_TEMPERATURE = 0.25

    # Qualifications to be a trusted validator sending checkpoints
    TOP_N_CHECKPOINTS = 10
    TOP_N_STAKE = 20
    STAKE_MIN = 1000.0
    AXON_NO_IP = "0.0.0.0"

    # Require at least this many successful checkpoints before building golden
    MIN_CHECKPOINTS_RECEIVED = 5


assert ValiConfig.CRYPTO_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.CRYPTO_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.FOREX_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.FOREX_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.INDICES_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.INDICES_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE


class TradePair(Enum):
    # crypto
    BTCUSD = ["BTCUSD", "BTC/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO]
    ETHUSD = ["ETHUSD", "ETH/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO]

    # forex
    AUDCAD = ["AUDCAD", "AUD/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    AUDUSD = ["AUDUSD", "AUD/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    AUDJPY = ["AUDJPY", "AUD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    AUDNZD = ["AUDNZD", "AUD/NZD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    CADCHF = ["CADCHF", "CAD/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    CADJPY = ["CADJPY", "CAD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    CHFJPY = ["CHFJPY", "CHF/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    EURCAD = ["EURCAD", "EUR/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    EURUSD = ["EURUSD", "EUR/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    EURCHF = ["EURCHF", "EUR/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    EURGBP = ["EURGBP", "EUR/GBP", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    EURJPY = ["EURJPY", "EUR/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    EURNZD = ["EURNZD", "EUR/NZD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    NZDCAD = ["NZDCAD", "NZD/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    NZDJPY = ["NZDJPY", "NZD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    NZDUSD = ["NZDUSD", "NZD/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDMXN = ["USDMXN", "USD/MXN", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]

    # indices
    SPX = ["SPX", "SPX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    DJI = ["DJI", "DJI", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    NDX = ["NDX", "NDX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    VIX = ["VIX", "VIX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]

    # Trade pairs not available for trading due to transitioning to Polygon.io which does not cutrrently support them.
    FTSE = ["FTSE", "FTSE", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
            TradePairCategory.INDICES]
    GDAXI = ["GDAXI", "GDAXI", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
             TradePairCategory.INDICES]

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
