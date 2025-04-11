# developer: Taoshi
import os
import math

from enum import Enum

from meta import load_version

BASE_DIR = base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta_dict = load_version(os.path.join(base_directory, "meta", "meta.json"))
if meta_dict is None:
    #  Databricks
    print('Unable to load meta_dict. This is expected if running on Databricks.')
    meta_version = "x.x.x"
else:
    meta_version = meta_dict.get("subnet_version", "x.x.x")

class TradePairCategory(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    INDICES = "indices"
    EQUITIES = "equities"


class ValiConfig:
    # versioning
    VERSION = meta_version
    DAYS_IN_YEAR = 365  # annualization factor
    STATISTICAL_CONFIDENCE_MINIMUM_N = 60

    # Market-specific configurations
    ANNUAL_RISK_FREE_PERCENTAGE = 4.19  # From tbill rates
    ANNUAL_RISK_FREE_DECIMAL = ANNUAL_RISK_FREE_PERCENTAGE / 100
    DAILY_LOG_RISK_FREE_RATE = math.log(1 + ANNUAL_RISK_FREE_DECIMAL) / DAYS_IN_YEAR
    MS_RISK_FREE_RATE = math.log(1 + ANNUAL_RISK_FREE_PERCENTAGE / 100) / (365 * 24 * 60 * 60 * 1000)

    # Time Configurations
    TARGET_CHECKPOINT_DURATION_MS = 1000 * 60 * 60 * 12  # 12 hours
    DAILY_MS = 1000 * 60 * 60 * 24  # 1 day
    DAILY_CHECKPOINTS = DAILY_MS // TARGET_CHECKPOINT_DURATION_MS  # 2 checkpoints per day

    # Set the target ledger window in days directly
    TARGET_LEDGER_WINDOW_DAYS = 90  # 90 days
    TARGET_LEDGER_WINDOW_MS = DAILY_MS * TARGET_LEDGER_WINDOW_DAYS
    TARGET_LEDGER_N_CHECKPOINTS = TARGET_LEDGER_WINDOW_MS // TARGET_CHECKPOINT_DURATION_MS  # 180 checkpoints
    WEIGHTED_AVERAGE_DECAY_RATE = 0.08
    WEIGHTED_AVERAGE_DECAY_MIN = 0.40
    WEIGHTED_AVERAGE_DECAY_MAX = 1.0
    POSITIONAL_EQUIVALENCE_WINDOW_MS = 1000 * 60 * 60 * 24  # 1 day

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
    EQUITIES_MIN_LEVERAGE = 0.1
    EQUITIES_MAX_LEVERAGE = 3

    CAPITAL = 100_000  # conversion of 1x leverage to $100K in capital

    MAX_DAILY_DRAWDOWN = 0.95  # Portfolio should never fall below .95 x of initial value when measured day to day
    MAX_TOTAL_DRAWDOWN = 0.9  # Portfolio should never fall below .90 x of initial value when measured at any instant
    MAX_TOTAL_DRAWDOWN_V2 = 0.95
    MAX_OPEN_ORDERS_PER_HOTKEY = 200
    ORDER_COOLDOWN_MS = 10000  # 10 seconds
    ORDER_MIN_LEVERAGE = 0.001
    ORDER_MAX_LEVERAGE = 500

    # Risk Profiling
    RISK_PROFILING_STEPS_MIN_LEVERAGE = min(CRYPTO_MIN_LEVERAGE, FOREX_MIN_LEVERAGE, INDICES_MIN_LEVERAGE, EQUITIES_MIN_LEVERAGE)
    RISK_PROFILING_STEPS_CRITERIA = 3
    RISK_PROFILING_MONOTONIC_CRITERIA = 2
    RISK_PROFILING_MARGIN_CRITERIA = 0.5
    RISK_PROFILING_LEVERAGE_ADVANCE = 1.5
    RISK_PROFILING_SCOPING_MECHANIC = 100
    RISK_PROFILING_SIGMOID_SHIFT = 1.2
    RISK_PROFILING_SIGMOID_SPREAD = 4
    # RISK_PROFILING_TIME_DECAY = 5
    # RISK_PROFILING_TIME_CYCLE = POSITIONAL_EQUIVALENCE_WINDOW_MS
    RISK_PROFILING_TIME_CRITERIA = 0.185  # threshold for the normalized error of a position’s order time intervals

    PLAGIARISM_MATCHING_TIME_RESOLUTION_MS = 60 * 1000 * 2  # 2 minutes
    PLAGIARISM_MAX_LAGS = 60
    PLAGIARISM_LOOKBACK_RANGE_MS = 10 * 24 * 60 * 60 * 1000  # 10 days
    PLAGIARISM_FOLLOWER_TIMELAG_THRESHOLD = 1.0005
    PLAGIARISM_FOLLOWER_SIMILARITY_THRESHOLD = 0.75
    PLAGIARISM_REPORTING_THRESHOLD = 0.8
    PLAGIARISM_REFRESH_TIME_MS = 1000 * 60 * 60 * 24 # 1 day
    PLAGIARISM_ORDER_TIME_WINDOW_MS = 1000 * 60 * 60 * 12
    PLAGIARISM_MINIMUM_FOLLOW_MS = 1000 * 10 # Minimum follow time of 10 seconds for each order

    EPSILON = 1e-6
    RETURN_SHORT_LOOKBACK_TIME_MS = 5 * 24 * 60 * 60 * 1000  # 5 days
    RETURN_SHORT_LOOKBACK_LEDGER_WINDOWS = RETURN_SHORT_LOOKBACK_TIME_MS // TARGET_CHECKPOINT_DURATION_MS


    MINIMUM_POSITION_DURATION_MS = 1 * 60 * 1000  # 1 minutes

    SHORT_LOOKBACK_WINDOW = 7 * DAILY_CHECKPOINTS

    # Scoring weights
    SCORING_OMEGA_WEIGHT = 0.2
    SCORING_SHARPE_WEIGHT = 0.2
    SCORING_SORTINO_WEIGHT = 0.2
    SCORING_STATISTICAL_CONFIDENCE_WEIGHT = 0.2
    SCORING_CALMAR_WEIGHT = 0.2
    SCORING_RETURN_WEIGHT = 0.0

    # Scoring hyperparameters
    OMEGA_LOSS_MINIMUM = 0.01   # Equivalent to 1% loss
    OMEGA_NOCONFIDENCE_VALUE = 0.0
    SHARPE_STDDEV_MINIMUM = 0.01  # Equivalent to 1% standard deviation
    SHARPE_NOCONFIDENCE_VALUE = -100
    SORTINO_DOWNSIDE_MINIMUM = 0.01  # Equivalent to 1% standard deviation
    SORTINO_NOCONFIDENCE_VALUE = -100
    STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE = -100
    CALMAR_NOCONFIDENCE_VALUE = -100

    # MDD penalty calculation
    APPROXIMATE_DRAWDOWN_PERCENTILE = 0.75
    DRAWDOWN_UPPER_SCALING = 5
    DRAWDOWN_MAXVALUE_PERCENTAGE = 10
    DRAWDOWN_MINVALUE_PERCENTAGE = 0.5

    # Challenge period
    CHALLENGE_PERIOD_WEIGHT = 1.2e-05  # essentially nothing
    CHALLENGE_PERIOD_MS = 90 * 24 * 60 * 60 * 1000  # 90 days
    CHALLENGE_PERIOD_PERCENTILE_THRESHOLD = 0.75 # miners must pass 75th percentile to enter the main competition

    # Plagiarism
    ORDER_SIMILARITY_WINDOW_MS = 60000 * 60 * 24
    MINER_COPYING_WEIGHT = 0.01
    MAX_MINER_PLAGIARISM_SCORE = 0.9  # want to make sure we're filtering out the bad actors

    BASE_DIR = base_directory = BASE_DIR

    METAGRAPH_UPDATE_REFRESH_TIME_MS = 60 * 1000  # 1 minute
    ELIMINATION_CHECK_INTERVAL_MS = 60 * 5 * 1000  # 5 minutes
    ELIMINATION_FILE_DELETION_DELAY_MS = 60 * 30 * 1000  # 30 min

    # Distributional statistics
    SOFTMAX_TEMPERATURE = 0.125

    # Qualifications to be a trusted validator sending checkpoints
    TOP_N_CHECKPOINTS = 10
    TOP_N_STAKE = 20
    STAKE_MIN = 1000.0
    AXON_NO_IP = "0.0.0.0"

    # Require at least this many successful checkpoints before building golden
    MIN_CHECKPOINTS_RECEIVED = 5

    # Cap leverage across miner's entire portfolio
    PORTFOLIO_LEVERAGE_CAP = 10

assert ValiConfig.CRYPTO_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.CRYPTO_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.FOREX_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.FOREX_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.INDICES_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.INDICES_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE
assert ValiConfig.EQUITIES_MIN_LEVERAGE >= ValiConfig.ORDER_MIN_LEVERAGE
assert ValiConfig.EQUITIES_MAX_LEVERAGE <= ValiConfig.ORDER_MAX_LEVERAGE

class TradePair(Enum):
    # crypto
    BTCUSD = ["BTCUSD", "BTC/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO]
    ETHUSD = ["ETHUSD", "ETH/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO]
    SOLUSD = ["SOLUSD", "SOL/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
              TradePairCategory.CRYPTO]
    XRPUSD = ["XRPUSD", "XRP/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
                TradePairCategory.CRYPTO]
    DOGEUSD = ["DOGEUSD", "DOGE/USD", 0.001, ValiConfig.CRYPTO_MIN_LEVERAGE, ValiConfig.CRYPTO_MAX_LEVERAGE,
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
    GBPAUD = ["GBPAUD", "GBP/AUD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
                TradePairCategory.FOREX]
    GBPCAD = ["GBPCAD", "GBP/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
                TradePairCategory.FOREX]
    GBPJPY = ["GBPJPY", "GBP/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    GBPUSD = ["GBPUSD", "GBP/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDCAD = ["USDCAD", "USD/CAD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDCHF = ["USDCHF", "USD/CHF", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDJPY = ["USDJPY", "USD/JPY", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]
    USDMXN = ["USDMXN", "USD/MXN", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE,
              TradePairCategory.FOREX]

    # "Commodities" (Bundle with Forex for now)
    XAUUSD = ["XAUUSD", "XAU/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE, TradePairCategory.FOREX]
    XAGUSD = ["XAGUSD", "XAG/USD", 0.00007, ValiConfig.FOREX_MIN_LEVERAGE, ValiConfig.FOREX_MAX_LEVERAGE, TradePairCategory.FOREX]

    # Equities
    NVDA = ["NVDA", "NVDA", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    AAPL = ["AAPL", "AAPL", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    TSLA = ["TSLA", "TSLA", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    AMZN = ["AMZN", "AMZN", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    MSFT = ["MSFT", "MSFT", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    GOOG = ["GOOG", "GOOG", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]
    META = ["META", "META", 0.00009, ValiConfig.EQUITIES_MIN_LEVERAGE, ValiConfig.EQUITIES_MAX_LEVERAGE, TradePairCategory.EQUITIES]


    # indices (no longer allowed for trading as we moved to equities tickers instead)
    SPX = ["SPX", "SPX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    DJI = ["DJI", "DJI", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    NDX = ["NDX", "NDX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
    VIX = ["VIX", "VIX", 0.00009, ValiConfig.INDICES_MIN_LEVERAGE, ValiConfig.INDICES_MAX_LEVERAGE,
           TradePairCategory.INDICES]
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
    def is_equities(self):
        return self.trade_pair_category == TradePairCategory.EQUITIES
    @property
    def is_indices(self):
        return self.trade_pair_category == TradePairCategory.INDICES

    @property
    def leverage_multiplier(self) -> int:
        trade_pair_leverage_multiplier = {TradePairCategory.CRYPTO: 10,
                                          TradePairCategory.FOREX: 1,
                                          TradePairCategory.INDICES: 1,
                                          TradePairCategory.EQUITIES: 2}
        return trade_pair_leverage_multiplier[self.trade_pair_category]

    @classmethod
    def categories(cls):
        return {tp.trade_pair_id: tp.trade_pair_category.value for tp in cls}

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
