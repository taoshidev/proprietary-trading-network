import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig

class PriceSlippageModel:
    def __init__(self, running_unit_tests=False):
        # self.live_price_fetcher = live_price_fetcher
        self.parameters = self.read_slippage_model_parameters()
        self.sampled_df = None

        self.running_unit_tests = running_unit_tests

        # if self.pds is None:
        secrets = ValiUtils.get_secrets(running_unit_tests=self.running_unit_tests)
        live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
        self.pds = live_price_fetcher.polygon_data_service

    # TODO: uses model and parameters to calculate slippage percentage. in the validator we take the slippage percentage
    #  and multiply it by the price to get the avg execution price

    def calculate_slippage(self, trade_pair: TradePair, side: str, size: float, processed_ms: int) -> float:
        """
        returns the percentage slippage of the current order.
        each asset class uses a unique model
        """
        if trade_pair.is_equities:
            slippage_percentage = self.calc_slippage_equities(trade_pair, side, size, processed_ms)
        elif trade_pair.is_forex:
            slippage_percentage = self.calc_slippage_forex(trade_pair, size, processed_ms)
        elif trade_pair.is_crypto:
            slippage_percentage = self.calc_slippage_crypto(trade_pair)
        else:
            raise ValueError(f"Invalid trade pair {trade_pair.trade_pair_id} to calculate slippage")
        return slippage_percentage

    def calc_slippage_equities(self, trade_pair: TradePair, side: str, size: float, processed_ms: int) -> float:
        """
        Slippage percentage = intercept + c1 * spread/price + c2 * annualized_volatility + c3 * order_value/avg_daily_volume
        """
        # bucket size
        size_str = self.get_order_size_bucket(size)
        model_config = self.parameters["equity"][trade_pair.trade_pair_id][side][size_str]
        print(model_config)
        intercept, c1, c2, c3 = (model_config[key] for key in ["intercept", "spread/price", "annualized_vol", "buy_order_value/adv"])

        annualized_volatility, avg_daily_volume = self.get_bar_features(trade_pair, processed_ms)

        print(annualized_volatility, avg_daily_volume)
        ask, bid = self.pds.get_last_quote(trade_pair, processed_ms)
        print(ask, bid)
        spread = bid - ask
        mid_price = (bid + ask) / 2

        slippage_cost_percentage = intercept + (c1 * spread / mid_price) + (c2 * annualized_volatility) + (c3 * size / avg_daily_volume)
        return slippage_cost_percentage

    def get_bar_features(self, trade_pair: TradePair, processed_ms: int):
        order_date = self.ms_to_date_string(processed_ms)

        bars_df = self.get_bars_with_features(trade_pair, processed_ms)
        # print(bars_df)
        # order_date_dt = datetime.strptime(order_date, "%Y-%m-%d")
        # print(order_date)
        row_selected = bars_df[bars_df['datetime'] == order_date]
        print(row_selected)
        annualized_volatility = row_selected['annualized_vol'].values[0]
        avg_daily_volume = row_selected[f'adv_last_{ValiConfig.AVERAGE_DAILY_VOLUME_LOOKBACK_DAYS}_days'].values[0]

        return annualized_volatility, avg_daily_volume

    def calc_slippage_forex(self, trade_pair: TradePair, size: float, processed_ms: int) -> float:
        """
        Using the BB+ model as a stand-in for forex
        volume(standard lots)
        slippage percentage = 0.433 * spread/mid_price + 0.335 * sqrt( TODO
        """
        annualized_volatility, avg_daily_volume = self.get_bar_features(trade_pair, processed_ms)
        print(annualized_volatility, avg_daily_volume)
        ask, bid = self.pds.get_last_quote(trade_pair, processed_ms)
        print(ask, bid)
        spread = bid - ask
        mid_price = (bid + ask) / 2

        volume_standard_lots = size / 100_000  # TODO: convert the usd dollar amt to standard lots

        term1 = 0.433 * spread / mid_price
        term2 = 0.335 * math.sqrt(annualized_volatility ** 2 / 3 / 250)
        term3 = math.sqrt(volume_standard_lots / (0.3 * avg_daily_volume))
        slippage_pct = term1 + term2 * term3

        return slippage_pct

    def calc_slippage_crypto(self, trade_pair: TradePair) -> float:
        """
        slippage values for crypto
        """
        if trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD]:
            return 0.00001
        elif trade_pair in [TradePair.SOLUSD, TradePair.XRPUSD, TradePair.DOGEUSD]:
            return 0.0001
        else:
            raise ValueError(f"Unknown crypto slippage for trade pair {trade_pair.trade_pair_id}")

    def get_bars_with_features(self, trade_pair, processed_ms, adv_lookback_window=10, calc_vol_window=30, trading_days_in_a_year=252):
        """
        fetch data for average daily volume (estimated daily volume) and annualized volatility
        """
        order_date = self.ms_to_date_string(processed_ms)
        start_date = self.get_starting_trading_date(order_date, max(ValiConfig.AVERAGE_DAILY_VOLUME_LOOKBACK_DAYS,
                                                           ValiConfig.ANNUALIZED_VOLATILITY_LOOKBACK_DAYS) + 1)

        price_info_raw = self.pds.get_candles_for_trade_pair_simple(trade_pair, start_date, order_date, timespan="day")
        aggs = []
        try:
            for a in price_info_raw:
                aggs.append(a)
        except Exception as e:
            print(f"Error fetching data from Polygon: {e}")

        bars_pd = pd.DataFrame(aggs)
        bars_pd['datetime'] = pd.to_datetime(bars_pd['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
        # bars_pd['weekday'] = bars_pd['datetime'].dt.weekday
        # bars_pd = bars_pd[bars_pd['weekday'] < 5].reset_index(drop=True)  # filter out weekends
        # bars_pd.loc[:, 'day'] = bars_pd['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).date())
        bars_pd[f'adv_last_{adv_lookback_window}_days'] = (bars_pd['volume'].rolling(window=adv_lookback_window + 1).sum() - bars_pd['volume']) / adv_lookback_window  # excluding the current day when calculating adv
        bars_pd['daily_returns'] = np.log(bars_pd["close"] / bars_pd["close"].shift(1))
        # excluding the current day when calculating vol
        bars_pd[f'rolling_avg_daily_vol_{calc_vol_window}d'] = bars_pd['daily_returns'].rolling(window=calc_vol_window, closed='left').std()
        bars_pd[f"annualized_vol"] = bars_pd[f'rolling_avg_daily_vol_{calc_vol_window}d'] * np.sqrt(trading_days_in_a_year)

        return bars_pd

    def get_starting_trading_date(self, date_str: str, days_ago: int) -> str:
        """
        get the date n trading days ago
        excludes weekends and trading holidays
        """
        holidays = ["2025-01-01", "2025-01-09", "2025-01-20", "2024-12-25"] + self.pds.get_market_holidays()
        # print("holidays:", holidays)
        date_np = np.datetime64(date_str)
        past_date = np.busday_offset(date_np, -days_ago, roll='backward', holidays=holidays) # subtract trading days (weekends and holidays are skipped)
        return str(past_date)

    @staticmethod
    def ms_to_date_string(timestamp_ms: int, date_format: str = "%Y-%m-%d") -> str:
        """
        convert ms timestamp to a formatted date string YYYY-MM-DD
        """
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)  # convert to datetime
        return dt.strftime(date_format)  # convert to string

    @staticmethod
    def get_order_size_bucket(size: float) -> str:
        """
        bucket the order size float into a string for reading from model parameters file
        """
        all_order_value_ranges = [(1_000, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, 200_000), (200_000, 300_000)]
        order_value_labels = ["1k_10k", "10k_50k", "50k_100k", "100k_200k", "200k_300k"]

        for (low, high), label in zip(all_order_value_ranges, order_value_labels):
            if low <= size < high:
                return label

        # TODO: order size range
        if size > 300_000:
            raise ValueError("Order size must be less than $300K")
        else:
            # TODO <1k the slippage is 0
            raise ValueError("Order size must be at least $1K")

    @staticmethod
    def read_slippage_model_parameters() -> dict:
        equity_parameters = ValiUtils.get_vali_json_file_dict(ValiBkpUtils.get_equity_slippage_model_parameters_file())
        # print(equity_parameters)
        return equity_parameters


if __name__ == "__main__":
    psm = PriceSlippageModel()
    slippage = psm.calculate_slippage(TradePair.AAPL, "buy", 75000, 1737655200000)
    print(slippage)
