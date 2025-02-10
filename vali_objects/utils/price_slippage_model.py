import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class PriceSlippageModel:
    # tp_to_adv = defaultdict()
    # tp_to_vol = defaultdict()
    # parameters = read_slippage_model_parameters

    def __init__(self, live_price_fetcher=None, running_unit_tests=False, metagraph=None):
        self.running_unit_tests = running_unit_tests
        self.last_day_refreshed = -1
        self.tp_to_adv = defaultdict()
        self.tp_to_vol = defaultdict()
        self.parameters = self.read_slippage_model_parameters()

        if live_price_fetcher is None:
            secrets = ValiUtils.get_secrets(running_unit_tests=self.running_unit_tests)
            live_price_fetcher = LivePriceFetcher(secrets, disable_ws=False)
        self.pds = live_price_fetcher.polygon_data_service

    def calculate_slippage(self, bid:float, ask:float, order:Order):
        """
        returns the percentage slippage of the current order.
        each asset class uses a unique model
        """
        trade_pair = order.trade_pair
        size = order.leverage * ValiConfig.LEVERAGE_TO_CAPITAL

        if size < 1000:
            return 0  # assume 0 slippage when order size is under 1k

        if trade_pair.is_equities:
            slippage_percentage = self.calc_slippage_equities(bid, ask, order)
        elif trade_pair.is_forex:
            slippage_percentage = self.calc_slippage_forex(bid, ask, order)
        elif trade_pair.is_crypto:
            slippage_percentage = self.calc_slippage_crypto(order)
        else:
            raise ValueError(f"Invalid trade pair {trade_pair.trade_pair_id} to calculate slippage")
        return slippage_percentage

    def calc_slippage_equities(self, bid:float, ask:float, order:Order) -> float:
        """
        Slippage percentage = intercept + c1 * spread/price + c2 * annualized_volatility + c3 * order_value/avg_daily_volume
        """
        # bucket size
        size = order.leverage * ValiConfig.LEVERAGE_TO_CAPITAL
        size_str = self.get_order_size_bucket(size)
        model_config = self.parameters["equity"][order.trade_pair.trade_pair_id][order.side][size_str]
        intercept, c1, c2, c3 = (model_config[key] for key in ["intercept", "spread/price", "annualized_vol", f"{order.side}_order_value/adv"])

        annualized_volatility = self.tp_to_vol[order.trade_pair.trade_pair_id]
        avg_daily_volume = self.tp_to_vol[order.trade_pair.trade_pair_id]

        spread = bid - ask
        mid_price = (bid + ask) / 2

        slippage_pct = intercept + (c1 * spread / mid_price) + (c2 * annualized_volatility) + (c3 * size / avg_daily_volume)
        return abs(slippage_pct)

    def calc_slippage_forex(self, bid:float, ask:float, order:Order) -> float:
        """
        Using the BB+ model as a stand-in for forex
        slippage percentage = 0.433 * spread/mid_price + 0.335 * sqrt(annualized_volatility**2 / 3 / 250) * sqrt(volume / (0.3 * estimated daily volume))
        """
        annualized_volatility = self.tp_to_vol[order.trade_pair.trade_pair_id]
        avg_daily_volume = self.tp_to_vol[order.trade_pair.trade_pair_id]
        spread = bid - ask
        mid_price = (bid + ask) / 2

        size = order.leverage * ValiConfig.LEVERAGE_TO_CAPITAL
        base, _ = order.trade_pair.trade_pair.split("/")
        base_to_usd_conversion = self.pds.get_currency_conversion(base=base, quote="USD") if base != "USD" else 1  # TODO: fallback?
        # print(base_to_usd_conversion)
        volume_standard_lots = size / (100_000 * base_to_usd_conversion)  # Volume expressed in terms of standard lots (1 std lot = 100,000 base currency)

        term1 = 0.433 * spread / mid_price
        term2 = 0.335 * math.sqrt(annualized_volatility ** 2 / 3 / 250)
        term3 = math.sqrt(volume_standard_lots / (0.3 * avg_daily_volume))
        slippage_pct = term1 + term2 * term3
        return slippage_pct

    def calc_slippage_crypto(self, order:Order) -> float:
        """
        slippage values for crypto
        """
        trade_pair = order.trade_pair
        if trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD]:
            return 0.00001
        elif trade_pair in [TradePair.SOLUSD, TradePair.XRPUSD, TradePair.DOGEUSD]:
            return 0.0001
        else:
            raise ValueError(f"Unknown crypto slippage for trade pair {trade_pair.trade_pair_id}")

    def refresh_features_daily(self):
        """
        update the values for average daily volume and annualized volatility for all trade pairs once a day
        """
        current_day = datetime.utcnow().weekday()
        if current_day != self.last_day_refreshed:
            bt.logging.info(
                f"Refreshing avg daily volume and annualized volatility for new day UTC {datetime.utcnow().date()}")
            trade_pairs = [tp for tp in TradePair if tp.is_forex or tp.is_equities]
            self.calculate_features(trade_pairs=trade_pairs, processed_ms=TimeUtil.now_in_millis())
            self.write_features_from_memory_to_disk()
            self.last_day_refreshed = current_day
            bt.logging.info(
                f"Completed refreshing avg daily volume and annualized volatility for new day UTC {datetime.utcnow().date()}")

    def calculate_features(self, trade_pairs: list[TradePair], processed_ms: int, adv_lookback_window: int = 10,
                           calc_vol_window: int = 30):
        for trade_pair in trade_pairs:
            bars_df = self.get_bars_with_features(trade_pair, processed_ms, adv_lookback_window, calc_vol_window)
            row_selected = bars_df.iloc[-1]
            annualized_volatility = row_selected['annualized_vol']
            avg_daily_volume = row_selected[f'adv_last_{adv_lookback_window}_days']

            self.tp_to_vol[trade_pair.trade_pair_id] = annualized_volatility
            self.tp_to_adv[trade_pair.trade_pair_id] = avg_daily_volume

    def get_bars_with_features(self, trade_pair: TradePair, processed_ms: int, adv_lookback_window: int=10, calc_vol_window: int=30, trading_days_in_a_year: int=252) -> pd.DataFrame:
        """
        fetch data for average daily volume (estimated daily volume) and annualized volatility
        """
        order_date = TimeUtil.millis_to_short_date_str(processed_ms)
        start_date = TimeUtil.get_n_business_days_ago(order_date, max(adv_lookback_window+1, calc_vol_window+2))

        price_info_raw = self.pds.unified_candle_fetcher(trade_pair, start_date, order_date, timespan="day")
        aggs = []
        try:
            for a in price_info_raw:
                aggs.append(a)
        except Exception as e:
            print(f"Error fetching data from Polygon: {e}")

        bars_pd = pd.DataFrame(aggs)
        bars_pd['datetime'] = pd.to_datetime(bars_pd['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
        bars_pd[f'adv_last_{adv_lookback_window}_days'] = (bars_pd['volume'].rolling(window=adv_lookback_window + 1).sum() - bars_pd['volume']) / adv_lookback_window  # excluding the current day when calculating adv
        bars_pd['daily_returns'] = np.log(bars_pd["close"] / bars_pd["close"].shift(1))
        # excluding the current day when calculating vol,
        bars_pd[f'rolling_avg_daily_vol_{calc_vol_window}d'] = bars_pd['daily_returns'].rolling(window=calc_vol_window, closed='left').std()  # requires +2 days
        bars_pd["annualized_vol"] = bars_pd[f'rolling_avg_daily_vol_{calc_vol_window}d'] * np.sqrt(trading_days_in_a_year)
        return bars_pd

    def write_features_from_memory_to_disk(self):
        features = {
            "adv": self.tp_to_adv,
            "vol": self.tp_to_vol,
            "timestamp": TimeUtil.now_in_millis()
        }
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_slippage_model_features_file(),
            features
        )

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
            raise ValueError("Order size must be at least $1K")

    @staticmethod
    def read_slippage_model_parameters() -> dict:
        equity_parameters = ValiUtils.get_vali_json_file_dict(ValiBkpUtils.get_slippage_model_parameters_file())
        # print(equity_parameters)
        return equity_parameters


if __name__ == "__main__":
    psm = PriceSlippageModel()
    slippage = psm.calculate_slippage(TradePair.AAPL, "buy", 75000, 1737655200000)
    print(slippage)
