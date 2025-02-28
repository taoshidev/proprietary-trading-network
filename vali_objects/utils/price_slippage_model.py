import math
from collections import defaultdict
from datetime import datetime

import holidays
import numpy as np
import pandas as pd
import bittensor as bt

from data_generator.polygon_data_service import PolygonDataService
from time_util.time_util import TimeUtil
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class PriceSlippageModel:
    features = defaultdict(dict)
    parameters: dict = {}
    pds: PolygonDataService = None
    holidays_nyse = None
    is_backtesting = False

    def __init__(self, live_price_fetcher=None, running_unit_tests=False, is_backtesting=False):
        if not PriceSlippageModel.parameters:
            PriceSlippageModel.holidays_nyse = holidays.financial_holidays('NYSE')
            PriceSlippageModel.parameters = self.read_slippage_model_parameters()

            if live_price_fetcher is None:
                secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
                live_price_fetcher = LivePriceFetcher(secrets, disable_ws=False)
            PriceSlippageModel.pds = live_price_fetcher.polygon_data_service
        PriceSlippageModel.is_backtesting = is_backtesting

    @classmethod
    def calculate_slippage(cls, bid:float, ask:float, order:Order, leverage_to_capital=ValiConfig.LEVERAGE_TO_CAPITAL):
        """
        returns the percentage slippage of the current order.
        each asset class uses a unique model
        """
        trade_pair = order.trade_pair
        size = abs(order.leverage) * leverage_to_capital
        if size <= 1000:
            return 0  # assume 0 slippage when order size is under 1k
        if cls.is_backtesting:
            cls.refresh_features_daily(order.processed_ms, write_to_disk=False)
        else:
            cls.refresh_features_daily(order.processed_ms, write_to_disk=True)

        if trade_pair.is_equities:
            slippage_percentage = cls.calc_slippage_equities(bid, ask, order)
        elif trade_pair.is_forex:
            slippage_percentage = cls.calc_slippage_forex(bid, ask, order)
        elif trade_pair.is_crypto:
            slippage_percentage = cls.calc_slippage_crypto(order)
        else:
            raise ValueError(f"Invalid trade pair {trade_pair.trade_pair_id} to calculate slippage")
        return np.clip(slippage_percentage, 0.0, 0.03)

    @classmethod
    def calc_slippage_equities(cls, bid:float, ask:float, order:Order) -> float:
        """
        Fitted BB+ model (dec 2024)
        Slippage percentage = intercept + c1 * spread/price + c2 * annualized_volatility + c3 * order_volume/avg_daily_volume

        Use the direct BB+ model for pre dec 2024 orders
        slippage percentage = 0.433 * spread/mid_price + 0.335 * sqrt(annualized_volatility**2 / 3 / 250) * sqrt(volume / (0.3 * estimated daily volume))
        """
        order_date = TimeUtil.millis_to_short_date_str(order.processed_ms)
        annualized_volatility = cls.features[order_date]["vol"][order.trade_pair.trade_pair_id]
        avg_daily_volume = cls.features[order_date]["adv"][order.trade_pair.trade_pair_id]
        spread = ask - bid
        mid_price = (bid + ask) / 2

        size = abs(order.leverage) * ValiConfig.LEVERAGE_TO_CAPITAL
        volume_shares = size / mid_price

        if order.processed_ms > 1733040000000:  # Use fitted BB+ for orders after dec 1, 2024, 08:00:00 UTC
            size_str = cls.get_order_size_bucket(size)
            side = "buy" if order.leverage > 0 else "sell"
            model_config = cls.parameters["equity"][order.trade_pair.trade_pair_id][side][size_str]
            intercept, c1, c2, c3 = (model_config[key] for key in ["intercept", "spread/price", "annualized_vol", f"{side}_order_size/adv"])
            slippage_pct = intercept + (c1 * spread / mid_price) + (c2 * annualized_volatility) + (c3 * volume_shares / avg_daily_volume)
        else:
            # direct BB+ model for orders before
            term1 = 0.433 * spread / mid_price
            term2 = 0.335 * math.sqrt(annualized_volatility ** 2 / 3 / 250)
            term3 = math.sqrt(volume_shares / (0.3 * avg_daily_volume))
            slippage_pct = term1 + term2 * term3
        return slippage_pct

    @classmethod
    def calc_slippage_forex(cls, bid:float, ask:float, order:Order) -> float:
        """
        Using the direct BB+ model as a stand-in for forex
        slippage percentage = 0.433 * spread/mid_price + 0.335 * sqrt(annualized_volatility**2 / 3 / 250) * sqrt(volume / (0.3 * estimated daily volume))
        """
        order_date = TimeUtil.millis_to_short_date_str(order.processed_ms)
        annualized_volatility = cls.features[order_date]["vol"][order.trade_pair.trade_pair_id]
        avg_daily_volume = cls.features[order_date]["adv"][order.trade_pair.trade_pair_id]
        spread = ask - bid
        mid_price = (bid + ask) / 2

        # bt.logging.info(f"bid: {bid}, ask: {ask}, adv: {avg_daily_volume}, vol: {annualized_volatility}")

        size = abs(order.leverage) * ValiConfig.LEVERAGE_TO_CAPITAL
        base, _ = order.trade_pair.trade_pair.split("/")
        base_to_usd_conversion = cls.pds.get_currency_conversion(base=base, quote="USD") if base != "USD" else 1  # TODO: fallback?
        # print(base_to_usd_conversion)
        volume_standard_lots = size / (100_000 * base_to_usd_conversion)  # Volume expressed in terms of standard lots (1 std lot = 100,000 base currency)

        term1 = 0.433 * spread / mid_price
        term2 = 0.335 * math.sqrt(annualized_volatility ** 2 / 3 / 250)
        term3 = math.sqrt(volume_standard_lots / (0.3 * avg_daily_volume))
        slippage_pct = term1 + term2 * term3
        # bt.logging.info(f"slippage_pct: {slippage_pct}")
        return slippage_pct

    @classmethod
    def calc_slippage_crypto(cls, order:Order) -> float:
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

    @classmethod
    def refresh_features_daily(cls, time_ms:int=None, write_to_disk:bool=True):
        """
        Calculate and store model features (average daily volume and annualized volatility) for new days
        """
        if not time_ms:
            time_ms = TimeUtil.now_in_millis()
        current_date = TimeUtil.millis_to_short_date_str(time_ms)

        if current_date not in cls.features:
            bt.logging.info(
                f"Calculating avg daily volume and annualized volatility for new day UTC {datetime.utcnow().date()}")
            trade_pairs = [tp for tp in TradePair if tp.is_forex or tp.is_equities]
            tp_to_adv, tp_to_vol = cls.get_features(trade_pairs=trade_pairs, processed_ms=time_ms)
            cls.features[current_date]["adv"] = tp_to_adv
            cls.features[current_date]["vol"] = tp_to_vol
            if write_to_disk:
                cls.write_features_from_memory_to_disk()
            bt.logging.info(
                    f"Completed refreshing avg daily volume and annualized volatility for new day UTC {datetime.utcnow().date()}")

    @classmethod
    def get_features(cls, trade_pairs: list[TradePair], processed_ms: int, adv_lookback_window: int = 10,
                     calc_vol_window: int = 30):
        """
        return dict of features (avg daily volume and annualized volatility) for each trade pair
        """
        tp_to_adv = defaultdict()
        tp_to_vol = defaultdict()
        for trade_pair in trade_pairs:
            bars_df = cls.get_bars_with_features(trade_pair, processed_ms, adv_lookback_window, calc_vol_window)
            row_selected = bars_df.iloc[-1]
            annualized_volatility = row_selected['annualized_vol']
            avg_daily_volume = row_selected[f'adv_last_{adv_lookback_window}_days']

            tp_to_vol[trade_pair.trade_pair_id] = annualized_volatility
            tp_to_adv[trade_pair.trade_pair_id] = avg_daily_volume
        return tp_to_adv, tp_to_vol

    @classmethod
    def get_bars_with_features(cls, trade_pair: TradePair, processed_ms: int, adv_lookback_window: int=10, calc_vol_window: int=30, trading_days_in_a_year: int=252) -> pd.DataFrame:
        """
        fetch data for average daily volume (estimated daily volume) and annualized volatility
        """
        order_date = TimeUtil.millis_to_short_date_str(processed_ms)
        days_ago = max(adv_lookback_window, calc_vol_window) + 4  # +1 for last day, +1 because daily_returns is NaN for 1st day, +2 for padding (unexpected holidays)
        start_date = cls.holidays_nyse.get_nth_working_day(order_date, -days_ago).strftime("%Y-%m-%d")

        price_info_raw = cls.pds.unified_candle_fetcher(trade_pair, start_date, order_date, timespan="day")
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

    @classmethod
    def write_features_from_memory_to_disk(cls):
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_slippage_model_features_file(),
            cls.features
        )

    @staticmethod
    def get_order_size_bucket(size: float) -> str:
        """
        bucket the order size float into a string for reading from model parameters file
        """
        all_order_value_ranges = [(1_000, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, 200_000), (200_000, 300_000)]
        order_value_labels = ["1k_10k", "10k_50k", "50k_100k", "100k_200k", "200k_300k"]

        for (low, high), label in zip(all_order_value_ranges, order_value_labels):
            if low < size <= high:
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
    equities_order_buy = Order(price=100, processed_ms=TimeUtil.now_in_millis(),
                                    order_uuid="test_order",
                                    trade_pair=TradePair.NVDA,
                                    order_type=OrderType.LONG, leverage=1)
    slippage_buy = PriceSlippageModel.calculate_slippage(bid=99, ask=100, order=equities_order_buy)
    print(slippage_buy)
