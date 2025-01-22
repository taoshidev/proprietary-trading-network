from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from polygon import RESTClient

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair

class PriceSlippageModel:
    def __init__(self, live_price_fetcher):
        self.live_price_fetcher = live_price_fetcher
        self.parameters = self.read_slippage_model_parameters()
        self.sampled_df = None

    # TODO: uses model and parameters to calculate slippage percentage. in the validator we take the slippage percentage
    #  and multiply it by the price to get the avg execution price

    def calculate_slippage(self, trade_pair: TradePair, side: str, size: float, processed_ms: int) -> float:
        """
        returns the percentage slippage of the current order
        """
        # bucket size
        size_str = self.get_order_size_bucket(size)

        model_config = self.parameters[trade_pair.trade_pair_id][side][size_str]

        print(model_config)

        intercept = model_config["intercept"]
        features = model_config["features"]

        c1 = features["spread/price"]
        c2 = features["annualized_vol"]
        c3 = features["buy_order_value/adv"]

        metrics = self.calculate_metrics(trade_pair, processed_ms)

        slippage_cost_percentage = intercept + (c1 * metrics["spread"] / metrics["price"]) + (c2 * metrics["annualized_vol"]) + (c3 * size / metrics["adv"])
        return slippage_cost_percentage

    def calculate_metrics(self, trade_pair, processed_ms, adv_lookback_window=10, calc_vol_window=252):
        # Convert timestamp to datetime
        processed_date = datetime.fromtimestamp(processed_ms / 1000, tz=timezone.utc)
        start_date = (processed_date - timedelta(days=adv_lookback_window)).strftime("%Y-%m-%d")
        end_date = processed_date.strftime("%Y-%m-%d")

        client = self.live_price_fetcher.POLYGON_CLIENT

        # Fetch order book data for the specific timestamp
        try:
            #order_book = client.get_last_quote(ticker)
            lowest_ask = self.live_price_fetcher.get_lowest_ask(trade_pair, processed_ms)  # order_book.ask_price
            highest_bid = self.live_price_fetcher.get_highest_bid(trade_pair, processed_ms)  # order_book.bid_price

            spread = lowest_ask - highest_bid
            price = (lowest_ask + highest_bid) / 2
        except Exception as e:
            raise RuntimeError(f"Error fetching order book data: {e}")

        # Fetch historical data for ADV and volatility
        try:
            aggs = []
            for a in client.list_aggs(trade_pair.trade_pair_id, 1, "day", start_date, end_date, limit=5000):
                aggs.append(a)

            bars_pd = pd.DataFrame(aggs)
            bars_pd['daily_returns'] = np.log(bars_pd["close"] / bars_pd["close"].shift(1))

            # Calculate ADV
            adv = (bars_pd['volume'].rolling(window=adv_lookback_window + 1).sum() - bars_pd['volume']) / adv_lookback_window  # excluding the current day

            # Calculate annualized volatility
            rolling_std = bars_pd['daily_returns'].rolling(window=calc_vol_window).std()
            annualized_vol = rolling_std * np.sqrt(calc_vol_window)

        except Exception as e:
            raise RuntimeError(f"Error fetching historical data: {e}")

        return {
            "spread": spread,
            "price": price,
            "adv": adv,
            "annualized_vol": annualized_vol
        }

    @staticmethod
    def get_order_size_bucket(size: float) -> str:
        all_order_value_ranges = [(1_000, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, 300_000)]
        order_value_labels = ["1k_10k", "10k_50k", "50k_100k", "100k_300k"]

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
        equity_parameters = ValiUtils.get_vali_json_file_dict(ValiBkpUtils.get_equity_slippage_model_parameters_file())
        # print(equity_parameters)
        return equity_parameters


if __name__ == "__main__":
    PriceSlippageModel().read_slippage_model_parameters()
