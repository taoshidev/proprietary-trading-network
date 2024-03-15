import requests
from datetime import datetime

import time
from typing import List, Tuple

from data_generator.financial_markets_generator.base_financial_markets_generator.base_financial_markets_generator import \
    BaseFinancialMarketsGenerator

import requests

from time_util.time_util import TimeUtil
from vali_config import ValiConfig


class ByBitData(BaseFinancialMarketsGenerator):
    def __init__(self):
        super().__init__()
        self._symbols = {
            "BTCUSD": "BTCUSDT"
        }

    def get_data(self,
                 symbol='BTCUSD',
                 interval=ValiConfig.STANDARD_TF,
                 start=None,
                 end=None,
                 retries=0,
                 limit=1000):

        if symbol != "BTCUSDT":
            symbol = self._symbols[symbol]

        downshifted_start_by_one_unit = start - TimeUtil.minute_in_millis(interval)
        downshifted_end_by_one_unit = end - TimeUtil.minute_in_millis(interval)

        url = f"https://api.bybit.com/v5/market/kline?" \
              f"category=spot&symbol={symbol}&interval={interval}&start={downshifted_start_by_one_unit}&end={downshifted_end_by_one_unit}&limit={limit}"
        response = requests.get(url)

        try:
            if response.status_code == 200:
                results = response.json()["result"]["list"]
                reversed_data = sorted(results, key=lambda x: int(x[0]))
                return reversed_data
            else:
                raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
        except Exception:
            if retries < 5:
                time.sleep(retries)
                retries += 1
                # print("retrying getting historical bybit data")
                self.get_data(symbol, interval, start, end, retries)
            else:
                raise ConnectionError("max number of retries exceeded trying to get bybit data")

    def get_data_and_structure_data_points(self, symbol: str, tf: int, data_structure: List[List], ts_range: Tuple[int, int]):
        bd = self.get_data(symbol=symbol, interval= tf, start=ts_range[0], end=ts_range[1])
        # print("received bybit historical data from : ", TimeUtil.millis_to_timestamp(ts_range[0]),
        #       TimeUtil.millis_to_timestamp(ts_range[1]))
        self.convert_output_to_data_points(data_structure,
                                           bd,
                                           [0, 4, 2, 3, 5]
                                           )

    @staticmethod
    def convert_output_to_data_points(data_structure: List[List], days_data: List[List], order_to_ds: List[int]):
        """
        return close time, close, high, low, volume
        """
        for tf_row in days_data:
            # bybit receives only open and not close
            data_structure[0].append(int(tf_row[order_to_ds[0]])+TimeUtil.minute_in_millis(ValiConfig.STANDARD_TF))
            data_structure[1].append(float(tf_row[order_to_ds[1]]))
            data_structure[2].append(float(tf_row[order_to_ds[2]]))
            data_structure[3].append(float(tf_row[order_to_ds[3]]))
            data_structure[4].append(float(tf_row[order_to_ds[4]]))

