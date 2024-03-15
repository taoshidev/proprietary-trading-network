import time
from typing import List, Tuple

from data_generator.financial_markets_generator.base_financial_markets_generator.base_financial_markets_generator import \
    BaseFinancialMarketsGenerator

import requests

from time_util.time_util import TimeUtil
from vali_config import ValiConfig


class KrakenData(BaseFinancialMarketsGenerator):
    def __init__(self):
        super().__init__()
        self.symbols = {
            "BTCUSD": "XXBTZUSD"
        }

    def get_data(self,
                 symbol='BTCUSD',
                 interval=ValiConfig.STANDARD_TF,
                 start=None,
                 end=None,
                 retries=0):

        pair = self.symbols[symbol]

        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={start}"

        response = requests.get(url)

        try:
            if response.status_code == 200:
                data = response.json()["result"][pair]
                return [entry for entry in data if entry[0] <= end * 1000]
            else:
                raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
        except Exception:
            if retries < 5:
                time.sleep(retries)
                retries += 1
                print("retrying getting historical kraken data")
                self.get_data(symbol, interval, start, end, retries)
            else:
                raise ConnectionError("max number of retries exceeded trying to get kraken data")

    def get_data_and_structure_data_points(self, symbol: str, data_structure: List[List], ts_range: Tuple[int, int]):
        kd = self.get_data(symbol=symbol, start=ts_range[0], end=ts_range[1])
        print("received kraken historical data from : ", TimeUtil.millis_to_timestamp(ts_range[0]),
              TimeUtil.millis_to_timestamp(ts_range[1]))
        for row in kd:
            print(TimeUtil.seconds_to_timestamp(row[0]))
        self.convert_output_to_data_points(data_structure,
                                           kd,
                                           [1,2,3,6]
                                           )

