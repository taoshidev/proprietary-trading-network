from typing import List

from time_util.time_util import TimeUtil
from vali_config import TradePair

from twelvedata import TDClient


class TwelveData:
    def __init__(self, api_key):
        self._api_key = api_key

    def _fetch_data(self,
                    symbols,
                    interval,
                    output_size):

        td = TDClient(apikey=self._api_key)

        ts = td.time_series(
            symbol=symbols,
            interval=interval,
            outputsize=output_size
        )

        response = ts.as_json()

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Error fetching data from twelvedata")

    def get_closes(self, trade_pairs: List[TradePair] | TradePair,
                   output_size: int = 1,
                   interval: str = '1min'):

        stringified_trade_pairs = ','.join(map(str, trade_pairs))

        all_trade_pair_closes = {}

        data = self._fetch_data(stringified_trade_pairs, interval, output_size)
        for k, v in data.items():
            all_trade_pair_closes[TradePair[k]] = v["values"]["close"]

        return all_trade_pair_closes
