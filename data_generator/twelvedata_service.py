import time
from typing import List

from requests import ReadTimeout

from time_util.time_util import TimeUtil
from vali_config import TradePair

from twelvedata import TDClient


class TwelveDataService:
    def __init__(self, api_key):
        self._api_key = api_key

    def _fetch_data(self, symbols, interval, output_size):
        td = TDClient(apikey=self._api_key)

        ts = td.time_series(symbol=symbols, interval=interval, outputsize=output_size)

        response = ts.as_json()
        return response

    def get_closes(
        self,
        trade_pairs: List[TradePair],
        output_size: int = 1,
        interval: str = "1min",
    ):
        trade_pair_values = [trade_pair.value for trade_pair in trade_pairs]
        stringified_trade_pairs = ",".join(map(str, trade_pair_values))

        all_trade_pair_closes = {}

        trade_pair_lookup = {pair.value: pair for pair in TradePair}

        data = self._fetch_data(stringified_trade_pairs, interval, output_size)
        for k, v in data.items():
            for c in v:
                all_trade_pair_closes[trade_pair_lookup[k]] = float(c["close"])

        return all_trade_pair_closes

    def get_close(
        self,
        trade_pair: TradePair,
        output_size: int = 1,
        interval: str = "1min",
        retries: int = 5,
    ):
        try:
            data = self._fetch_data(trade_pair.value, interval, output_size)
        except ReadTimeout:
            time.sleep(5)
            retries -= 1
            if retries > 0:
                self.get_close(trade_pair)

        all_trade_pair_closes = {trade_pair: float(data[0]["close"])}

        return all_trade_pair_closes
