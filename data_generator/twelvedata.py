import requests

from time_util.time_util import TimeUtil


class TwelveData():
    def __init__(self, api_key):
        self._api_key = api_key

    @staticmethod
    def _fetch_data(symbol,
                    interval,
                    output_size,
                    exchange=None):

        url = f'https://api.twelvedata.com/time_series?' \
              f'symbol={symbol}' \
              f'&interval={interval}' \
              f'&outputsize={output_size}' \
              f'&exchange={exchange}'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['values']
        else:
            raise Exception("Error fetching data from twelvedata")

    def get_data(self, symbol: str,
                 output_size: int = 1,
                 interval = '1min',
                 start: int = None,
                 end: int = None,
                 exchange: str = None,
                 close_only: bool = True):

        def request_close_only(_close_only):
            if _close_only:
                return item['close']
            else:
                return item

        aggregated_data = []
        data = self._fetch_data(symbol, interval, output_size, exchange)
        for item in data:
            timestamp = TimeUtil.timestamp_to_millis(item['datetime'])
            if start is not None and end is not None and start <= timestamp <= end:
                aggregated_data.append(request_close_only(item))
            else:
                aggregated_data.append(request_close_only(item))
        return aggregated_data
