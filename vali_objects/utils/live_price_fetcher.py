from data_generator.twelvedata_service import TwelveDataService

import time
from functools import wraps
import bittensor as bt

from vali_config import TradePair
from vali_objects.utils.vali_utils import ValiUtils


def retry(tries=5, delay=5, backoff=1):
    """
    Retry decorator with exponential backoff, works for all exceptions.

    Parameters:
    - tries: number of times to try (not retry) before giving up.
    - delay: initial delay between retries in seconds.
    - backoff: backoff multiplier e.g. value of 2 will double the delay each retry.

    Usage:
    @retry(tries=5, delay=5, backoff=2)
    def my_func():
        pass
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    bt.logging.error(f"Error: {str(e)}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)  # Last attempt
        return f_retry
    return deco_retry

class LivePriceFetcher():
    def __init__(self, secrets):
        self.twelve_data = TwelveDataService(api_key=secrets["twelvedata_apikey"])

    @retry(tries=2, delay=5, backoff=2)
    def get_close(self, trade_pair):
        return self.twelve_data.get_close(trade_pair=trade_pair)

    @retry(tries=2, delay=5, backoff=2)
    def get_closes(self, trade_pairs: list):
        return self.twelve_data.get_closes(trade_pairs)

    def is_market_closed_for_trade_pair(self, trade_pair):
        return self.twelve_data.trade_pair_market_likely_closed(trade_pair)



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets)
    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD]
    ans = live_price_fetcher.get_closes(trade_pairs)
    for k, v in ans.items():
        print(f"{k.trade_pair_id}: {v}")
    print("Done")

