import json
import threading
from collections import defaultdict
from typing import List

from polygon.rest.models import MarketStatus
from polygon.websocket import WebSocketClient, Market, EquityAgg, CurrencyAgg
from concurrent.futures import ThreadPoolExecutor, as_completed

from time_util.time_util import TimeUtil
from vali_config import TradePair, TradePairCategory
import time
from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt
from polygon import RESTClient

LOCK = threading.Lock()
UNSUPPORTED_TRADE_PAIRS = (TradePair.FTSE, TradePair.GDAXI)

trade_pair_to_price_history = defaultdict(list)
last_websocket_ping_time_s = {}
last_rest_update_time_s = {}
last_rest_datetime_received = {}
latest_websocket_prices = {}
closed_market_prices = {tp: None for tp in TradePair}
n_events_global = 0
DEBUG = 0
MARKET_STATUS = None
trade_pair_to_longest_seen_lag = {}

if DEBUG:
    import matplotlib.pyplot as plt

secrets = ValiUtils.get_secrets()

if 'polygon_apikey' not in secrets:
    POLYGON_CLIENT = None
else:
    POLYGON_CLIENT = RESTClient(api_key=secrets['polygon_apikey'])

POLY_WEBSOCKETS = {
    Market.Crypto: None,
    Market.Forex:  None,
    Market.Indices: None
}

POLY_WEBSOCKET_THREADS = {
    Market.Crypto: None,
    Market.Forex:  None,
    Market.Indices: None
}

def handle_msg(msgs: List[CurrencyAgg | EquityAgg]):
    global LOCK, n_events_global, closed_market_prices, latest_websocket_prices, last_websocket_ping_time_s, trade_pair_to_price_history
    with LOCK:
        try:
            for m in msgs:
                # bt.logging.info(f"Received price event: {event}")
                #print('received message:', m, type(m))
                if isinstance(m, EquityAgg):
                    tp = PolygonDataService.symbol_to_trade_pair(m.symbol[2:])  # I:SPX -> SPX
                elif isinstance(m, CurrencyAgg):
                    tp = PolygonDataService.symbol_to_trade_pair(m.pair)
                else:
                    raise ValueError(f"Unknown message in POLY websocket: {m}")
                price = float(m.close)
                latest_websocket_prices[tp] = price
                last_websocket_ping_time_s[tp] = time.time()
                # Reset the closed market price, indicating that a new close should be fetched after the current day's close
                closed_market_prices[tp] = None
                n_events_global += 1
                if DEBUG:
                    formatted_time = TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())
                    trade_pair_to_price_history[tp].append((formatted_time, price))
            if DEBUG:
                print('last message:', m, 'n msgs total:', len(msgs))
                history_size = sum(len(v) for v in trade_pair_to_price_history.values())
                bt.logging.info("History Size: " + str(history_size))
                bt.logging.info(f"n_events_global: {n_events_global}")
        except Exception as e:
            bt.logging.error(f"Failed to handle POLY websocket message with error: {e}")


def main_forex():
    global POLY_WEBSOCKETS
    POLY_WEBSOCKETS[Market.Forex].run(handle_msg)

def main_indices():
    global POLY_WEBSOCKETS
    POLY_WEBSOCKETS[Market.Indices].run(handle_msg)

def main_crypto():
    global POLY_WEBSOCKETS
    POLY_WEBSOCKETS[Market.Crypto].run(handle_msg)

def stop_start_websocket_threads():
    global POLY_WEBSOCKETS, POLY_WEBSOCKET_THREADS, LOCK
    if POLY_WEBSOCKETS[Market.Indices]:
        POLY_WEBSOCKETS[Market.Indices].close()
        POLY_WEBSOCKETS[Market.Forex].close()
        POLY_WEBSOCKETS[Market.Crypto].close()
    POLY_WEBSOCKETS[Market.Indices] = WebSocketClient(market=Market.Indices, api_key=secrets['polygon_apikey'])
    POLY_WEBSOCKETS[Market.Forex] = WebSocketClient(market=Market.Forex, api_key=secrets['polygon_apikey'])
    POLY_WEBSOCKETS[Market.Crypto] = WebSocketClient(market=Market.Crypto, api_key=secrets['polygon_apikey'])
    PolygonDataService.subscribe_websockets()
    if POLY_WEBSOCKET_THREADS[Market.Indices]:
        POLY_WEBSOCKET_THREADS[Market.Indices].join()
        POLY_WEBSOCKET_THREADS[Market.Forex].join()
        POLY_WEBSOCKET_THREADS[Market.Crypto].join()
        time.sleep(5)

    LOCK = threading.Lock()
    POLY_WEBSOCKET_THREADS[Market.Indices] = threading.Thread(target=main_indices, daemon=True)
    POLY_WEBSOCKET_THREADS[Market.Forex] = threading.Thread(target=main_forex, daemon=True)
    POLY_WEBSOCKET_THREADS[Market.Crypto] = threading.Thread(target=main_crypto, daemon=True)
    POLY_WEBSOCKET_THREADS[Market.Indices].start()
    POLY_WEBSOCKET_THREADS[Market.Forex].start()
    POLY_WEBSOCKET_THREADS[Market.Crypto].start()


def debug_log():
    global last_websocket_ping_time_s, latest_websocket_prices, closed_market_prices, trade_pair_to_longest_seen_lag, n_events_global
    trade_pairs_to_track = list(last_websocket_ping_time_s.keys())
    for tp in trade_pairs_to_track:
        lag = PolygonDataService.get_websocket_lag_for_trade_pair_s(tp)
        if tp not in trade_pair_to_longest_seen_lag:
            trade_pair_to_longest_seen_lag[tp] = lag
        else:
            if lag > trade_pair_to_longest_seen_lag[tp]:
                trade_pair_to_longest_seen_lag[tp] = lag
    # log how long it has been since the last ping
    formatted_lags = {tp.trade_pair_id: f"{lag:.2f}" for tp, lag in trade_pair_to_longest_seen_lag.items()}
    bt.logging.info(f"Worst POLY lags seen: {formatted_lags}")
    # Log the last time since websocket ping
    formatted_lags = {tp.trade_pair_id: f"{time.time() - timestamp:.2f}" for tp, timestamp in
                      last_websocket_ping_time_s.items()}
    bt.logging.info(f"Last POLY websocket pings: {formatted_lags}")
    # Log the prices
    formatted_prices = {tp.trade_pair_id: f"{price:.2f}" for tp, price in latest_websocket_prices.items()}
    bt.logging.info(f"Latest POLY websocket prices: {formatted_prices}")
    # Log which trade pairs are likely in closed markets
    closed_trade_pairs = {}
    for trade_pair in TradePair:
        if not PolygonDataService.is_market_open(trade_pair):
            closed_trade_pairs[trade_pair.trade_pair] = closed_market_prices[trade_pair]

    bt.logging.info(f"POLY Market closed with closing prices for {closed_trade_pairs}")
    bt.logging.info(f'POLY websocket n_events_global: {n_events_global}')

def websocket_manager():
    global MARKET_STATUS, n_events_global, websocket_client_crypto, websocket_client_indices, websocket_client_forex, POLYGON_CLIENT
    prev_n_events = None
    while True:
        if prev_n_events is None or prev_n_events == n_events_global:
            bt.logging.error(f"POLY websocket has not received any events in the last 60 seconds. n_events {n_events_global} Restarting websocket.")
            stop_start_websocket_threads()
        MARKET_STATUS = POLYGON_CLIENT.get_market_status()
        if not isinstance(MARKET_STATUS, MarketStatus):
            bt.logging.error(f"Failed to fetch market status. Received: {MARKET_STATUS}")

        debug_log()
        prev_n_events = n_events_global
        time.sleep(60)

class PolygonDataService:

    def __init__(self, api_key):
        self.init_time = time.time()
        self._api_key = api_key
        self.timespan_to_ms = {'second': 1000, 'minute': 1000 * 60, 'hour': 1000 * 60 * 60, 'day': 1000 * 60 * 60 * 24}


        self.trade_pair_lookup = {pair.trade_pair: pair for pair in TradePair}
        self.trade_pair_category_to_longest_allowed_lag = {TradePairCategory.CRYPTO: 30, TradePairCategory.FOREX: 30,
                                                           TradePairCategory.INDICES: 30}
        for trade_pair in TradePair:
            assert trade_pair.trade_pair_category in self.trade_pair_category_to_longest_allowed_lag, \
                f"Trade pair {trade_pair} has no allowed lag time"

        # Start thread to refresh market status
        self.websocket_manager_thread = threading.Thread(target=websocket_manager, daemon=True)
        self.websocket_manager_thread.start()
        time.sleep(3) # Let the websocket_manager_thread start

    @staticmethod
    def subscribe_websockets():
        global POLY_WEBSOCKETS
        for tp in TradePair:
            if tp in UNSUPPORTED_TRADE_PAIRS:
                continue  # not supported by polygon
            if tp.is_crypto:
                symbol = "XAS." + tp.trade_pair.replace('/', '-')
                POLY_WEBSOCKETS[Market.Crypto].subscribe(symbol)
            elif tp.is_forex:
                symbol = "CAS." + tp.trade_pair
                POLY_WEBSOCKETS[Market.Forex].subscribe(symbol)
            elif tp.is_indices:
                symbol = "A.I:" + tp.trade_pair
                POLY_WEBSOCKETS[Market.Indices].subscribe(symbol)
            else:
                raise ValueError(f"Unknown trade pair category: {tp.trade_pair_category}")

    @staticmethod
    def is_market_open(trade_pair: TradePair):
        global MARKET_STATUS
        if MARKET_STATUS is None:
            return False
        if not isinstance(MARKET_STATUS, MarketStatus):
            return False
        if trade_pair.trade_pair_category == TradePairCategory.CRYPTO:
            return MARKET_STATUS.currencies.crypto == 'open'
        elif trade_pair.trade_pair_category == TradePairCategory.FOREX:
            return MARKET_STATUS.currencies.fx == 'open'
        elif trade_pair.trade_pair_category == TradePairCategory.INDICES:
            #if trade_pair == TradePair.SPX:
            #    return MARKET_STATUS.indicesGroups.s_and_p == 'open'
            #elif trade_pair == TradePair.DJI:
            #    return MARKET_STATUS.indicesGroups.dow_jones == 'open'
            #elif trade_pair in UNSUPPORTED_TRADE_PAIRS:
            #    return False
            #else:
            #    raise ValueError(f"Unknown trade pair id: {trade_pair.trade_pair_id}")
            return MARKET_STATUS.market == 'open'

        else:
            raise ValueError(f"Unknown trade pair: {trade_pair}")


    @staticmethod
    def get_websocket_lag_for_trade_pair_s(trade_pair: TradePair):
        global last_websocket_ping_time_s
        if trade_pair in last_websocket_ping_time_s:
            return time.time() - last_websocket_ping_time_s.get(trade_pair, 0)
        return None

    def spill_price_history(self):
        # Write the price history to disk in a format that will let us plot it
        filename = f"price_history.json"
        with open(filename, 'w') as f:
            json.dump(trade_pair_to_price_history, f)

    @staticmethod
    def symbol_to_trade_pair(symbol: str):
        # Should work for indices and forex
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol)
        if tp:
            return tp
        # Should work for crypto. Anything else will return None
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol.replace('-', '/'))
        if not tp:
            raise ValueError(f"Unknown symbol: {symbol}")
        return tp

    def get_closes_rest(self, trade_pairs: List[TradePair]) -> dict:
        all_trade_pair_closes = {}
        # Multi-threaded fetching of REST data over all requested trade pairs. Max parallelism is 5.
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Dictionary to keep track of futures
            future_to_trade_pair = {executor.submit(self.get_close_rest, trade_pair): trade_pair for trade_pair in
                                    trade_pairs}

            for future in as_completed(future_to_trade_pair):
                trade_pair = future_to_trade_pair[future]
                try:
                    result = future.result()
                    all_trade_pair_closes.update(result)
                except Exception as exc:
                    print(f"{trade_pair} generated an exception: {exc}")

        return all_trade_pair_closes

    def get_close_rest(
        self,
        trade_pair: TradePair
    ) -> dict:
        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)
        #bt.logging.info(f"Fetching REST data for {polygon_ticker}")

        if not PolygonDataService.is_market_open(trade_pair):
            return {trade_pair: self.get_price_before_market_close(trade_pair)}

        aggs = []
        now = TimeUtil.now_in_millis()
        price = None
        prev_timestamp = None
        for a in POLYGON_CLIENT.list_aggs(
                polygon_ticker,
                1,
                "second",
                now - 10000,
                now + 2000
        ):
            #print('agg:', a)
            price = a.close
            epoch_miliseconds = a.timestamp
            assert prev_timestamp is None or prev_timestamp < epoch_miliseconds
            #formatted_date = TimeUtil.millis_to_formatted_date_str(epoch_miliseconds // 1000)
            aggs.append(a)
            prev_timestamp = epoch_miliseconds
        if not aggs:
            bt.logging.error(f"Polygon failed to fetch REST data for {trade_pair.trade_pair}. If you keep seeing this error, report it to the team ASAP")
        else:
            #print('found aggs:', aggs, 'date', formatted_date)
            self.update_last_rest_update_time({trade_pair: a})
        return {trade_pair: price}

    def update_last_rest_update_time(self, data):
        bt.logging.trace(f"update_last_rest_update_time received data: {[(k.trade_pair, v) for k, v in data.items()]}")
        for trade_pair, d in data.items():
            symbol = trade_pair.trade_pair
            previous_datetime = last_rest_datetime_received.get(symbol, '')
            last_rest_datetime_received[symbol] = d.timestamp
            if previous_datetime != last_rest_datetime_received[symbol]:
                last_rest_update_time_s[symbol] = time.time()
                bt.logging.trace(
                    f"Updated last_rest_update_time_s for {trade_pair.trade_pair} at {last_rest_update_time_s[symbol]}")


    def trade_pair_to_polygon_ticker(self, trade_pair: TradePair):
        if trade_pair.trade_pair_category == TradePairCategory.CRYPTO:
            return 'X:' + trade_pair.trade_pair_id
        elif trade_pair.trade_pair_category == TradePairCategory.FOREX:
            return 'C:' + trade_pair.trade_pair_id
        elif trade_pair.trade_pair_category == TradePairCategory.INDICES:
            return 'I:' + trade_pair.trade_pair_id
        else:
            raise ValueError(f"Unknown trade pair category: {trade_pair.trade_pair_category}")

    def get_price_before_market_close(self, trade_pair: TradePair):
        global closed_market_prices
        if closed_market_prices[trade_pair] is not None:
            return closed_market_prices[trade_pair]
        elif trade_pair in UNSUPPORTED_TRADE_PAIRS:
            return None

        # start 7 days ago
        end_time_ms = TimeUtil.now_in_millis()
        start_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 24 * 7
        candles = self.get_candles_for_trade_pair(trade_pair, start_time_ms, end_time_ms)
        if len(candles) == 0:
            msg = f"Failed to fetch market close for {trade_pair.trade_pair}"
            raise ValueError(msg)

        ans = candles[-1].close
        closed_market_prices[trade_pair] = ans
        return closed_market_prices[trade_pair]

    def get_closes_websocket(self, trade_pairs: List[TradePair]):
        closes = {}
        for trade_pair in trade_pairs:
            if trade_pair in latest_websocket_prices:
                price = latest_websocket_prices[trade_pair]
                lag = PolygonDataService.get_websocket_lag_for_trade_pair_s(trade_pair)
                is_stale = lag > self.trade_pair_category_to_longest_allowed_lag[trade_pair.trade_pair_category]
                if is_stale:
                    bt.logging.warning(f"Found stale POLY websocket data for {trade_pair.trade_pair}. Lag: {lag} seconds. "
                                       f"Max allowed lag for category: "
                                       f"{self.trade_pair_category_to_longest_allowed_lag[trade_pair.trade_pair_category]} seconds."
                                       f"Ignoring this data.")
                else:
                    closes[trade_pair] = price

        return closes

    def get_close_websocket(self, trade_pair: TradePair):
        if trade_pair in latest_websocket_prices and trade_pair in last_websocket_ping_time_s:
            price = latest_websocket_prices[trade_pair]
            timestamp = last_websocket_ping_time_s.get(trade_pair, 0)
            max_allowed_lag = self.trade_pair_category_to_longest_allowed_lag[trade_pair.trade_pair_category]
            is_stale = time.time() - timestamp > max_allowed_lag
            if is_stale:
                bt.logging.info(f"Found stale POLY websocket data for {trade_pair.trade_pair}. Lag: {time.time() - timestamp} "
                                f"seconds. Max allowed lag for category: {max_allowed_lag} seconds. Ignoring this data.")
            else:
                return price

        return None

    def get_close(self, trade_pair: TradePair) -> float | None:
        ans = self.get_close_websocket(trade_pair)
        if not ans:
            bt.logging.info(f"Fetching stale trade pair using POLY REST: {trade_pair.trade_pair}")
            ans = self.get_close_rest(trade_pair)[trade_pair]
            bt.logging.info(f"Received POLY REST data for {trade_pair.trade_pair}: {ans}")
        else:
            bt.logging.info(f"Using POLY websocket data for {trade_pair.trade_pair}")
        return ans

    def get_closes(self, trade_pairs: List[TradePair], websocket_only=False) -> dict:
        closes = self.get_closes_websocket(trade_pairs)
        missing_trade_pairs = []
        for tp in trade_pairs:
            if tp not in closes or closes[tp] is None:
                missing_trade_pairs.append(tp)

        debug = {k.trade_pair: v for k, v in closes.items()}
        bt.logging.info(f"Received POLY websocket data: {debug}")

        if missing_trade_pairs and not websocket_only:
            rest_closes = self.get_closes_rest(missing_trade_pairs)
            debug = {k.trade_pair: v for k, v in rest_closes.items()}
            bt.logging.info(f"Received stale/websocket-less data using POLY REST: {debug}")
            closes.update(rest_closes)

        return closes

    def get_close_in_past_hour_fallback(self, trade_pair: TradePair, timestamp_ms: int):
        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)

        if not PolygonDataService.is_market_open(trade_pair):
            return self.get_price_before_market_close(trade_pair)

        prev_timestamp = None
        smallest_delta = None
        corresponding_price = None
        start_time = None
        n_responses = 0
        candle = None
        timespan = "hour"

        def try_updating_found_price(t, p):
            nonlocal smallest_delta, corresponding_price, start_time, candle
            time_delta_ms = abs(t - timestamp_ms)
            if smallest_delta is None or time_delta_ms < smallest_delta:
                smallest_delta = time_delta_ms
                corresponding_price = p
                start_time = epoch_miliseconds
                candle = a

        for a in POLYGON_CLIENT.list_aggs(
                polygon_ticker,
                1,
                timespan,
                timestamp_ms - 1000 * 60 * 60 * 48,
                timestamp_ms + 1000 * 60 * 30
        ):
            n_responses += 1
            epoch_miliseconds = a.timestamp

            try_updating_found_price(epoch_miliseconds, a.open)
            try_updating_found_price(epoch_miliseconds + self.timespan_to_ms[timespan], a.close)

            assert prev_timestamp is None or prev_timestamp < epoch_miliseconds
            prev_timestamp = epoch_miliseconds

        print(f"hourly fallback smallest delta s: {smallest_delta / 1000 if smallest_delta else None}, input_timestamp: {timestamp_ms}, candle_start_time_ms: {start_time}, candle: {candle}, n_responses: {n_responses}")
        return corresponding_price



    def get_close_at_date_minute_fallback(self, trade_pair: TradePair, timestamp_ms: int):
        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)

        if not PolygonDataService.is_market_open(trade_pair):
            return self.get_price_before_market_close(trade_pair)

        prev_timestamp = None
        smallest_delta = None
        corresponding_price = None
        start_time = None
        n_responses = 0
        candle = None
        timespan = "minute"

        def try_updating_found_price(t, p):
            nonlocal smallest_delta, corresponding_price, start_time, candle
            time_delta_ms = abs(t - timestamp_ms)
            if smallest_delta is None or time_delta_ms < smallest_delta:
                smallest_delta = time_delta_ms
                corresponding_price = p
                start_time = epoch_miliseconds
                candle = a

        for a in POLYGON_CLIENT.list_aggs(
                polygon_ticker,
                1,
                timespan,
                timestamp_ms - 1000 * 60 * 30,
                timestamp_ms + 1000 * 60 * 30
        ):
            n_responses += 1
            epoch_miliseconds = a.timestamp

            try_updating_found_price(epoch_miliseconds, a.open)
            try_updating_found_price(epoch_miliseconds + self.timespan_to_ms[timespan], a.close)

            assert prev_timestamp is None or prev_timestamp < epoch_miliseconds
            prev_timestamp = epoch_miliseconds

        print(f"minute fallback smallest delta ms: {smallest_delta}, input_timestamp: {timestamp_ms}, candle_start_time_ms: {start_time}, candle: {candle}, n_responses: {n_responses}")
        return corresponding_price

    def get_close_at_date_second(self, trade_pair: TradePair, timestamp_ms: int):
        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)

        if not PolygonDataService.is_market_open(trade_pair):
            return self.get_price_before_market_close(trade_pair)

        prev_timestamp = None
        smallest_delta = None
        corresponding_price = None
        start_time = None
        n_responses = 0
        candle = None
        timespan = "second"

        def try_updating_found_price(t, p):
            nonlocal smallest_delta, corresponding_price, start_time, candle
            time_delta_ms = abs(t - timestamp_ms)
            if smallest_delta is None or time_delta_ms < smallest_delta:
                smallest_delta = time_delta_ms
                corresponding_price = p
                start_time = epoch_miliseconds
                candle = a

        for a in POLYGON_CLIENT.list_aggs(
                polygon_ticker,
                1,
                timespan,
                timestamp_ms - 1000 * 8,
                timestamp_ms + 1000 * 8
        ):
            n_responses += 1
            epoch_miliseconds = a.timestamp

            try_updating_found_price(epoch_miliseconds, a.open)
            try_updating_found_price(epoch_miliseconds + self.timespan_to_ms[timespan], a.close)

            assert prev_timestamp is None or prev_timestamp < epoch_miliseconds
            prev_timestamp = epoch_miliseconds

        print(f"smallest delta ms: {smallest_delta}, input_timestamp: {timestamp_ms}, candle_start_time_ms: {start_time}, candle: {candle}, n_responses: {n_responses}")
        return corresponding_price

    def get_range_of_closes(self, trade_pair, start_date: str, end_date: str):
        ts = self.td.time_series(symbol=trade_pair, interval='1min', start_date=start_date, end_date=end_date, outputsize=5000)
        response = ts.as_json()
        closes = [(d['datetime'], float(d["close"])) for d in response]
        return closes

    def get_candles(self, trade_pairs, start_time_ms, end_time_ms):
        # Dictionary to store the minimum prices for each trade pair
        ret = {}

        # Create a ThreadPoolExecutor with a maximum of 5 threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Future objects dictionary to hold the ongoing computations
            futures = {executor.submit(self.get_candles_for_trade_pair, tp, start_time_ms, end_time_ms): tp for tp in trade_pairs}

            # Retrieve the results as they complete
            for future in as_completed(futures):
                trade_pair = futures[future]
                try:
                    # Collect the result from future
                    result = future.result()
                    ret[trade_pair] = result
                except Exception as exc:
                    print(f'{trade_pair} get_min_prices_in_window generated an exception: {exc}')

        # Return the collected results
        return ret

    def get_candles_for_trade_pair(
        self,
        trade_pair: TradePair,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> list | None:

        delta_time_ms = end_timestamp_ms - start_timestamp_ms
        delta_time_seconds = delta_time_ms / 1000
        delta_time_minutes = delta_time_seconds / 60
        delta_time_hours = delta_time_minutes / 60

        if delta_time_seconds < 70:
            timespan = "second"
        elif delta_time_minutes < 70:
            timespan = "minute"
        elif delta_time_hours < 70:
            timespan = "hour"
        else:
            timespan = "day"

        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)

        aggs = []
        prev_timestamp = None
        for i, a in enumerate(POLYGON_CLIENT.list_aggs(
                polygon_ticker,
                1,
                timespan,
                start_timestamp_ms,
                end_timestamp_ms
        )):
            epoch_miliseconds = a.timestamp
            assert prev_timestamp is None or epoch_miliseconds >= prev_timestamp, ('candles not sorted', prev_timestamp, epoch_miliseconds)
            #formatted_date = TimeUtil.millis_to_formatted_date_str(epoch_miliseconds)
            #if i != -1:
                #print('        agg:', a.low, formatted_date, trade_pair.trade_pair_id, timespan)
            aggs.append(a)
            prev_timestamp = epoch_miliseconds
        if not aggs:
            bt.logging.trace(f"Polygon failed to fetch candle data for {trade_pair.trade_pair}. "
                             f" Perhaps this trade pair was closed during the specified window.")
            return None
        else:
            #print('found aggs:', aggs, 'date', formatted_date)
            self.update_last_rest_update_time({trade_pair: aggs[-1]})
            return aggs



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()

    # Initialize client
    polygon_data_provider = PolygonDataService(api_key=secrets['polygon_apikey'])

    #time.sleep(12)
    for tp in [TradePair.SPX]:
        print('getting candles for', tp.trade_pair_id)
        candles = polygon_data_provider.get_candles_for_trade_pair(tp, 1713206962000, 1713207000000)
        for c in candles[:10]:
            print('    ', c)
    #print(polygon_data_provider.get_close_at_date_second(TradePair.CADJPY, 1712746241174))
    assert 0



    #for t in polygon_data_provider.polygon_client.list_tickers(market="indices", limit=1000):
    #    if t.ticker.startswith("I:F"):
    #        print(t.ticker)

    times_to_test = []

    start_time_ms = TimeUtil.now_in_millis() - 1000 * 10  # 10 seconds ago
    end_time_ms = TimeUtil.now_in_millis() - 1000 * 5  # 5 seconds ago
    times_to_test.append((start_time_ms, end_time_ms))

    start_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 10  # 10 minutes ago
    end_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 5  # 5 minutes ago
    times_to_test.append((start_time_ms, end_time_ms))

    start_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 10  # 10 hours ago
    end_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 9  # 9 hours ago
    times_to_test.append((start_time_ms, end_time_ms))

    start_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 10 * 24  # 10 days ago
    end_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 5 * 24  # 5 days ago
    times_to_test.append((start_time_ms, end_time_ms))

    for start_time_ms, end_time_ms in times_to_test:
        start_date_formatted = TimeUtil.millis_to_formatted_date_str(start_time_ms)
        end_date_formatted = TimeUtil.millis_to_formatted_date_str(end_time_ms)
        print('-------------------------------------------------------------------')
        print(f"Testing between {start_date_formatted} and {end_date_formatted}")

        tps = [tp for tp in TradePair if tp not in UNSUPPORTED_TRADE_PAIRS]
        candles = polygon_data_provider.get_candles(tps, start_time_ms, end_time_ms)
        print(f"    candles: {candles}")

    for tp in TradePair:
        if tp != TradePair.GBPUSD:
            continue

        is_open = PolygonDataService.is_market_open(tp)
        print(f'market is open for {tp}: ', is_open)
        print('PRICE BEFORE MARKET CLOSE: ', polygon_data_provider.get_price_before_market_close(tp))
        print('getting close for', tp.trade_pair_id, ':', polygon_data_provider.get_close_rest(tp)[tp])






    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SPX, TradePair.GBPUSD, TradePair.DJI]
    print("-----------------REST-----------------")
    ans_rest = polygon_data_provider.get_closes_rest(trade_pairs)
    for k, v in ans_rest.items():
        print(f"{k.trade_pair_id}: {v}")
    print("-----------------WEBSOCKET-----------------")
    ans_ws = polygon_data_provider.get_closes_websocket(trade_pairs)
    for k, v in ans_ws.items():
        print(f"{k.trade_pair_id}: {v}")
    print("-----------------Normal-----------------")
    ans_n = polygon_data_provider.get_closes(trade_pairs)
    for k, v in ans_n.items():
        print(f"{k.trade_pair_id}: {v}")
    print("-----------------Done-----------------")






    cached_prices = {}
    cached_times = {}
    while True:
        print('main thread perspective - n_events_global:', n_events_global)
        for k, price in latest_websocket_prices.items():
            t = last_websocket_ping_time_s[k]
            cached_price = cached_prices.get(k, None)
            cached_time = cached_times.get(k, None)
            if cached_price and cached_time:
                assert cached_time <= t, (cached_time, t)

            cached_prices[k] = price
            cached_times[k] = t
        debug_log()
        time.sleep(10)