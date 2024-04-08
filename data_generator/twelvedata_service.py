import json
import threading
from collections import defaultdict
from typing import List

from time_util.time_util import TimeUtil
from vali_config import TradePair, TradePairCategory
import time
from twelvedata import TDClient
from datetime import datetime
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt
import requests

WS = None
trade_pair_to_price_history = defaultdict(list)
last_websocket_ping_time_s = ValiBkpUtils.safe_load_dict_from_disk('last_websocket_ping_time_s.json', {})
last_rest_update_time_s = ValiBkpUtils.safe_load_dict_from_disk('last_rest_update_time_s.json', {})
last_rest_datetime_received = ValiBkpUtils.safe_load_dict_from_disk('last_rest_datetime_received.json', {})
latest_websocket_prices = ValiBkpUtils.safe_load_dict_from_disk('latest_websocket_prices.json', {})
n_events_global = 0
DEBUG = 0

if DEBUG:
    import matplotlib.pyplot as plt

class TwelveDataService:

    def __init__(self, api_key):
        self.n_events_local = 0
        self.n_resets = 0
        self.init_time = time.time()
        self._api_key = api_key
        self.td = TDClient(apikey=self._api_key)

        self._reset_websocket()
        self._heartbeat_thread = threading.Thread(target=self._websocket_heartbeat)
        self._heartbeat_thread.daemon = True
        self._heartbeat_thread.start()

        self._periodic_save_thread = threading.Thread(target=self._periodic_save)
        self._periodic_save_thread.daemon = True
        self._periodic_save_thread.start()

        self.trade_pair_lookup = {pair.trade_pair: pair for pair in TradePair}
        self.trade_pair_to_longest_seen_lag = {}
        self.trade_pair_category_to_longest_allowed_lag = {TradePairCategory.CRYPTO: 60, TradePairCategory.FOREX: 60,
                                                           TradePairCategory.INDICES: 60}
        for trade_pair in TradePair:
            assert trade_pair.trade_pair_category in self.trade_pair_category_to_longest_allowed_lag, \
                f"Trade pair {trade_pair} has no allowed lag time"

    def _periodic_save(self):
        while True:
            time.sleep(60)  # Save every 60 seconds
            ValiBkpUtils.safe_save_dict_to_disk('last_websocket_ping_time_s.json', last_websocket_ping_time_s)
            ValiBkpUtils.safe_save_dict_to_disk('last_rest_update_time_s.json', last_rest_update_time_s)
            ValiBkpUtils.safe_save_dict_to_disk('last_rest_datetime_received.json', last_rest_datetime_received)
            ValiBkpUtils.safe_save_dict_to_disk('latest_websocket_prices.json', latest_websocket_prices)

    def trade_pair_market_likely_closed(self, trade_pair: TradePair):
        # Check if the last websocket ping happened within the last 15 minutes
        symbol = trade_pair.trade_pair
        if symbol in last_websocket_ping_time_s:
            return time.time() - last_websocket_ping_time_s[symbol] > 900
        # Check if the last REST ping succeeded within the last 15 minutes (validators who don't have enterprise tier...)
        if symbol in last_rest_update_time_s:
            return time.time() - last_rest_update_time_s[symbol] > 900
        return True


    def _on_event(self, event):
        global n_events_global
        if event['event'] == 'price':
            #bt.logging.info(f"Received price event: {event}")
            symbol = event['symbol']
            price = float(event['price'])
            formatted_time = TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())
            latest_websocket_prices[symbol] = price
            last_websocket_ping_time_s[symbol] = time.time()
            trade_pair_to_price_history[symbol].append((formatted_time, price))
            self.n_events_local += 1
            n_events_global += 1
            if DEBUG:
                history_size = sum(len(v) for v in trade_pair_to_price_history.values())
                bt.logging.info("History Size: " + str(history_size))
                bt.logging.info(f"n_events_global: {n_events_global}, n_events_local: {self.n_events_local}")

        elif event['event'] == 'subscribe-status':
            if event['status'] == 'ok':
                bt.logging.info(f"Subscribed to symbols: {event['success']}")
            else:
                bt.logging.error(f"YOU LIKELY NEED TO UPGRADE YOUR TWELVE DATA API KEY TO ENTERPRISE. "
                                f"USING A LOWER TIER WILL CAUSE YOU TO FALL OUT OF CONSENSUS. Failed to subscribe to "
                                f"websocket symbols: {event['fails']}")
        elif event['event'] == 'heartbeat':
            if event['status'] != 'ok':
                bt.logging.error(f"Heartbeat failed: {event}. If this doesn't resolve, websocket will reset."
                                 f" If errors persist, contact the team")
        else:
            bt.logging.error(f"Received unexpected event: {event}")

    def internet_reachable(self, timeout=5):
        try:
            response = requests.head("https://www.google.com", timeout=timeout)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False

    def _should_reset_websocket(self):
        if not self.internet_reachable():
            return False

        # Give a chance for some prices to come in
        now = time.time()
        if now - self.init_time < 180:
            return False

        if not latest_websocket_prices:
            return True

        # You get 3 minutes to get any websocket data or you're out
        last_ping = max(timestamp for timestamp in last_websocket_ping_time_s.values())
        if now - last_ping > 180:
            return True

    def get_websocket_lag_for_trade_pair_s(self, trade_pair: str):
        if trade_pair in last_websocket_ping_time_s:
            return time.time() - last_websocket_ping_time_s.get(trade_pair, 0)
        return None

    def spill_price_history(self):
        # Write the price history to disk in a format that will let us plot it
        filename = f"price_history.json"
        with open(filename, 'w') as f:
            json.dump(trade_pair_to_price_history, f)

    def _websocket_heartbeat(self):
        global WS
        while True:
            if WS:
                WS.heartbeat()
            time.sleep(10)
            if self._should_reset_websocket():
                self._reset_websocket()
            if DEBUG:
                self.spill_price_history()
                self.debug_log()

    def debug_log(self):
        trade_pairs_to_track = [k for k, v in last_websocket_ping_time_s.items()]
        for tp in trade_pairs_to_track:
            lag = self.get_websocket_lag_for_trade_pair_s(tp)
            if tp not in self.trade_pair_to_longest_seen_lag:
                self.trade_pair_to_longest_seen_lag[tp] = lag
            else:
                if lag > self.trade_pair_to_longest_seen_lag[tp]:
                    self.trade_pair_to_longest_seen_lag[tp] = lag
        # log how long it has been since the last ping
        formatted_lags = {tp: f"{lag:.2f}" for tp, lag in self.trade_pair_to_longest_seen_lag.items()}
        bt.logging.warning(f"Worst lags seen: {formatted_lags}")
        # Log the last time since websocket ping
        formatted_lags = {tp: f"{time.time() - timestamp:.2f}" for tp, timestamp in
                          last_websocket_ping_time_s.items()}
        bt.logging.warning(f"Last websocket pings: {formatted_lags}")
        # Log the prices
        formatted_prices = {tp: f"{price:.2f}" for tp, price in latest_websocket_prices.items()}
        bt.logging.warning(f"Latest websocket prices: {formatted_prices}")
        # Log which trade pairs are likely in closed markets
        trade_pair_is_closed = {}
        for trade_pair in TradePair:
            if self.trade_pair_market_likely_closed(trade_pair):
                trade_pair_is_closed[trade_pair.trade_pair] = True

        bt.logging.warning(f"Market likely closed for {trade_pair_is_closed}")


    def _reset_websocket(self):
        global WS
        self.n_resets += 1
        bt.logging.info(f"last_websocket_ping_time_s: n = {len(last_websocket_ping_time_s)}")
        bt.logging.info(f"last_rest_update_time_s: n = {len(last_rest_update_time_s)}")
        bt.logging.info(f"last_rest_datetime_received: n = {len(last_rest_datetime_received)}")
        bt.logging.info(f"latest_websocket_prices: n = {len(latest_websocket_prices)}")
        new_ws = self.td.websocket(on_event=self._on_event)
        pairs = []
        for trade_pair in TradePair:
            s = trade_pair.trade_pair
            #if trade_pair.is_crypto:
            #    s += ':Coinbase Pro'
            pairs.append(s)
        new_ws.subscribe(pairs)
        new_ws.connect()
        old_ws = WS
        WS = new_ws
        if old_ws:
            old_ws.disconnect()

        # Only
    def _fetch_data_rest(self, symbols, interval, output_size):
        ts = self.td.time_series(symbol=symbols, interval=interval, outputsize=output_size)
        response = ts.as_json()
        return response

    def get_closes_rest(
        self,
        trade_pairs: List[TradePair],
        output_size: int = 1,
        interval: str = "1min",
    ):
        if len(trade_pairs) == 1:
            return {trade_pairs[0]: self.get_close_rest(trade_pairs[0], output_size, interval)}

        trade_pair_values = [trade_pair.trade_pair for trade_pair in trade_pairs]
        stringified_trade_pairs = ",".join(map(str, trade_pair_values))

        all_trade_pair_closes = {}

        data = self._fetch_data_rest(stringified_trade_pairs, interval, output_size)
        debug = {}
        for k, v in data.items():
            for c in v:
                debug[self.trade_pair_lookup[k]] = c
                all_trade_pair_closes[self.trade_pair_lookup[k]] = float(c["close"])
        self.update_last_rest_update_time(debug)
        return all_trade_pair_closes

    def update_last_rest_update_time(self, data):
        bt.logging.trace(f"update_last_rest_update_time received data: {[(k.trade_pair, v) for k, v in data.items()]}")
        for trade_pair, d in data.items():
            symbol = trade_pair.trade_pair
            previous_datetime = last_rest_datetime_received.get(symbol, '')
            last_rest_datetime_received[symbol] = d["datetime"]
            if previous_datetime != last_rest_datetime_received[symbol]:
                last_rest_update_time_s[symbol] = time.time()
                bt.logging.trace(
                    f"Updated last_rest_update_time_s for {trade_pair.trade_pair} at {last_rest_update_time_s[symbol]}")


    def get_close_rest(
        self,
        trade_pair: TradePair,
        output_size: int = 1,
        interval: str = "1min"
    ):
        data = self._fetch_data_rest(trade_pair.trade_pair, interval, output_size)
        self.update_last_rest_update_time({trade_pair: data[0]})
        return float(data[0]["close"])

    def get_closes_websocket(self, trade_pairs: List[TradePair]):
        closes = {}
        for trade_pair in trade_pairs:
            symbol = trade_pair.trade_pair
            if symbol in latest_websocket_prices:
                price = latest_websocket_prices[symbol]
                lag = self.get_websocket_lag_for_trade_pair_s(symbol)
                is_stale = lag > self.trade_pair_category_to_longest_allowed_lag[trade_pair.trade_pair_category]
                if is_stale:
                    bt.logging.warning(f"Found stale websocket data for {trade_pair.trade_pair}. Lag: {lag} seconds. "
                                       f"Max allowed lag for category: "
                                       f"{self.trade_pair_category_to_longest_allowed_lag[trade_pair.trade_pair_category]} seconds."
                                       f"Ignoring this data.")
                else:
                    closes[trade_pair] = price

        return closes

    def get_close_websocket(self, trade_pair: TradePair):
        symbol = trade_pair.trade_pair
        if symbol in latest_websocket_prices and symbol in last_websocket_ping_time_s:
            price = latest_websocket_prices[symbol]
            timestamp = last_websocket_ping_time_s.get(symbol, 0)
            max_allowed_lag = self.trade_pair_category_to_longest_allowed_lag[trade_pair.trade_pair_category]
            is_stale = time.time() - timestamp > max_allowed_lag
            if is_stale:
                bt.logging.info(f"Found stale websocket data for {trade_pair.trade_pair}. Lag: {time.time() - timestamp} "
                                f"seconds. Max allowed lag for category: {max_allowed_lag} seconds. Ignoring this data.")
            else:
                return price

        return None

    def get_close(self, trade_pair: TradePair):
        ans = self.get_close_websocket(trade_pair)
        if not ans:
            bt.logging.info(f"Fetching stale trade pair using REST: {trade_pair}")
            ans = self.get_close_rest(trade_pair)
            bt.logging.info(f"Received REST data for {trade_pair.trade_pair}: {ans}")

        bt.logging.info(f"Using websocket data for {trade_pair.trade_pair}")
        return ans

    def get_closes(self, trade_pairs: List[TradePair]):
        closes = self.get_closes_websocket(trade_pairs)
        missing_trade_pairs = []
        for tp in trade_pairs:
            if tp not in closes:
                missing_trade_pairs.append(tp)
        if closes:
            debug = {k.trade_pair: v for k, v in closes.items()}
            bt.logging.info(f"Received websocket data: {debug}")

        if missing_trade_pairs:
            rest_closes = self.get_closes_rest(missing_trade_pairs)
            debug = {k.trade_pair: v for k, v in rest_closes.items()}
            bt.logging.info(f"Received stale/websocket-less data using REST: {debug}")
            closes.update(rest_closes)

        return closes

    def get_close_at_date(self, trade_pair: TradePair, date: str):
        symbol = trade_pair.trade_pair
        ts = self.td.time_series(symbol=symbol, interval='1min', outputsize=1, date=date)
        response = ts.as_json()
        return float(response[0]["close"]), response[0]["datetime"]

    def get_range_of_closes(self, trade_pair, start_date: str, end_date: str):
        ts = self.td.time_series(symbol=trade_pair, interval='1min', start_date=start_date, end_date=end_date, outputsize=5000)
        response = ts.as_json()
        closes = [(d['datetime'], float(d["close"])) for d in response]
        return closes



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()

    # Initialize client
    twelve_data = TwelveDataService(api_key=secrets['twelvedata_apikey'])
    data = None
    #data = twelve_data.get_close_at_date(TradePair.SPX, '2024-04-01')
    # read in the cached data
    with open('/Users/jbonilla/Documents/prop-net/price_history_2.json', 'r') as f:
        data = json.load(f)

    for trade_pair, websocket_price_history in data.items():
        print(trade_pair)
        websocket_times = [datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S') for x in websocket_price_history]
        websocket_prices = [x[1] for x in websocket_price_history]

        min_time = min(websocket_price_history, key=lambda x: x[0])[0]
        max_time = max(websocket_price_history, key=lambda x: x[0])[0]
        closes_historical = twelve_data.get_range_of_closes(trade_pair, min_time, max_time)
        historical_times = [datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S') for x in closes_historical]
        historical_prices = [x[1] for x in closes_historical]

        plt.figure(figsize=(10, 6))
        plt.plot(historical_times, historical_prices, label='Historical Data', marker='o', linestyle='-', markersize=4)
        plt.plot(websocket_times, websocket_prices, label='Websocket Data', marker='x', linestyle='--', markersize=4)
        plt.title(f'Trade Pair: {trade_pair}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Step 5: Display or Save the Plot
        plt.show()


    assert 0


    ######################
    def on_event(e):
        # do whatever is needed with data
        print(e)

    td = TDClient(apikey=secrets['twelvedata_apikey'])
    ws = td.websocket(symbols="BTC/USD", on_event=on_event)
    ws.subscribe(['ETH/BTC', 'AAPL'])
    ws.connect()

    time.sleep(1000000)
    raise Exception('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    ############################

    for i, secret in enumerate([secrets['twelvedata_apikey'], secrets['twelvedata_apikey2']]):
        if i == 0:
            print("USING ENTERPRISE TIER")
        else:
            print("USING FREE TIER")
        twelve_data = TwelveDataService(api_key=secret)

        #twelve_data.get_closes_rest([TradePair.BTCUSD, TradePair.SPX, TradePair.ETHUSD])
        #time.sleep(70)
        #twelve_data.get_closes_rest([TradePair.BTCUSD, TradePair.SPX, TradePair.ETHUSD])
        #assert 0


        time.sleep(90)
        trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SPX, TradePair.GBPUSD]
        print("-----------------REST-----------------")
        ans_rest = twelve_data.get_closes_rest(trade_pairs)
        for k, v in ans_rest.items():
            print(f"{k.trade_pair_id}: {v}")
        print("-----------------WEBSOCKET-----------------")
        ans_ws = twelve_data.get_closes_rest(trade_pairs)
        for k, v in ans_ws.items():
            print(f"{k.trade_pair_id}: {v}")
        print("-----------------Normal-----------------")
        ans_n = twelve_data.get_closes(trade_pairs)
        for k, v in ans_n.items():
            print(f"{k.trade_pair_id}: {v}")
        print("-----------------Done-----------------")