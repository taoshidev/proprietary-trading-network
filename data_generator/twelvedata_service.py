import json
import os
import threading
from typing import List

import pytz

from vali_config import TradePair, TradePairCategory
import time
from twelvedata import TDClient
from datetime import datetime
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt
import requests

class TwelveDataService:

    def __init__(self, api_key):
        self.init_time = time.time()
        self._api_key = api_key
        self.td = TDClient(apikey=self._api_key)
        self.ws = None
        self.last_websocket_ping_time_s = self._load_from_disk('last_websocket_ping_time_s.json', {})
        self.last_rest_update_time_s = self._load_from_disk('last_rest_update_time_s.json', {})
        self.last_rest_datetime_received = self._load_from_disk('last_rest_datetime_received.json', {})
        self.latest_websocket_prices = self._load_from_disk('latest_websocket_prices.json', {})

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

    def _load_from_disk(self, filename, default_value):
        try:
            full_path = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + filename
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    return json.load(f)

            temp_filename = f"{filename}.tmp"
            full_path_temp = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + temp_filename
            if os.path.exists(full_path_temp):
                with open(full_path_temp, 'r') as f:
                    ans = json.load(f)
                    # Write to disk with the correct filename
                    self._save_to_disk(filename, ans, skip_temp_write=True)
        except Exception as e:
            bt.logging.error(f"Error loading {filename} from disk: {e}")

        return default_value

    def _save_to_disk(self, filename, data, skip_temp_write=False):
        try:
            temp_filename = f"{filename}.tmp"
            full_path_temp = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + temp_filename
            full_path_orig = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + filename
            if skip_temp_write:
                with open(full_path_orig, 'w') as f:
                    json.dump(data, f)
            else:
                with open(full_path_temp, 'w') as f:
                    json.dump(data, f)
                os.replace(full_path_temp, full_path_orig)
        except Exception as e:
            bt.logging.error(f"Error saving {filename} to disk: {e}")

    def _periodic_save(self):
        while True:
            time.sleep(60)  # Save every 60 seconds
            self._save_to_disk('last_websocket_ping_time_s.json', self.last_websocket_ping_time_s)
            self._save_to_disk('last_rest_update_time_s.json', self.last_rest_update_time_s)
            self._save_to_disk('last_rest_datetime_received.json', self.last_rest_datetime_received)
            self._save_to_disk('latest_websocket_prices.json', self.latest_websocket_prices)

    def trade_pair_market_likely_closed(self, trade_pair: TradePair):
        # Check if the last websocket ping happened within the last 15 minutes
        symbol = trade_pair.trade_pair
        if symbol in self.last_websocket_ping_time_s:
            return time.time() - self.last_websocket_ping_time_s[symbol] > 900
        # Check if the last REST ping succeeded within the last 15 minutes (validators who don't have enterprise tier...)
        if symbol in self.last_rest_update_time_s:
            return time.time() - self.last_rest_update_time_s[symbol] > 900
        return True


    def _on_event(self, event):
        if event['event'] == 'price':
            #bt.logging.info(f"Received price event: {event}")
            symbol = event['symbol']
            price = float(event['price'])
            self.latest_websocket_prices[symbol] = price
            self.last_websocket_ping_time_s[symbol] = time.time()
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

        if not self.latest_websocket_prices:
            return True

        # You get 3 minutes to get any websocket data or you're out
        last_ping = max(timestamp for timestamp in self.last_websocket_ping_time_s.values())
        if now - last_ping > 180:
            return True

    def get_websocket_lag_for_trade_pair_s(self, trade_pair: str):
        if trade_pair in self.last_websocket_ping_time_s:
            return time.time() - self.last_websocket_ping_time_s.get(trade_pair, 0)
        return None

    def _websocket_heartbeat(self):
        while True:
            if self.ws:
                self.ws.heartbeat()
            time.sleep(10)
            if self._should_reset_websocket():
                self._reset_websocket()
            #self.debug_log()

    def debug_log(self):
        trade_pairs_to_track = [k for k, v in self.last_websocket_ping_time_s.items()]
        for tp in trade_pairs_to_track:
            lag = self.get_websocket_lag_for_trade_pair_s(tp)
            if tp not in self.trade_pair_to_longest_seen_lag:
                self.trade_pair_to_longest_seen_lag[tp] = lag
            else:
                if lag > self.trade_pair_to_longest_seen_lag[tp]:
                    self.trade_pair_to_longest_seen_lag[tp] = lag
            # log how long it has been since the last ping
        bt.logging.error(f"Worst lags seen: {self.trade_pair_to_longest_seen_lag}")
        # Log which trade pairs are likely in closed markets
        trade_pair_is_closed = {}
        for trade_pair in TradePair:
            trade_pair_is_closed[trade_pair.trade_pair] = self.trade_pair_market_likely_closed(trade_pair)

        bt.logging.info(f"Market likely closed for {trade_pair_is_closed}")


    def _reset_websocket(self):
        bt.logging.info(f"last_websocket_ping_time_s: n = {len(self.last_websocket_ping_time_s)}")
        bt.logging.info(f"last_rest_update_time_s: n = {len(self.last_rest_update_time_s)}")
        bt.logging.info(f"last_rest_datetime_received: n = {len(self.last_rest_datetime_received)}")
        bt.logging.info(f"latest_websocket_prices: n = {len(self.latest_websocket_prices)}")
        new_ws = self.td.websocket(on_event=self._on_event)
        pairs = []
        for trade_pair in TradePair:
            s = trade_pair.trade_pair
            #if trade_pair.is_crypto:
            #    s += ':Coinbase Pro'
            pairs.append(s)
        new_ws.subscribe(pairs)
        new_ws.connect()
        old_ws = self.ws
        self.ws = new_ws
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
            previous_datetime = self.last_rest_datetime_received.get(symbol, '')
            self.last_rest_datetime_received[symbol] = d["datetime"]
            if previous_datetime != self.last_rest_datetime_received[symbol]:
                self.last_rest_update_time_s[symbol] = time.time()
                bt.logging.trace(
                    f"Updated last_rest_update_time_s for {trade_pair.trade_pair} at {self.last_rest_update_time_s[symbol]}")


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
            if symbol in self.latest_websocket_prices:
                price = self.latest_websocket_prices[symbol]
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
        if symbol in self.latest_websocket_prices and symbol in self.last_websocket_ping_time_s:
            price = self.latest_websocket_prices[symbol]
            timestamp = self.last_websocket_ping_time_s.get(symbol, 0)
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



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()

    from twelvedata import TDClient

    # Initialize client
    twelve_data = TwelveDataService(api_key=secrets['twelvedata_apikey'])

    data = twelve_data.get_close_at_date(TradePair.SPX, '2024-04-01')
    print(data)

    assert 0

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