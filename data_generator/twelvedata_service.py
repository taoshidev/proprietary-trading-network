import json
import threading
from collections import defaultdict
from typing import List, Tuple
import matplotlib.pyplot as plt

from data_generator.base_data_service import BaseDataService, TWELVEDATA_PROVIDER_NAME
from time_util.time_util import TimeUtil
from vali_config import TradePair, TradePairCategory
import time
from twelvedata import TDClient

from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt
import requests

DEBUG = 0


class TwelveDataService(BaseDataService):

    def __init__(self, api_key, disable_ws=False):
        trade_pair_category_to_longest_allowed_lag_s = {TradePairCategory.CRYPTO: 60, TradePairCategory.FOREX: 60,
                                                             TradePairCategory.INDICES: 60}
        timespan_to_ms = {'1min': 1000 * 60, '1h': 1000 * 60 * 60, '1day': 1000 * 60 * 60 * 24}
        super().__init__(trade_pair_category_to_longest_allowed_lag_s=trade_pair_category_to_longest_allowed_lag_s,
                         timespan_to_ms=timespan_to_ms, provider_name=TWELVEDATA_PROVIDER_NAME)
        self.WS = None
        self.disable_ws = disable_ws
        self.n_resets = 0
        self.init_time_ms = TimeUtil.now_in_millis()
        self._api_key = api_key
        self.td = TDClient(apikey=self._api_key)

        if disable_ws:
            self._heartbeat_thread = None
        else:
            self._reset_websocket()
            self._heartbeat_thread = threading.Thread(target=self._websocket_heartbeat)
            self._heartbeat_thread.daemon = True
            self._heartbeat_thread.start()

        self.trade_pair_to_longest_seen_lag_s = {}


    def _on_event(self, event):
        """
        Received price event: {'event': 'price', 'symbol': 'GDAXI', 'currency': 'EUR', 'exchange': 'XETR',
        'mic_code': 'XETR', 'type': 'Index', 'timestamp': 1713275588, 'price': 17727.85, 'bid': 17727.1,
        'ask': 17728.6, 'day_volume': 35008912}

        """
        if event['event'] == 'price':
            symbol = event['symbol']
            if symbol == 'GSPC':  # The realtime version of SPX (lol)
                symbol = 'SPX'
            price = float(event['price'])
            timestamp_ms = int(event['timestamp']) * 1000
            lag_time_ms = TimeUtil.now_in_millis() - timestamp_ms
            if lag_time_ms < 0:
                bt.logging.error(f"Received TD websocket data in the future {symbol}. Ignoring this data.")
                return
            formatted_event_price = TimeUtil.millis_to_formatted_date_str(timestamp_ms)
            prev_event = self.latest_websocket_events.get(symbol)
            prev_event_time_ms = prev_event.start_ms + prev_event.timespan_ms if prev_event else None
            if prev_event is None or timestamp_ms > prev_event_time_ms:
                #print(f"Received valid price event: {event} for time {formatted_event_price}")
                self.latest_websocket_events[symbol] = PriceSource(source=TWELVEDATA_PROVIDER_NAME + '_ws', timespan_ms=0, open=price,
                                                              close=price, vwap=None, high=price, low=price, start_ms=timestamp_ms,
                                                              websocket=True, lag_ms=lag_time_ms, volume=None)
                self.trade_pair_to_recent_events[symbol].add_event(self.latest_websocket_events[symbol], False)
            #else:
                #formatted_disk_time = TimeUtil.millis_to_formatted_date_str(prev_event_time_ms)
                #print(f"Received TD websocket data in the past {symbol}. Disk time {formatted_disk_time} "
                #      f"event time {formatted_event_price} Ignoring this data.")
            self.n_events_global += 1
            if DEBUG:
                formatted_time = TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())
                self.trade_pair_to_price_history[symbol].append((formatted_time, price))
                history_size = sum(len(v) for v in self.trade_pair_to_price_history.values())
                bt.logging.info("History Size: " + str(history_size))
                bt.logging.info(f" n_events_global: {self.n_events_global}")

        elif event['event'] == 'subscribe-status':
            if event['status'] == 'ok':
                bt.logging.info(f"TD Websocket Subscribed to symbols: {event['success']}")
            else:
                bt.logging.error(f"YOU LIKELY NEED TO UPGRADE YOUR TWELVE DATA API KEY TO PRO. "
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

    def _websocket_heartbeat(self):
        time_of_last_debug_log = 0
        time_of_last_heartbeat = 0
        while True:
            now = time.time()
            if self.WS and now - time_of_last_heartbeat >= 10:
                self.WS.heartbeat()
                time_of_last_heartbeat = now
            if self._should_reset_websocket():
                self._reset_websocket()
            if time.time() - time_of_last_debug_log >= 60:
                self.debug_log()
                time_of_last_debug_log = time.time()
            if DEBUG:
                self.spill_price_history()
            time.sleep(1) # avoid tight loop


    def _reset_websocket(self):
        self.n_resets += 1
        bt.logging.info(f"{TWELVEDATA_PROVIDER_NAME} latest_websocket_prices: n = {len(self.latest_websocket_events)}")
        new_ws = self.td.websocket(on_event=self._on_event)
        pairs = ['GSPC']
        for trade_pair in TradePair:
            s = trade_pair.trade_pair
            #if trade_pair.is_crypto:
            #    s += ':Coinbase Pro'
            pairs.append(s)
        new_ws.subscribe(pairs)
        new_ws.connect()
        old_ws = self.WS
        self.WS = new_ws
        if old_ws:
            old_ws.disconnect()

        # Only
    def _fetch_data_rest(self, symbols, interval, output_size, date_str=None, start_date_str=None, end_date_str=None) \
            -> dict[str: PriceSource] | None:
        """
         Response CADJPY: ({'datetime': '2024-04-10 10:50:00', 'open': '111.82000', 'high':
         '111.83000', 'low': '111.81000', 'close': '111.82650'},)

         Response: BTCUSD ({'datetime': '2024-04-10 10:50:00', 'open': '68765.48000', 'high': '68798.82000',
          'low': '68765.48000', 'close': '68775.81000'},)

          Response: DJI ({'datetime': '2024-04-10 10:50:00', 'open': '38556.23047', 'high': '38559.35156',
          'low': '38551.42969', 'close': '38555.89844', 'volume': '420023'},)
        """
        #n_symbols = len(symbols.split(','))
        def parse_key_from_response(response, key):
            if key in response and response[key]:
                return float(response[key])
            return None

        args = {'symbol': symbols, 'interval': interval, 'outputsize': output_size, 'timezone': 'UTC'}
        if date_str:
            args['date'] = date_str
        if start_date_str:
            args['start_date'] = start_date_str
        if end_date_str:
            args['end_date'] = end_date_str
        ts = self.td.time_series(**args)
        response = ts.as_json()
        if not response:
            return None

        if len(response) == 1 and isinstance(response, tuple):
            response = {symbols: response}

        ret = defaultdict(list)
        now_ms = TimeUtil.now_in_millis()
        for k, dat in response.items():
            for r in dat:
                close = parse_key_from_response(r, 'close')
                start_time_ms = TimeUtil.formatted_date_str_to_millis(r['datetime'])
                end_time_ms = start_time_ms + self.timespan_to_ms[interval]
                lag_ms = now_ms - end_time_ms
                if lag_ms < 0:
                    bt.logging.error(f"Received {self.provider_name} REST data in the future {symbols}. Ignoring this data.")
                    continue

                event = PriceSource(source=TWELVEDATA_PROVIDER_NAME + '_rest',
                                    timespan_ms=self.timespan_to_ms[interval],
                                    open=parse_key_from_response(r, 'open'),
                                    close=close,
                                    vwap=None,  # Not provided by TD
                                    high=parse_key_from_response(r, 'high'),
                                    low=parse_key_from_response(r, 'low'),
                                    start_ms=start_time_ms,
                                    end_ms=end_time_ms,
                                    websocket=False,
                                    lag_ms=lag_ms,
                                    volume=parse_key_from_response(r, 'volume'))

                if start_date_str and end_date_str:
                    ret[k].append(event)
                else:
                    if ret[k]:
                        bt.logging.warning(f"{self.provider_name} Received unexpected number of REST events [{len(response)}] for {k}: {response}. Using the latest one...")
                    ret[k] = [event]

        return ret

    def get_closes_rest(
        self,
        trade_pairs: List[TradePair],
        output_size: int = 1,
        interval: str = "1min",
    ) -> dict[TradePair, PriceSource]:
        if len(trade_pairs) == 1:
            return {trade_pairs[0]: self.get_close_rest(trade_pairs[0], output_size, interval)}

        trade_pair_values = [trade_pair.trade_pair for trade_pair in trade_pairs]
        stringified_trade_pairs = ",".join(map(str, trade_pair_values))

        all_trade_pair_events = {}

        data = self._fetch_data_rest(stringified_trade_pairs, interval, output_size)
        for k, events in data.items():
            all_trade_pair_events[self.trade_pair_lookup[k]] = events[0]
        return all_trade_pair_events

    def _should_reset_websocket(self):
        if not self.internet_reachable():
            return False

        # Give a chance for some prices to come in
        now_ms = TimeUtil.now_in_millis()
        if now_ms - self.init_time_ms < 180000:
            return False

        if not self.latest_websocket_events:
            return True

        # You get 3 minutes to get any websocket data or you're out
        last_ping_ms = max(x.end_ms for x in self.latest_websocket_events.values())
        if now_ms - last_ping_ms > 180000:
            return True

    def get_close_rest(
        self,
        trade_pair: TradePair,
        output_size: int = 1,
        interval: str = "1min"
    ) -> PriceSource | None:
        events_dict = self._fetch_data_rest(trade_pair.trade_pair, interval, output_size)
        if not events_dict:
            return None
        if trade_pair.trade_pair not in events_dict or not events_dict[trade_pair.trade_pair]:
            return None
        event = events_dict[trade_pair.trade_pair][0]
        return event


    def get_close_at_date(self, trade_pair: TradePair, timestamp_ms: int) -> Tuple[float, int] | Tuple[None, None]:

        input_time_formatted = TimeUtil.millis_to_formatted_date_str(timestamp_ms)
        timespan = '1min'
        events_dict = self._fetch_data_rest(trade_pair.trade_pair, interval=timespan, output_size=1, date_str=input_time_formatted)
        if not events_dict:
            return None
        #print(f'Response:', events_dict)
        events = events_dict.get(trade_pair.trade_pair)
        if not events:
            return None
        event = events[0]

        smallest_delta_ms = None
        price = None
        corresponding_date = None
        #print('Got response: ', response)
        def update_best_answer(p, t_ms):
            nonlocal smallest_delta_ms, price, corresponding_date
            delta_ms = abs(timestamp_ms - t_ms)
            if smallest_delta_ms is None or delta_ms < smallest_delta_ms:
                price = p
                smallest_delta_ms = delta_ms
                corresponding_date = TimeUtil.millis_to_formatted_date_str(t_ms)

        update_best_answer(event.open, event.start_ms)
        update_best_answer(event.close, event.end_ms)


        #print(f"Input time: {input_time_formatted}, Response time: {t_str}")

        smallest_delta_s = smallest_delta_ms / 1000 if smallest_delta_ms is not None else None
        #print('TD Delta time in s: ', smallest_delta_s, 'Reported time', corresponding_date, 'Input date', input_time_formatted)

        return price, smallest_delta_ms


    def get_range_of_closes(self, trade_pair: str, start_date: str, end_date: str):
        events_dict = self._fetch_data_rest(trade_pair, '1min', 5000, start_date_str=start_date, end_date_str=end_date)

        events = events_dict[trade_pair]
        closes = [(TimeUtil.millis_to_formatted_date_str(event.close_ms), event.close) for event in events]
        return closes



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()

    # Initialize client
    twelve_data = TwelveDataService(api_key=secrets['twelvedata_apikey'])

    now = TimeUtil.now_in_millis()
    while True:
        # use trade_pair_to_recent_events to get the closest event to "now"
        for tp in TradePair:
            symbol = tp.trade_pair
            closest_event = twelve_data.trade_pair_to_recent_events[symbol].get_closest_event(now)
            n_events = twelve_data.trade_pair_to_recent_events[symbol].count_events()
            delta_time_s = (now - closest_event.start_ms) / 1000.0 if closest_event else None
            print(f"Closest event to {TimeUtil.millis_to_formatted_date_str(now)} for {tp.trade_pair_id}: {closest_event}. Total_n_events: {n_events}. Lag (s): {delta_time_s}")
        time.sleep(10)

    #print(twelve_data.get_close_at_date(TradePair.CADCHF, TimeUtil.millis_to_formatted_date_str(1720130492000)))
    print(twelve_data.get_close_at_date(TradePair.DJI, 1712746241174))
    assert 0


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