import threading
import traceback
import json
from multiprocessing import Process

import requests
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_generator.base_data_service import BaseDataService, TIINGO_PROVIDER_NAME, exception_handler_decorator
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
import time

from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource

from tiingo import TiingoClient#, TiingoWebsocketClient

from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker

DEBUG = 0
TIINGO_COINBASE_EXCHANGE_STR = 'gdax'

class TiingoDataService(BaseDataService):

    def __init__(self, api_key, disable_ws=False, ipc_manager=None):
        self.init_time = time.time()
        self._api_key = api_key
        self.disable_ws = disable_ws

        super().__init__(provider_name=TIINGO_PROVIDER_NAME, ipc_manager=ipc_manager)

        self.MARKET_STATUS = None
        self.UNSUPPORTED_TRADE_PAIRS = (TradePair.SPX, TradePair.DJI, TradePair.NDX, TradePair.VIX, TradePair.FTSE, TradePair.GDAXI)

        self.config = {'api_key': self._api_key, 'session': True}
        self.TIINGO_CLIENT = None  # Instantiate the TiingoClient after process starts

        self.subscribe_message = {
            'eventName': 'subscribe',
            'authorization': self.config['api_key'],
            #see https://api.tiingo.com/documentation/websockets/iex > Request for more info
            'eventData': {
                'thresholdLevel':5
            }
        }

        # Start thread to refresh market status
        if disable_ws:
            self.websocket_manager_thread = None
        else:
            if ipc_manager:
                self.websocket_manager_thread = Process(target=self.websocket_manager, daemon=True)
            else:
                self.websocket_manager_thread = threading.Thread(target=self.websocket_manager, daemon=True)
            self.websocket_manager_thread.start()
            #time.sleep(3) # Let the websocket_manager_thread start

    def instantiate_not_pickleable_objects(self):
        self.TIINGO_CLIENT = TiingoClient(self.config)

    def run_pseudo_websocket(self, tpc: TradePairCategory):
        verbose = False
        POLLING_INTERVAL_S = 5
        if tpc == TradePairCategory.EQUITIES:
            desired_trade_pairs = [x for x in TradePair if x.is_equities]
        elif tpc == TradePairCategory.FOREX:
            desired_trade_pairs = [x for x in TradePair if x.is_forex]
        elif tpc == TradePairCategory.CRYPTO:
            desired_trade_pairs = [x for x in TradePair if x.is_crypto]
        else:
            raise ValueError(f'Unexpected trade pair category {tpc}')

        last_poll_time = 0

        while True:
            current_time = time.time()
            elapsed_time = current_time - last_poll_time

            if elapsed_time < POLLING_INTERVAL_S:
                time.sleep(1)
                continue

            trade_pairs_to_query = [pair for pair in desired_trade_pairs if self.is_market_open(pair)]
            price_sources = self.get_closes_rest(trade_pairs_to_query, verbose=verbose)

            for trade_pair, price_source in price_sources.items():
                price_source.websocket = True
                self.tpc_to_n_events[trade_pair.trade_pair_category] += 1
                self.process_ps_from_websocket(trade_pair, price_source)

            last_poll_time = current_time

            if verbose:
                elapsed_since_last_poll = time.time() - current_time
                print(f'Pseudo websocket update took {elapsed_since_last_poll:.2f} seconds for tpc {tpc}')


    def main_forex(self):
        #TiingoWebsocketClient(self.subscribe_message, endpoint="fx", on_msg_cb=self.handle_msg)
        self.run_pseudo_websocket(TradePairCategory.FOREX)
    def main_stocks(self):
        #TiingoWebsocketClient(self.subscribe_message, endpoint="iex", on_msg_cb=self.handle_msg)
        self.run_pseudo_websocket(TradePairCategory.EQUITIES)
    def main_crypto(self):
        #TiingoWebsocketClient(self.subscribe_message, endpoint="crypto", on_msg_cb=self.handle_msg)
        self.run_pseudo_websocket(TradePairCategory.CRYPTO)

    def handle_msg(self, msg):
        """
        {'service': 'iex', 'messageType': 'A', 'data': ['T', '2024-11-15T08:41:29.291307201-05:00', 1731678089291307201, 'srad', None, None, None, None, None, 17.58, 30, None, 1, 0, 1, 0]}

         CurrencyAgg(event_type='XAS', pair='ETH-USD', open=3084.37, close=3084.24, high=3084.37, low=3084.08,
         volume=0.99917426, vwap=3084.1452, start_timestamp=1713273981000, end_timestamp=1713273982000, avg_trade_size=0)
        """
        def msg_to_price_sources(m:dict, tp:TradePair) -> PriceSource | None:
            symbol = tp.trade_pair
            data = m['data']
            bid_price = 0
            ask_price = 0
            if tp.is_forex:
                assert len(data) == 8, data
                mode, ticker, date_str, bid_size, bid_price, mid_price, ask_size, ask_price = data
                start_timestamp_orig = TimeUtil.parse_iso_to_ms(date_str)
                start_timestamp = round(start_timestamp_orig, -3)  # round to nearest second which allows aggresssive filtering via dup logic
                #print(tp.trade_pair, start_timestamp_orig, start_timestamp)
                #print(f'Received forex message {symbol} price {new_price} time {TimeUtil.millis_to_formatted_date_str(start_timestamp)}')
                #print(m, symbol in self.trade_pair_to_recent_events, self.trade_pair_to_recent_events[symbol].timestamp_exists(start_timestamp))
                if self.using_ipc and symbol in self.trade_pair_to_recent_events_realtime and self.trade_pair_to_recent_events_realtime[symbol].timestamp_exists(start_timestamp):
                    self.trade_pair_to_recent_events_realtime[symbol].update_prices_for_median(start_timestamp, bid_price)
                    return None
                elif not self.using_ipc and symbol in self.trade_pair_to_recent_events and self.trade_pair_to_recent_events[symbol].timestamp_exists(start_timestamp):
                    self.trade_pair_to_recent_events[symbol].update_prices_for_median(start_timestamp, bid_price)
                    return None

                open = vwap = high = low = bid_price
            elif tp.is_crypto:
                mode, ticker, date_str, exchange, volume, price = data
                start_timestamp = TimeUtil.parse_iso_to_ms(date_str)
                start_timestamp = round(start_timestamp, -3)  # round to nearest second which allows aggresssive filtering via dup logic
                if mode != 'T':
                    print(f'Skipping crypto due to non-T mode {m}')
                    return None
                open = vwap = high = low = price

            elif tp.is_equities:
                (mode, date_str, timestamp_ns, ticker, bid_size, bid_price, mid_price, ask_price, ask_size, last_price,
                 last_size, halted, after_hours, intermarket_sweep, oddlot, nms) = data
                if mode != 'T':
                    #print(f'Skipping equities trade due to non-T mode {m}')
                    return None
                elif (after_hours and int(after_hours) == 1):
                    self.n_equity_events_skipped_afterhours += 1
                    return None
                # Exchange here is always iex!
                timestamp_ms = timestamp_ns // 1e6
                start_timestamp = round(timestamp_ms, -3)  # round to nearest second which allows aggresssive filtering via dup logic
                open = vwap = high = low = last_price

            elif tp.is_indices:
                raise Exception(f'TODO! {msg}')
            else:
                raise Exception(f'Unknown trade pair category {msg}')

            now_ms = TimeUtil.now_in_millis()
            price_source1 = PriceSource(
                source=f'{TIINGO_PROVIDER_NAME}_ws',
                timespan_ms=0,
                open=open,
                close=open,
                vwap=vwap,
                high=high,
                low=low,
                start_ms=start_timestamp,
                websocket=True,
                lag_ms=now_ms - start_timestamp,
                bid=bid_price,
                ask=ask_price
            )

            return price_source1

        try:
            msg = json.loads(msg)
            if not isinstance(msg, dict):
                # print(f'Non-dict message: {msg}')
                raise ValueError(f'Non-dict message: {msg}')
            if msg['messageType'] != 'A':
                # print(f'Non-A message type: {msg}')
                return
            if msg['service'] == 'fx':
                raw_tp = msg['data'][1]
                tp0, tp1 = raw_tp[0:3].upper(), raw_tp[3:].upper()
                ptn_trade_pair_id = f'{tp0}{tp1}'
                tp = TradePair.from_trade_pair_id(ptn_trade_pair_id)
            elif msg['service'] == 'iex':
                ptn_trade_pair_id = msg['data'][3].upper()
                tp = TradePair.from_trade_pair_id(ptn_trade_pair_id)
                #if tp:
                #    print(msg)
            elif msg['service'] == 'crypto_data':
                ptn_trade_pair_id = msg['data'][1].upper()
                tiingo_exchange_str = msg['data'][3].lower()
                tp = TradePair.from_trade_pair_id(ptn_trade_pair_id)
                if tp and tiingo_exchange_str == TIINGO_COINBASE_EXCHANGE_STR and tp.is_crypto:  # gbpusd shows up in crypto feed
                    #print(msg)
                    pass
                else:
                    return
            else:
                raise ValueError(f"Unknown service: {msg}")
            if not tp:
                return

            self.tpc_to_n_events[tp.trade_pair_category] += 1
            ps1 = msg_to_price_sources(msg, tp)

            self.process_ps_from_websocket(tp, ps1)

        except Exception as e:
            full_traceback = traceback.format_exc()
            # Slice the last 1000 characters of the traceback
            limited_traceback = full_traceback[-1000:]
            bt.logging.error(f"Failed to handle {TIINGO_PROVIDER_NAME} websocket message with error: {e}, "
                             f"type: {type(e).__name__}, traceback: {limited_traceback}")

    def process_ps_from_websocket(self, tp: TradePair, ps1: PriceSource):
        if ps1 is None:
            return

        symbol = tp.trade_pair
        # Reset the closed market price, indicating that a new close should be fetched after the current day's close
        self.closed_market_prices[tp] = None

        self.latest_websocket_events[symbol] = ps1
        if not self.using_ipc and symbol not in self.trade_pair_to_recent_events:
            self.trade_pair_to_recent_events[symbol] = RecentEventTracker()
        elif self.using_ipc and symbol not in self.trade_pair_to_recent_events_realtime:
            self.trade_pair_to_recent_events_realtime[symbol] = RecentEventTracker()

        if self.using_ipc:
            self.trade_pair_to_recent_events_realtime[symbol].add_event(ps1, tp.is_forex,
                                                           f"{self.provider_name}:{tp.trade_pair}")
        else:
            self.trade_pair_to_recent_events[symbol].add_event(ps1, tp.is_forex,
                                                           f"{self.provider_name}:{tp.trade_pair}")

        if DEBUG:
            formatted_time = TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())
            self.trade_pair_to_price_history[tp].append((formatted_time, ps1.close))


    def symbol_to_trade_pair(self, symbol: str):
        # Should work for indices and forex
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol)
        if tp:
            return tp
        # Should work for crypto. Anything else will return None
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol.replace('-', '/'))
        if not tp:
            raise ValueError(f"Unknown symbol: {symbol}")
        return tp

    def get_closes_rest(self, pairs: List[TradePair], verbose=False) -> dict[TradePair: PriceSource]:
        tp_equities = [tp for tp in pairs if tp.trade_pair_category == TradePairCategory.EQUITIES]
        tp_crypto = [tp for tp in pairs if tp.trade_pair_category == TradePairCategory.CRYPTO]
        tp_forex = [tp for tp in pairs if tp.trade_pair_category == TradePairCategory.FOREX]

        # Jobs to parallelize
        jobs = []
        if tp_equities:
            jobs.append((self.get_closes_equities, tp_equities, verbose))
        if tp_crypto:
            jobs.append((self.get_closes_crypto, tp_crypto, verbose))
        if tp_forex:
            jobs.append((self.get_closes_forex, tp_forex, verbose))

        if verbose:
            print(f'Running {len(jobs)} jobs {jobs}')

        tp_to_price = {}

        if len(jobs) == 0:
            return tp_to_price
        elif len(jobs) == 1:
            func, tp_list, verbose = jobs[0]
            return func(tp_list, verbose)

        # Use ThreadPoolExecutor for parallelization if there are multiple jobs
        with ThreadPoolExecutor() as executor:
            future_to_category = {
                executor.submit(func, tp_list, verbose): func.__name__
                for func, tp_list, verbose in jobs
            }
            for future in as_completed(future_to_category):
                price_result = future.result()
                if price_result:  # Only update if result is not None
                    tp_to_price.update(price_result)

        return tp_to_price

    @exception_handler_decorator()
    def get_closes_equities(self, trade_pairs: List[TradePair], verbose=False) -> dict[TradePair: PriceSource]:
        tp_to_price = {}
        if not trade_pairs:
            return tp_to_price
        assert all(tp.trade_pair_category == TradePairCategory.EQUITIES for tp in trade_pairs), trade_pairs

        if all(not self.is_market_open(tp) for tp in trade_pairs) and all(self.closed_market_prices.get(tp) for tp in trade_pairs):
            if verbose:
                print(f'All equities markets closed {trade_pairs}. Returning closed market prices. {self.closed_market_prices}')
            return {tp: self.closed_market_prices[tp] for tp in trade_pairs}

        def tickers_to_tiingo_iex_url(tickers: List[str]) -> str:
            return f"https://api.tiingo.com/iex/?tickers={','.join(tickers)}&token={self.config['api_key']}"

        url = tickers_to_tiingo_iex_url([self.trade_pair_to_tiingo_ticker(x) for x in trade_pairs])
        if verbose:
            print('hitting url', url)
        requestResponse = requests.get(url, headers={'Content-Type': 'application/json'})
        if requestResponse.status_code == 200:
            time_now_ms = TimeUtil.now_in_millis()
            for x in requestResponse.json():
                tp = TradePair.get_latest_trade_pair_from_trade_pair_id(x['ticker'].upper())
                data_time_ms = TimeUtil.parse_iso_to_ms(x['timestamp'])

                price = float(x['tngoLast'])
                bid_price = x['bidPrice']
                ask_price = x['askPrice']
                p_name = f'{TIINGO_PROVIDER_NAME}_rest'
                attempting_previous_close = not self.is_market_open(tp)
                if attempting_previous_close:
                    p_name += '_prev_close'
                tp_to_price[tp] = PriceSource(
                                    source=p_name,
                                    timespan_ms=0,
                                    open=price,
                                    close=price,
                                    vwap=price,
                                    high=price,
                                    low=price,
                                    start_ms=data_time_ms,
                                    websocket=False,
                                    lag_ms=time_now_ms - data_time_ms,
                                    bid=float(bid_price) if bid_price else 0,
                                    ask=float(ask_price) if ask_price else 0
                                )
                if attempting_previous_close and tp_to_price[tp]:
                    self.closed_market_prices[tp] = tp_to_price[tp]

                if verbose:
                    time_delta_s = (time_now_ms - data_time_ms) / 1000
                    time_delta_formatted_2_decimals = round(time_delta_s, 2)
                    print((tp.trade_pair_id, tp_to_price[tp], time_delta_formatted_2_decimals, x['timestamp'], x['tngoLast'], x))

        return tp_to_price

    @exception_handler_decorator()
    def get_closes_forex(self, trade_pairs: List[TradePair], verbose=False) -> dict:
        def tickers_to_tiingo_forex_url(tickers: List[str]) -> str:
            return f"https://api.tiingo.com/tiingo/fx/top?tickers={','.join(tickers)}&token={self.config['api_key']}"

        tp_to_price = {}
        if not trade_pairs:
            return tp_to_price

        assert all(tp.trade_pair_category == TradePairCategory.FOREX for tp in trade_pairs), trade_pairs
        if all(not self.is_market_open(tp) for tp in trade_pairs) and all(self.closed_market_prices.get(tp) for tp in trade_pairs):
            return {tp: self.closed_market_prices[tp] for tp in trade_pairs}

        url = tickers_to_tiingo_forex_url([self.trade_pair_to_tiingo_ticker(x) for x in trade_pairs])
        if verbose:
            print('hitting url', url)
        requestResponse = requests.get(url, headers={'Content-Type': 'application/json'})
        if requestResponse.status_code == 200:
            time_now_ms = TimeUtil.now_in_millis()
            for x in requestResponse.json():
                tp = TradePair.get_latest_trade_pair_from_trade_pair_id(x['ticker'].upper())
                bid_raw = x['bidPrice']
                ask_raw = x['askPrice']
                if not bid_raw:
                    continue
                if not ask_raw:
                    continue
                bid = float(bid_raw) if bid_raw else 0
                ask = float(ask_raw) if ask_raw else 0
                mid_price = (bid + ask) / 2.0
                data_time_ms = TimeUtil.parse_iso_to_ms(x['quoteTimestamp'])

                p_name = f'{TIINGO_PROVIDER_NAME}_rest'
                attempting_previous_close = not self.is_market_open(tp)
                if attempting_previous_close:
                    p_name += '_prev_close'
                tp_to_price[tp] = PriceSource(
                    source=p_name,
                    timespan_ms=0,
                    open=mid_price,
                    close=mid_price,
                    vwap=mid_price,
                    high=ask,
                    low=bid,
                    start_ms=data_time_ms,
                    websocket=False,
                    lag_ms=time_now_ms - data_time_ms,
                    bid=bid,
                    ask=ask
                )

                if attempting_previous_close and tp_to_price[tp]:
                    self.closed_market_prices[tp] = tp_to_price[tp]

                if verbose:
                    time_now_ms = TimeUtil.now_in_millis()
                    time_delta_s = (time_now_ms - data_time_ms) / 1000
                    time_delta_formatted_2_decimals = round(time_delta_s, 2)
                    print((tp.trade_pair_id, tp_to_price[tp], time_delta_formatted_2_decimals, x['quoteTimestamp'], x['bidPrice'], x))

        return tp_to_price

    @exception_handler_decorator()
    def get_closes_crypto(self, trade_pairs: List[TradePair], verbose=False) -> dict:
        tp_to_price = {}
        if not trade_pairs:
            return tp_to_price
        assert all(tp.trade_pair_category == TradePairCategory.CRYPTO for tp in trade_pairs), trade_pairs

        def tickers_to_crypto_url(tickers: List[str]) -> str:
            return f"https://api.tiingo.com/tiingo/crypto/top?tickers={','.join(tickers)}&token={self.config['api_key']}&exchanges={TIINGO_COINBASE_EXCHANGE_STR.upper()}"

        url = tickers_to_crypto_url([self.trade_pair_to_tiingo_ticker(x) for x in trade_pairs])
        if verbose:
            print('hitting url', url)
        requestResponse = requests.get(url, headers={'Content-Type': 'application/json'})
        if requestResponse.status_code == 200:
            now_ms = TimeUtil.now_in_millis()
            for y in requestResponse.json():
                ticker = y['ticker']
                if len(y['topOfBookData']) != 1:
                    print('Tiingo unexpected data', y)

                x = y['topOfBookData'][0]

                """
                'topOfBookData': [{'quoteTimestamp': '2024-11-20T21:21:12.287613+00:00', 'lastSaleTimestamp': '2024-11-20T21:21:13.293452+00:00', 'bidSize': 0.14791063, 'bidPrice': 94150.01, 'askSize': 10.4599248, 'askPrice': 94120.0, 'lastSize': 1.795e-05, 'lastSizeNotional': 1.689006327, 'lastPrice': 94095.06, 'bidExchange': 'GDAX', 'askExchange': 'KRAKEN', 'lastExchange': 'GDAX'}]}
                """
                data_time_exchange_ms = TimeUtil.parse_iso_to_ms(x['lastSaleTimestamp'])
                data_time_quote_ms = TimeUtil.parse_iso_to_ms(x['quoteTimestamp'])
                delta_ms_exchange = now_ms - data_time_exchange_ms
                delta_ms_quote = now_ms - data_time_quote_ms
                THRESHOLD_FRESH_MS = 15 * 10000
                last_exchange = x['lastExchange'].lower() if x['lastExchange'] else None
                bid_exchange = x['bidExchange'].lower() if x['bidExchange'] else None
                ask_exchange = x['askExchange'].lower() if x['askExchange'] else None
                bid_price = float(x['bidPrice']) if x['bidPrice'] else 0
                ask_price = float(x['askPrice']) if x['askPrice'] else 0

                if last_exchange == TIINGO_COINBASE_EXCHANGE_STR and delta_ms_exchange < THRESHOLD_FRESH_MS:
                    data_time_ms = data_time_exchange_ms
                    price = x['lastPrice']
                    exchange = last_exchange
                elif bid_exchange == TIINGO_COINBASE_EXCHANGE_STR and delta_ms_quote < THRESHOLD_FRESH_MS:
                    data_time_ms = data_time_quote_ms
                    price = x['bidPrice']
                    exchange = bid_exchange
                elif ask_exchange == TIINGO_COINBASE_EXCHANGE_STR and delta_ms_quote < THRESHOLD_FRESH_MS:
                    data_time_ms = data_time_quote_ms
                    price = x['askPrice']
                    exchange = ask_exchange

                elif last_exchange and delta_ms_exchange < THRESHOLD_FRESH_MS:
                    data_time_ms = data_time_exchange_ms
                    price = x['lastPrice']
                    exchange = last_exchange
                elif bid_exchange and delta_ms_quote < THRESHOLD_FRESH_MS:
                    data_time_ms = data_time_quote_ms
                    price = x['bidPrice']
                    exchange = bid_exchange
                elif ask_exchange and delta_ms_quote < THRESHOLD_FRESH_MS:
                    data_time_ms = data_time_quote_ms
                    price = x['askPrice']
                    exchange = ask_exchange

                elif last_exchange:
                    data_time_ms = data_time_exchange_ms
                    price = x['lastPrice']
                    exchange = last_exchange
                elif bid_exchange:
                    data_time_ms = data_time_quote_ms
                    price = x['bidPrice']
                    exchange = bid_exchange
                elif ask_exchange:
                    data_time_ms = data_time_quote_ms
                    price = x['askPrice']
                    exchange = ask_exchange
                else:
                    raise Exception('unexpected Tiingo data', y)


                tp = TradePair.get_latest_trade_pair_from_trade_pair_id(ticker.upper())
                price = float(price)


                p_name = f'{TIINGO_PROVIDER_NAME}_{exchange}_rest'
                tp_to_price[tp] = PriceSource(
                    source=p_name,
                    timespan_ms=0,
                    open=price,
                    close=price,
                    vwap=price,
                    high=price,
                    low=price,
                    start_ms=data_time_ms,
                    websocket=False,
                    lag_ms=now_ms - data_time_ms,
                    bid=bid_price,
                    ask=ask_price
                )

                if verbose:
                    time_delta_s = (now_ms - data_time_ms) / 1000
                    time_delta_formatted_2_decimals = round(time_delta_s, 2)
                    print((tp.trade_pair_id, tp_to_price[tp], time_delta_formatted_2_decimals, x['quoteTimestamp'], price, exchange, x))


        return tp_to_price

    def get_close_rest(
        self,
        trade_pair: TradePair,
        attempting_prev_close: bool = False,
    ) -> PriceSource | None:
        if trade_pair.trade_pair_category == TradePairCategory.EQUITIES:
            ans = self.get_closes_equities([trade_pair]).get(trade_pair)
        elif trade_pair.trade_pair_category == TradePairCategory.CRYPTO:
            ans = self.get_closes_crypto([trade_pair]).get(trade_pair)
        elif trade_pair.trade_pair_category == TradePairCategory.FOREX:
            ans = self.get_closes_forex([trade_pair]).get(trade_pair)
        else:
            raise ValueError(f"Unknown trade pair category {trade_pair}")

        return ans


    def trade_pair_to_tiingo_ticker(self, trade_pair: TradePair):
        return trade_pair.trade_pair_id.lower()

    def get_websocket_event(self, trade_pair: TradePair) -> PriceSource | None:
        symbol = trade_pair.trade_pair
        cur_event = self.latest_websocket_events.get(symbol)
        if not cur_event:
            return None

        timestamp_ms = cur_event.end_ms
        max_allowed_lag_s = self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
        lag_s = time.time() - timestamp_ms / 1000.0
        is_stale = lag_s > max_allowed_lag_s
        if is_stale:
            bt.logging.info(f"Found stale Tiingo websocket data for {trade_pair.trade_pair}. Lag_s: {lag_s} "
                            f"seconds. Max allowed lag for category: {max_allowed_lag_s} seconds. Ignoring this data.")
        return cur_event



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    tds = TiingoDataService(api_key=secrets['tiingo_apikey'], disable_ws=False)
    #time.sleep(100000)
    #assert 0
    target_timestamp_ms = 1715288502999

    client = TiingoClient({'api_key': secrets['tiingo_apikey']})
    crypto_price = client.get_crypto_top_of_book(['BTCUSD'])


    # forex_price = client.get_(ticker='USDJPY')# startDate='2021-01-01', endDate='2021-01-02', frequency='daily')
    #tds = TiingoDataService(secrets['tiingo_apikey'], disable_ws=True)
    tp_to_prices = tds.get_closes_rest([TradePair.BTCUSD, TradePair.USDJPY, TradePair.NVDA], verbose=True)

    assert 0, {x.trade_pair_id: y for x, y in tp_to_prices.items()}




