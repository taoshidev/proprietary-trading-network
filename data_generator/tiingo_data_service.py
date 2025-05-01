import asyncio
import traceback
import json
from datetime import timedelta

import requests
from typing import List, Optional, Dict
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

        # preserve your existing fields
        self.init_time = time.time()
        self._api_key = api_key
        self.disable_ws = disable_ws

        # call super to set up BaseDataService state
        super().__init__(provider_name=TIINGO_PROVIDER_NAME, ipc_manager=ipc_manager)

        # your existing Tiingo-specific settings
        self.MARKET_STATUS = None
        self.UNSUPPORTED_TRADE_PAIRS = (
            TradePair.SPX, TradePair.DJI, TradePair.NDX,
            TradePair.VIX, TradePair.FTSE, TradePair.GDAXI
        )
        self.config = {'api_key': self._api_key, 'session': True}
        # Initialize REST client after ws creation to avoid pickling error
        self.TIINGO_CLIENT: Optional[TiingoClient] = None

        self.subscribe_message = {
            'eventName': 'subscribe',
            'authorization': self.config['api_key'],
            'eventData': {'thresholdLevel': 5}
        }

        if not self.disable_ws:
            self.start_ws_async()

    async def instantiate_not_pickleable_objects(self):
        self.TIINGO_CLIENT = TiingoClient(self.config)

    async def _close_create_websocket_objects(self, tpc: TradePairCategory):
        # No persistent socket to close
        return

    async def _close_websocket_safely(self, client):
        return

    async def _run_pseudo_websocket(self, tpc: TradePairCategory):
        """Run pseudo-websocket polling with proper state tracking"""
        # If websockets (pseudo-WS polling) disabled, exit immediately
        if self.disable_ws:
            return

        POLLING_INTERVAL_S = 5
        verbose = False

        # pick your universe
        if tpc == TradePairCategory.EQUITIES:
            desired = [x for x in TradePair if x.is_equities]
        elif tpc == TradePairCategory.FOREX:
            desired = [x for x in TradePair if x.is_forex]
        elif tpc == TradePairCategory.CRYPTO:
            desired = [x for x in TradePair if x.is_crypto]
        else:
            raise ValueError(f"Unexpected trade pair category {tpc}")

        # Initialize or reset state tracking for this category
        reconnect_attempts = self.ws_state[tpc].get("reconnect_attempts", 0)
        max_backoff = 60.0  # Maximum backoff in seconds

        iteration = 0

        while True:
            # Use the monitoring_task flag to check if we should keep running
            if not self.ws_state[tpc].get("monitoring_task", True):
                bt.logging.info(f"Tiingo {tpc} polling stopped by monitoring task. Restarting.")
                # Reset the flag so we can continue
                self.ws_state[tpc]["monitoring_task"] = True
                # Wait based on reconnect attempts
                backoff = min(2.0 ** reconnect_attempts, max_backoff)
                await asyncio.sleep(backoff)
                reconnect_attempts += 1
                self.ws_state[tpc]["reconnect_attempts"] = reconnect_attempts
                continue

            iteration += 1
            now = time.time()

            # Determine which pairs to query
            to_query = [p for p in desired if self.is_market_open(p)]

            if not to_query:
                # Nothing to do right now, still update last_activity to show we're alive
                self.ws_state[tpc]["last_activity"] = now
                await asyncio.sleep(1)
                continue

            try:
                # Perform the polling
                poll_start = time.time()
                prices = await self.get_closes_rest(to_query, verbose=verbose)

                # Track successful poll as activity
                self.ws_state[tpc]["last_activity"] = time.time()

                # Process the results
                received_data = False
                for tp, ps in prices.items():
                    if ps is None:
                        continue

                    received_data = True
                    ps.websocket = True
                    self.tpc_to_n_events[tp.trade_pair_category] += 1
                    self.process_ps_from_websocket(tp, ps)

                # If we got data, reset the reconnect counter
                if received_data:
                    reconnect_attempts = 0
                    self.ws_state[tpc]["reconnect_attempts"] = 0

                if verbose:
                    took = time.time() - poll_start
                    print(f"[PseudoWS {tpc.name}] iteration={iteration} took {took:.2f}s")

                # Wait for the polling interval
                await asyncio.sleep(POLLING_INTERVAL_S)

            except Exception as e:
                bt.logging.error(f"Error in Tiingo {tpc} pseudo-websocket: {e}")
                # Only log full traceback occasionally
                current_hour = int(time.time()) // 3600
                last_tb_hour = getattr(self, f'_last_tb_hour_{tpc.name}', 0)
                if current_hour > last_tb_hour:
                    setattr(self, f'_last_tb_hour_{tpc.name}', current_hour)
                    bt.logging.error(f"Traceback: {traceback.format_exc()}")

                # Update reconnect state
                reconnect_attempts += 1
                self.ws_state[tpc]["reconnect_attempts"] = reconnect_attempts

                # Calculate backoff based on attempts
                backoff = min(2.0 ** reconnect_attempts, max_backoff)
                bt.logging.warning(f"Will retry Tiingo {tpc} polling in {backoff:.1f} seconds...")

                # Wait before retrying
                await asyncio.sleep(backoff)

    async def main_stocks(self):
        await self._run_pseudo_websocket(TradePairCategory.EQUITIES)

    async def main_forex(self):
        await self._run_pseudo_websocket(TradePairCategory.FOREX)

    async def main_crypto(self):
        await self._run_pseudo_websocket(TradePairCategory.CRYPTO)

    def handle_msg(self, msg):
        """
        {'service': 'iex', 'messageType': 'A', 'data': ['T', '2024-11-15T08:41:29.291307201-05:00', 1731678089291307201, 'srad', None, None, None, None, None, 17.58, 30, None, 1, 0, 1, 0]}

         CurrencyAgg(event_type='XAS', pair='ETH-USD', open=3084.37, close=3084.24, high=3084.37, low=3084.08,
         volume=0.99917426, vwap=3084.1452, start_timestamp=1713273981000, end_timestamp=1713273982000, avg_trade_size=0)
        """
        def msg_to_price_sources(m: dict, tp: TradePair) -> PriceSource | None:
            symbol = tp.trade_pair
            data = m['data']
            bid_price = 0
            ask_price = 0
            if tp.is_forex:
                assert len(data) == 8, data
                mode, ticker, date_str, bid_size, bid_price, mid_price, ask_size, ask_price = data
                start_timestamp_orig = TimeUtil.parse_iso_to_ms(date_str)
                start_timestamp = round(start_timestamp_orig, -3)  # round to nearest second which allows aggresssive filtering via dup logic
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
            elif msg['service'] == 'crypto_data':
                ptn_trade_pair_id = msg['data'][1].upper()
                tiingo_exchange_str = msg['data'][3].lower()
                tp = TradePair.from_trade_pair_id(ptn_trade_pair_id)
                if tp and tiingo_exchange_str == TIINGO_COINBASE_EXCHANGE_STR and tp.is_crypto:  # gbpusd shows up in crypto feed
                    pass
                else:
                    return
            else:
                raise ValueError(f"Unknown service: {msg}")
            if not tp:
                return

            self.ws_state[tp.trade_pair_category]["last_activity"] = time.time()
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
        """Process a price source from websocket with activity tracking"""
        if ps1 is None:
            return

        # Update the activity timestamp for this category
        tpc = tp.trade_pair_category
        self.ws_state[tpc]["last_activity"] = time.time()

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

    async def get_closes_rest(
        self,
        pairs: List[TradePair],
        verbose: bool = False
    ) -> Dict[TradePair, PriceSource]:
        """
        Parallel async calls to get_closes_equities/crypto/forex.
        """
        # 1) Ensure the Tiingo client exists
        if self.TIINGO_CLIENT is None:
            await self.instantiate_not_pickleable_objects()

        # 2) Partition by category
        tp_equities = [tp for tp in pairs if tp.trade_pair_category == TradePairCategory.EQUITIES]
        tp_crypto   = [tp for tp in pairs if tp.trade_pair_category == TradePairCategory.CRYPTO]
        tp_forex    = [tp for tp in pairs if tp.trade_pair_category == TradePairCategory.FOREX]

        # 3) Schedule all three async fetchers
        tasks = []
        if tp_equities:
            tasks.append(self.get_closes_equities(tp_equities, verbose))
        if tp_crypto:
            tasks.append(self.get_closes_crypto(tp_crypto, verbose))
        if tp_forex:
            tasks.append(self.get_closes_forex(tp_forex, verbose))

        # 4) Run them in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 5) Merge dicts and log exceptions
        tp_to_price: Dict[TradePair, PriceSource] = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                bt.logging.error(f"get_closes_rest: task #{i} raised {result}")
            else:
                tp_to_price.update(result)

        return tp_to_price

    @exception_handler_decorator()
    async def get_closes_equities(self, trade_pairs: List[TradePair], verbose=False, target_time_ms=None) -> dict[TradePair: PriceSource]:
        if target_time_ms:
            raise Exception('TODO')
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
        requestResponse = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=5)
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

    def target_ms_to_start_end_formatted(self, target_time_ms):
        start_day_formatted = TimeUtil.millis_to_short_date_str(target_time_ms)
        end_day_datetime = TimeUtil.millis_to_datetime(target_time_ms)
        # One day ahead.
        end_day_datetime += timedelta(days=1)
        end_day_formatted = end_day_datetime.strftime("%Y-%m-%d")
        return start_day_formatted, end_day_formatted

    @exception_handler_decorator()
    async def get_closes_forex(self, trade_pairs: List[TradePair], verbose=False, target_time_ms=None) -> dict:

        def tickers_to_tiingo_forex_url(tickers: List[str]) -> str:
            if target_time_ms:
                start_day_formatted, end_day_formatted = self.target_ms_to_start_end_formatted(target_time_ms)
                return f"https://api.tiingo.com/tiingo/fx/prices?tickers={','.join(tickers)}&startDate={start_day_formatted}&endDate={end_day_formatted}&resampleFreq=1min&token={self.config['api_key']}"
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
        time_now_ms = TimeUtil.now_in_millis()
        requestResponse = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=5)
        if requestResponse.status_code == 200:
            lowest_delta = float('inf')
            for x in requestResponse.json():
                tp = TradePair.get_latest_trade_pair_from_trade_pair_id(x['ticker'].upper())
                if target_time_ms:
                    # Rows look like {'close': 148.636, 'date': '2025-03-21T00:00:00.000Z', 'high': 148.6575, 'low': 148.5975, 'open': 148.6245, 'ticker': 'usdjpy'}
                    attempting_previous_close = not self.is_market_open(tp, time_ms=target_time_ms)
                    data_time_ms = TimeUtil.parse_iso_to_ms(x['date'])
                    delta = abs(data_time_ms - target_time_ms)
                    if delta < lowest_delta:
                        bid = ask = 0  # Bid/ask not provided in historical data
                        p_name = f'{TIINGO_PROVIDER_NAME}_historical'
                        open = float(x['open'])
                        close = float(x['close'])
                        vwap = close
                        high = float(x['high'])
                        low = float(x['low'])
                        lag_ms = target_time_ms - data_time_ms
                        timespan_ms = self.timespan_to_ms['minute']
                        lowest_delta = delta
                    else:
                        continue
                else:
                    attempting_previous_close = not self.is_market_open(tp, time_ms=time_now_ms)
                    bid_raw = x['bidPrice']
                    ask_raw = x['askPrice']
                    if not bid_raw:
                        continue
                    if not ask_raw:
                        continue
                    bid = float(bid_raw) if bid_raw else 0
                    ask = float(ask_raw) if ask_raw else 0
                    high = ask
                    low = bid
                    mid_price = (bid + ask) / 2.0
                    open = close = vwap = mid_price
                    data_time_ms = TimeUtil.parse_iso_to_ms(x['quoteTimestamp'])
                    timespan_ms = 0
                    lag_ms = time_now_ms - data_time_ms
                    p_name = f'{TIINGO_PROVIDER_NAME}_rest'

                if attempting_previous_close:
                    p_name += '_prev_close'
                tp_to_price[tp] = PriceSource(
                    source=p_name,
                    timespan_ms=timespan_ms,
                    open=open,
                    close=close,
                    vwap=vwap,
                    high=high,
                    low=low,
                    start_ms=data_time_ms,
                    websocket=False,
                    lag_ms=lag_ms,
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
    async def get_closes_crypto(self, trade_pairs: List[TradePair], verbose=False, target_time_ms=None) -> dict:
        tp_to_price = {}
        if not trade_pairs:
            return tp_to_price
        assert all(tp.trade_pair_category == TradePairCategory.CRYPTO for tp in trade_pairs), trade_pairs

        def tickers_to_crypto_url(tickers: List[str]) -> str:
            if target_time_ms:
                # YYYY-MM-DD format.
                start_day_formatted, end_day_formatted = self.target_ms_to_start_end_formatted(target_time_ms)
                # "https://api.tiingo.com/tiingo/crypto/prices?tickers=btcusd&startDate=2019-01-02&resampleFreq=5min&token=ffb55f7fdd167d4b8047539e6b62d82b92b25f91"
                return f"https://api.tiingo.com/tiingo/crypto/prices?tickers={','.join(tickers)}&startDate={start_day_formatted}&endDate={end_day_formatted}&resampleFreq=1min&token={self.config['api_key']}&exchanges={TIINGO_COINBASE_EXCHANGE_STR.upper()}"
            return f"https://api.tiingo.com/tiingo/crypto/top?tickers={','.join(tickers)}&token={self.config['api_key']}&exchanges={TIINGO_COINBASE_EXCHANGE_STR.upper()}"

        url = tickers_to_crypto_url([self.trade_pair_to_tiingo_ticker(x) for x in trade_pairs])
        if verbose:
            print('hitting url', url)

        requestResponse = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=5)

        if requestResponse.status_code == 200:
            response_data = requestResponse.json()

            if target_time_ms:
                # Historical data has a different structure - the items are in data[0]['priceData']
                if not response_data or len(response_data) == 0:
                    return tp_to_price
                for crypto_data in response_data:
                    ticker = crypto_data['ticker']

                    # Skip if no price data available
                    if not crypto_data.get('priceData') or len(crypto_data['priceData']) == 0:
                        continue

                    # Find the closest price data point to target_time_ms
                    price_data = sorted(crypto_data['priceData'],
                                        key=lambda x: TimeUtil.parse_iso_to_ms(x['date']))

                    closest_data = min(price_data,
                                       key=lambda x: abs(TimeUtil.parse_iso_to_ms(x['date']) - target_time_ms))

                    data_time_ms = TimeUtil.parse_iso_to_ms(closest_data['date'])
                    price = float(closest_data['close'])
                    bid_price = ask_price = 0  # Bid/ask not provided in historical data

                    tp = TradePair.get_latest_trade_pair_from_trade_pair_id(ticker.upper())
                    source_name = f'{TIINGO_PROVIDER_NAME}_{TIINGO_COINBASE_EXCHANGE_STR}_historical'
                    exchange = TIINGO_COINBASE_EXCHANGE_STR

                    # Create PriceSource
                    tp_to_price[tp] = PriceSource(
                        source=source_name,
                        timespan_ms=self.timespan_to_ms['minute'],
                        open=float(closest_data['open']),
                        close=price,
                        vwap=price,
                        high=float(closest_data['high']),
                        low=float(closest_data['low']),
                        start_ms=data_time_ms,
                        websocket=False,
                        lag_ms=target_time_ms - data_time_ms,
                        bid=bid_price,
                        ask=ask_price
                    )

                    if verbose:
                        self.log_price_info(tp, tp_to_price[tp], target_time_ms, data_time_ms,
                                       closest_data['date'], price, exchange, closest_data)
            else:
                now_ms = TimeUtil.now_in_millis()
                # Current data format (top endpoint)
                for crypto_data in response_data:
                    ticker = crypto_data['ticker']
                    if len(crypto_data['topOfBookData']) != 1:
                        print('Tiingo unexpected data', crypto_data)
                        continue

                    book_data = crypto_data['topOfBookData'][0]

                    # Determine the data source and timestamp
                    data_time_ms, price, exchange, bid_price, ask_price = self.get_best_crypto_price_info(
                        book_data, now_ms, TIINGO_COINBASE_EXCHANGE_STR
                    )

                    # Create trade pair
                    tp = TradePair.get_latest_trade_pair_from_trade_pair_id(ticker.upper())
                    price = float(price)
                    source_name = f'{TIINGO_PROVIDER_NAME}_{exchange}_rest'

                    # Create PriceSource
                    tp_to_price[tp] = PriceSource(
                        source=source_name,
                        timespan_ms=self.timespan_to_ms['minute'],
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
                        self.log_price_info(tp, tp_to_price[tp], now_ms, data_time_ms,
                                       book_data['quoteTimestamp'], price, exchange, book_data)

        return tp_to_price

    def get_best_crypto_price_info(self, book_data, now_ms, preferred_exchange):
        """Helper function to determine the best price info from book data"""
        data_time_exchange_ms = TimeUtil.parse_iso_to_ms(book_data['lastSaleTimestamp'])
        data_time_quote_ms = TimeUtil.parse_iso_to_ms(book_data['quoteTimestamp'])
        delta_ms_exchange = now_ms - data_time_exchange_ms
        delta_ms_quote = now_ms - data_time_quote_ms
        THRESHOLD_FRESH_MS = 15 * 10000

        last_exchange = book_data['lastExchange'].lower() if book_data.get('lastExchange') else None
        bid_exchange = book_data['bidExchange'].lower() if book_data.get('bidExchange') else None
        ask_exchange = book_data['askExchange'].lower() if book_data.get('askExchange') else None

        bid_price = float(book_data['bidPrice']) if book_data.get('bidPrice') else 0
        ask_price = float(book_data['askPrice']) if book_data.get('askPrice') else 0

        # Prioritize data from preferred exchange that's fresh
        if last_exchange == preferred_exchange and delta_ms_exchange < THRESHOLD_FRESH_MS:
            return data_time_exchange_ms, book_data['lastPrice'], last_exchange, bid_price, ask_price
        elif bid_exchange == preferred_exchange and delta_ms_quote < THRESHOLD_FRESH_MS:
            return data_time_quote_ms, book_data['bidPrice'], bid_exchange, bid_price, ask_price
        elif ask_exchange == preferred_exchange and delta_ms_quote < THRESHOLD_FRESH_MS:
            return data_time_quote_ms, book_data['askPrice'], ask_exchange, bid_price, ask_price

        # Fresh data from any exchange
        elif last_exchange and delta_ms_exchange < THRESHOLD_FRESH_MS:
            return data_time_exchange_ms, book_data['lastPrice'], last_exchange, bid_price, ask_price
        elif bid_exchange and delta_ms_quote < THRESHOLD_FRESH_MS:
            return data_time_quote_ms, book_data['bidPrice'], bid_exchange, bid_price, ask_price
        elif ask_exchange and delta_ms_quote < THRESHOLD_FRESH_MS:
            return data_time_quote_ms, book_data['askPrice'], ask_exchange, bid_price, ask_price

        # Any data available
        elif last_exchange:
            return data_time_exchange_ms, book_data['lastPrice'], last_exchange, bid_price, ask_price
        elif bid_exchange:
            return data_time_quote_ms, book_data['bidPrice'], bid_exchange, bid_price, ask_price
        elif ask_exchange:
            return data_time_quote_ms, book_data['askPrice'], ask_exchange, bid_price, ask_price
        else:
            raise Exception('unexpected Tiingo data', book_data)

    def log_price_info(self, tp, price_source, now_ms, data_time_ms, timestamp, price, exchange, raw_data):
        """Helper function to log price information in verbose mode"""
        time_delta_s = (now_ms - data_time_ms) / 1000
        time_delta_formatted_2_decimals = round(time_delta_s, 2)
        print((tp.trade_pair_id, price_source, time_delta_formatted_2_decimals, timestamp, price, exchange, raw_data))

    async def get_close_rest(
            self,
            trade_pair: TradePair,
            attempting_prev_close: bool = False,
            target_time_ms: int | None = None) -> PriceSource | None:

        # Ensure Tiingo client is initialized
        if self.TIINGO_CLIENT is None:
            await self.instantiate_not_pickleable_objects()

        # Use the correct getter based on trade pair category
        category_getters = {
            TradePairCategory.EQUITIES: self.get_closes_equities,
            TradePairCategory.CRYPTO: self.get_closes_crypto,
            TradePairCategory.FOREX: self.get_closes_forex
        }

        getter = category_getters.get(trade_pair.trade_pair_category)
        if not getter:
            raise ValueError(f"Unknown trade pair category {trade_pair}")

        # Use the appropriate getter to fetch the result
        result = await getter([trade_pair], target_time_ms=target_time_ms)
        return result.get(trade_pair)

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
    #time.sleep(1000)
    tp_to_prices = asyncio.run(tds.get_closes_rest([TradePair.BTCUSD, TradePair.USDJPY, TradePair.NVDA], verbose=True))
    time.sleep(1000000)
    assert 0, {x.trade_pair_id: y for x, y in tp_to_prices.items()}


    #time.sleep(10000)
    for trade_pair in TradePair:
        if not trade_pair.is_forex:
            continue
        # Get rest data
        if trade_pair.is_indices:
            continue
        ps = tds.get_close_rest(trade_pair, target_time_ms=None)
        if ps:
            print(f"Got {ps} for {trade_pair}")
        else:
            print(f"No data for {trade_pair}")
    time.sleep(100000)
    #assert 0
    target_timestamp_ms = 1715288502999

    client = TiingoClient({'api_key': secrets['tiingo_apikey']})
    crypto_price = client.get_crypto_top_of_book(['BTCUSD'])


    # forex_price = client.get_(ticker='USDJPY')# startDate='2021-01-01', endDate='2021-01-02', frequency='daily')
    #tds = TiingoDataService(secrets['tiingo_apikey'], disable_ws=True)



