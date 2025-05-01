import asyncio
import functools
import random
import traceback

import requests

from typing import List, Dict, Any

from vali_objects.vali_dataclasses.order import Order
from polygon.websocket import Market, EquityAgg, EquityTrade, CryptoTrade, ForexQuote, WebSocketClient

from data_generator.base_data_service import BaseDataService, POLYGON_PROVIDER_NAME, exception_handler_decorator
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
import time

from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt
from polygon import RESTClient

from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker

DEBUG = 0

# Add a method to the WebSocketClient class to handle safe closing
if not hasattr(WebSocketClient, 'close_connection'):
    def close_connection(self):
        self._running = False
        if hasattr(self, '_ws') and self._ws:
            try:
                self._ws.close()
            except Exception as e:
                print(f"Error closing websocket connection: {e}")


    WebSocketClient.close_connection = close_connection

class Agg:
    """
    An efficient representation of an aggregate price data point. Use this over the Polygon Agg for speed.
    """
    def __init__(self, open, close, high, low, vwap, timestamp, bid, ask, volume):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.bid = bid
        self.ask = ask
        self.vwap = vwap
        self.timestamp = timestamp
        self.volume = volume

    def __str__(self):
        return (
            f"Agg("
            f"open={self.open}, close={self.close}, high={self.high}, low={self.low}, "
            f"bid={self.bid}, ask={self.ask}, vwap={self.vwap}, timestamp={self.timestamp}, vol={self.volume})"
        )

    # Optional: Add a __repr__ method for better representation in debugging and interactive sessions
    def __repr__(self):
        return (
            f"Agg("
            f"open={self.open}, close={self.close}, high={self.high}, low={self.low}, "
            f"bid={self.bid}, ask={self.ask}, vwap={self.vwap}, timestamp={self.timestamp}, vol={self.volume})"
        )


class ExchangeMappingHelper:
    def __init__(self, api_key, fetch_live_mapping=True):
        self.fetch_live_mapping = fetch_live_mapping
        self.api_key = api_key
        self.crypto_fallback_mapping = {
            'coinbase': 1,
            'bitfinex': 2,
            'bitstamp': 6,
            'binance': 10,
            'kraken': 23
        }
        self.stock_fallback_mapping = {
            "nyse american, llc": 1,
            "nasdaq omx bx, inc.": 2,
            "nyse national, inc.": 3,
            "finra alternative display facility": 4,
            "unlisted trading privileges": 5,
            "international securities exchange, llc - stocks": 6,
            "cboe edga": 7,
            "cboe edgx": 8,
            "nyse chicago, inc.": 9,
            "new york stock exchange": 10,
            "nyse arca, inc.": 11,
            "nasdaq": 12,
            "consolidated tape association": 13,
            "long-term stock exchange": 14,
            "investors exchange": 15
        }
        self.crypto_mapping = {}
        self.stock_mapping = {}

        self.create_crypto_mapping()
        self.create_stock_mapping()

    def create_crypto_mapping(self):
        if not self.fetch_live_mapping:
            self.crypto_mapping = self.crypto_fallback_mapping
            return
        endpoint = "https://api.polygon.io/v3/reference/exchanges"
        params = {
            "asset_class": "crypto",
            "apiKey": self.api_key
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response
            data = response.json()
            if 'results' in data and isinstance(data['results'], list):
                self.crypto_mapping = {
                    entry['name'].lower(): entry['id']
                    for entry in data['results']
                }
                print("Successfully created crypto mapping from API.")
            else:
                print("Unexpected response structure. Using fallback mapping.")
                self.crypto_mapping = self.crypto_fallback_mapping

        except Exception as e:
            print(f"API request failed: {e}. Using fallback mapping.")
            self.crypto_mapping = self.crypto_fallback_mapping

    def create_stock_mapping(self):
        if not self.fetch_live_mapping:
            self.stock_mapping = self.stock_fallback_mapping
            return
        endpoint = "https://api.polygon.io/v3/reference/exchanges"
        params = {
            "asset_class": "stocks",
            "apiKey": self.api_key
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response
            data = response.json()
            if 'results' in data and isinstance(data['results'], list):
                self.stock_mapping = {
                    entry['name'].lower(): entry['id']
                    for entry in data['results']
                }
                print("Successfully created stock mapping from API.")
            else:
                print("Unexpected response structure. Using fallback mapping.")
                self.stock_mapping = self.stock_fallback_mapping

        except Exception as e:
            print(f"API request failed: {e}. Using fallback mapping.")
            self.stock_mapping = self.stock_fallback_mapping


class PolygonDataService(BaseDataService):

    def __init__(self, api_key, disable_ws=False, ipc_manager=None, is_backtesting=False):
        bt.logging.enable_info()
        self.init_time = time.time()
        self._api_key = api_key
        ehm = ExchangeMappingHelper(api_key, fetch_live_mapping=not disable_ws)
        self.crypto_mapping = ehm.crypto_mapping
        self.equities_mapping = ehm.stock_mapping
        self.disable_ws = disable_ws
        self.N_CANDLES_LIMIT = 50000
        self.tp_to_mfs = {}
        self.is_backtesting = is_backtesting

        super().__init__(provider_name=POLYGON_PROVIDER_NAME, ipc_manager=ipc_manager)

        self.MARKET_STATUS = None
        self.UNSUPPORTED_TRADE_PAIRS = (
        TradePair.SPX, TradePair.DJI, TradePair.NDX, TradePair.VIX, TradePair.FTSE, TradePair.GDAXI)

        # Initialize REST client after ws creation to avoid pickling error
        self.POLYGON_CLIENT = None

        if not self.disable_ws:
            self.start_ws_async()

    def instantiate_not_pickleable_objects(self):
        """Initialize REST client and other non-pickleable objects"""
        try:
            if self.POLYGON_CLIENT is None:
                self.POLYGON_CLIENT = RESTClient(api_key=self._api_key, num_pools=20)
                bt.logging.info("Successfully initialized Polygon REST client")
        except Exception as e:
            bt.logging.error(f"Failed to initialize Polygon REST client: {e}")
            # Don't raise - allow operation without REST client and try again later

    async def _requests_get_async(self, url: str, **kwargs):
        """Perform HTTP GET requests asynchronously without blocking the event loop"""
        loop = asyncio.get_running_loop()

        # Set timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 10

        return await loop.run_in_executor(
            None,
            functools.partial(requests.get, url, **kwargs)
        )

    async def _close_websocket_safely(self, client):
        """Safely close a websocket client with proper error handling"""
        if client is None:
            return

        try:
            # First try our custom method if available
            if hasattr(client, 'close_connection'):
                client.close_connection()
            # Then try to close the inner websocket directly
            elif hasattr(client, '_ws') and client._ws:
                client._ws.close()
        except Exception as e:
            bt.logging.warning(f"Error during websocket close: {e}")
            # Continue anyway - we're cleaning up

    async def _close_create_websocket_objects(self, tpc: TradePairCategory):
        """Close old websocket and create a new one for a specific category only"""
        # Skip if this is indices (not supported)
        if tpc == TradePairCategory.INDICES:
            return

        # Close old client with timeout protection
        old_client = self.websocket_clients.get(tpc)
        if old_client:
            try:
                # Use a timeout for closing to prevent hanging
                close_task = asyncio.create_task(self._close_websocket_safely(old_client))
                await asyncio.wait_for(close_task, timeout=5.0)
                bt.logging.info(f"Closed old websocket for {tpc.name}")
            except asyncio.TimeoutError:
                bt.logging.warning(f"Timeout closing websocket for {tpc.name}, forcing cleanup")
            except Exception as e:
                bt.logging.warning(f"Error closing websocket for {tpc.name}: {e}")

        # Remove the old client reference
        self.websocket_clients.pop(tpc, None)

        # Create new client for this specific category
        market = {
            TradePairCategory.EQUITIES: Market.Stocks,
            TradePairCategory.FOREX: Market.Forex,
            TradePairCategory.CRYPTO: Market.Crypto,
        }[tpc]

        # Create new client
        client = WebSocketClient(market=market, api_key=self._api_key)
        self.websocket_clients[tpc] = client

        # Reset the connection state
        self.ws_state[tpc]["last_activity"] = time.time()

        # Subscribe to symbols for this category only
        await self._subscribe_websockets_async(tpc)

        bt.logging.info(f"Created new websocket client for {tpc.name}")

    async def _subscribe_websockets_async(self, tpc: TradePairCategory):
        """Subscribe to symbols asynchronously to prevent blocking"""
        client = self.websocket_clients.get(tpc)
        if not client:
            bt.logging.error(f"No websocket client for {tpc} to subscribe")
            return

        subscription_count = 0
        subscription_errors = 0

        for tp in TradePair:
            if tp in self.UNSUPPORTED_TRADE_PAIRS:
                continue
            if tpc and tp.trade_pair_category != tpc:
                continue

            symbol = None
            if tp.is_crypto and tpc == TradePairCategory.CRYPTO:
                symbol = "XT." + tp.trade_pair.replace('/', '-')
            elif tp.is_forex and tpc == TradePairCategory.FOREX:
                symbol = "C." + tp.trade_pair
            elif tp.is_equities and tpc == TradePairCategory.EQUITIES:
                symbol = "T." + tp.trade_pair
            else:
                continue  # Skip if not matching category

            if symbol:
                try:
                    client.subscribe(symbol)
                    subscription_count += 1
                    bt.logging.info(f"Subscribed to {symbol}")

                    # Give event loop a chance to process other tasks periodically
                    if subscription_count % 10 == 0:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    bt.logging.error(f"Failed to subscribe to {symbol}: {e}")
                    subscription_errors += 1

        bt.logging.info(f"Subscribed to {subscription_count} symbols for {tpc} with {subscription_errors} errors")

    async def _run_websocket_with_backoff(self, tpc: TradePairCategory):
        """Common implementation for running a websocket with proper backoff"""
        # Initial backoff parameters
        max_backoff = 60.0  # Maximum backoff in seconds

        while True:
            # Get the current backoff based on reconnect attempts
            reconnect_attempts = self.ws_state[tpc]["reconnect_attempts"]
            backoff = min(2.0 ** reconnect_attempts, max_backoff)

            ws = self.websocket_clients.get(tpc)
            if not ws:
                bt.logging.error(f"No websocket client for {tpc}, recreating...")
                try:
                    # Only recreate this specific category's websocket
                    await self._close_create_websocket_objects(tpc)
                except Exception as e:
                    bt.logging.error(f"Failed to create websocket client for {tpc}: {e}")
                    # Wait before retrying
                    await asyncio.sleep(backoff)
                    continue

                # Get the newly created client
                ws = self.websocket_clients.get(tpc)
                if not ws:
                    bt.logging.error(f"Still no websocket client for {tpc} after recreation attempt")
                    await asyncio.sleep(backoff)
                    continue

            try:
                # Set up a wrapper function that updates the last activity time
                def handle_msg_with_activity_tracking(msgs):
                    # Mark this category as active whenever messages arrive
                    if msgs:
                        self.ws_state[tpc]["last_activity"] = time.time()
                        # Track the message count
                        self.ws_state[tpc]["last_message_count"] = len(msgs)
                    self.handle_msg(msgs)

                # Start the websocket in a separate thread
                loop = asyncio.get_running_loop()
                run_task = loop.run_in_executor(
                    None,
                    lambda: ws.run(handle_msg_with_activity_tracking)
                )

                # INSTEAD OF WAITING FOR COMPLETION WITH TIMEOUT:
                # We'll monitor activity in a separate task
                start_time = time.time()
                self.ws_state[tpc]["last_activity"] = start_time
                self.ws_state[tpc]["monitoring_task"] = True

                while self.ws_state[tpc].get("monitoring_task", True):
                    # Check if market is open - only monitor while market is open
                    first_pair = self.get_first_trade_pair_in_category(tpc)
                    market_open = first_pair and self.is_market_open(first_pair)

                    # Get time since last activity
                    now = time.time()
                    last_activity = self.ws_state[tpc]["last_activity"]
                    inactive_time = now - last_activity

                    # Check if websocket has been inactive for too long
                    # Only consider this an issue if market is open AND we haven't seen activity
                    if market_open and inactive_time > 120.0:  # 2 minutes of inactivity
                        bt.logging.warning(
                            f"{tpc} websocket inactive for {inactive_time:.1f} seconds while market is open. "
                            f"Forcing reconnect."
                        )
                        # Stop monitoring and break out to reconnect
                        self.ws_state[tpc]["monitoring_task"] = False
                        break

                    # Check if the run_task completed unexpectedly
                    if run_task.done():
                        # The websocket closed on its own
                        try:
                            result = run_task.result()  # This will raise exception if task failed
                            bt.logging.warning(f"{tpc} websocket closed normally (result: {result}), reconnecting")
                        except Exception as e:
                            bt.logging.error(f"Error in {tpc} websocket: {e}")

                        # Stop monitoring and break out to reconnect
                        self.ws_state[tpc]["monitoring_task"] = False
                        break

                    # Sleep before checking again
                    await asyncio.sleep(10)  # Check every 10 seconds

                # Reset backoff on clean exit
                self.ws_state[tpc]["reconnect_attempts"] = 0

            except Exception as e:
                bt.logging.error(f"Error in {tpc} websocket monitoring: {e}")
                # Increment reconnect attempts
                self.ws_state[tpc]["reconnect_attempts"] += 1

            # Cancel the run_task if it's still running
            if 'run_task' in locals() and not run_task.done():
                run_task.cancel()

            # Always recreate the client before reconnecting, but ONLY for this category
            bt.logging.info(f"Reconnecting {tpc} websocket in {backoff:.1f} seconds...")
            await asyncio.sleep(backoff)

            try:
                # Only close and recreate this specific category's websocket
                await self._close_create_websocket_objects(tpc)
            except Exception as e:
                bt.logging.error(f"Failed to recreate {tpc} client: {e}")

    async def main_stocks(self):
        """Run the websocket for stocks with proper error handling and backoff"""
        tpc = TradePairCategory.EQUITIES
        await self._run_websocket_with_backoff(tpc)

    async def main_forex(self):
        """Run the websocket for forex with proper error handling and backoff"""
        tpc = TradePairCategory.FOREX
        await self._run_websocket_with_backoff(tpc)

    async def main_crypto(self):
        """Run the websocket for crypto with proper error handling and backoff"""
        tpc = TradePairCategory.CRYPTO
        await self._run_websocket_with_backoff(tpc)

    def parse_price_for_forex(self, m, stats=None, is_ws=False) -> (float, float, float):
        """Parse forex price data with validation"""
        try:
            t_ms = m.timestamp if is_ws else m.participant_timestamp // 1000000

            # Validate inputs
            if not hasattr(m, 'bid_price') or not hasattr(m, 'ask_price'):
                return None, None, None

            # Handle zero or negative prices
            if m.bid_price <= 0 or m.ask_price <= 0:
                return None, None, None

            delta = abs(m.bid_price - m.ask_price) / m.bid_price * 100.0

            # Update stats if provided
            if stats:
                stats['n'] += 1
                stats['sum_deltas'] += delta
                stats['avg_delta'] = stats['sum_deltas'] / (stats['n'])
                stats['max_delta'] = max(stats['max_delta'], delta)

            # Filter out abnormal bid/ask spreads
            if delta > .20:  # Threshold for wonky data
                if stats:
                    stats['n_skipped'] += 1
                    if stats['n'] % 10 == 0:
                        bt.logging.warning(
                            f"Ignoring unusual Forex price data bid: {m.bid_price:.4f}, ask: {m.ask_price:.4f}, "
                            f"{delta:.4f} time {TimeUtil.millis_to_formatted_date_str(t_ms // 1000000)}"
                        )
                return None, None, None

            return m.bid_price, m.ask_price, delta

        except Exception as e:
            bt.logging.error(f"Error parsing forex price: {e}")
            return None, None, None

    def _get_category_from_message(self, msg):
        """Determine the message category"""
        try:
            if isinstance(msg, EquityAgg) or isinstance(msg, EquityTrade):
                return "equities"
            elif isinstance(msg, CryptoTrade):
                return "crypto"
            elif isinstance(msg, ForexQuote):
                return "forex"
            else:
                return "unknown"
        except:
            return "unknown"

    def handle_msg(self, msgs):
        """
        Handle incoming websocket messages
        """
        if not msgs:
            return

        try:
            # Process all messages in the batch
            for m in msgs:
                # Process each message safely
                self._process_single_message(m)

                # Update the websocket state for the appropriate category
                if hasattr(self, '_get_category_from_message'):
                    category = self._get_category_from_message(m)
                    if category in self.ws_state:
                        self.ws_state[category]["last_activity"] = time.time()

        except Exception as e:
            # Simple error logging - just the error and a short traceback
            bt.logging.error(f"Error in handle_msg: {e}")
            tb_lines = traceback.format_exc().splitlines()
            if len(tb_lines) > 3:
                bt.logging.error(f"Traceback: {tb_lines[-3:]}")
            else:
                bt.logging.error(f"Traceback: {tb_lines}")

    def _process_single_message(self, m):
        """Process a single websocket message safely"""
        try:
            # Determine the trade pair from the message
            if isinstance(m, EquityAgg):
                tp = self.symbol_to_trade_pair(m.symbol[2:])  # I:SPX -> SPX
            elif isinstance(m, CryptoTrade):
                tp = self.symbol_to_trade_pair(m.pair)
            elif isinstance(m, ForexQuote):
                tp = self.symbol_to_trade_pair(m.pair)
            elif isinstance(m, EquityTrade):
                tp = self.symbol_to_trade_pair(m.symbol)
            else:
                bt.logging.warning(f"Unknown message type in Polygon websocket: {type(m)}")
                return

            # Track events by category
            self.tpc_to_n_events[tp.trade_pair_category] += 1

            # Process the message to extract price sources
            symbol = tp.trade_pair
            ps1, ps2 = self._msg_to_price_sources(m, tp)

            if ps1 is None and ps2 is None:
                return

            # Reset the closed market price, indicating that a new close should be fetched after the current day's close
            self.closed_market_prices[tp] = None

            # Update our price sources
            for ps in [ps1, ps2]:
                if ps is None:
                    continue

                # Store the latest event
                self.latest_websocket_events[symbol] = ps

                # Initialize recent event tracker if needed
                if symbol not in self.trade_pair_to_recent_events:
                    if self.using_ipc:
                        self.trade_pair_to_recent_events[symbol] = RecentEventTracker()
                    else:
                        self.trade_pair_to_recent_events_realtime[symbol] = RecentEventTracker()

                # Add the event to the appropriate tracker
                if self.using_ipc:
                    self.trade_pair_to_recent_events_realtime[symbol].add_event(
                        ps, tp.is_forex, f"{self.provider_name}:{tp.trade_pair}"
                    )
                else:
                    self.trade_pair_to_recent_events[symbol].add_event(
                        ps, tp.is_forex, f"{self.provider_name}:{tp.trade_pair}"
                    )


        except Exception as e:
            bt.logging.error(f"Error processing message {type(m)}: {e}")

    def _msg_to_price_sources(self, m, tp):
        """Convert a polygon message to price sources"""
        symbol = tp.trade_pair
        bid = 0
        ask = 0

        try:
            if tp.is_forex:
                bid, ask, _ = self.parse_price_for_forex(m, is_ws=True)
                if bid is None:
                    return None, None

                start_timestamp = m.timestamp
                end_timestamp = start_timestamp + 999

                # Check if we've already seen this timestamp to avoid duplicates
                if symbol in self.trade_pair_to_recent_events and self.trade_pair_to_recent_events[
                    symbol].timestamp_exists(start_timestamp):
                    buffer = self.trade_pair_to_recent_events_realtime if self.using_ipc else self.trade_pair_to_recent_events
                    buffer[symbol].update_prices_for_median(start_timestamp, bid, ask)
                    buffer[symbol].update_prices_for_median(start_timestamp + 999, bid, ask)
                    return None, None
                else:
                    open = close = vwap = high = low = bid

            elif tp.is_equities:
                # Skip data from non-primary exchanges
                if m.exchange != self.equities_mapping['nasdaq']:
                    return None, None

                # Skip after-hours trading
                if isinstance(m, EquityTrade) and isinstance(m.conditions, list) and 12 in m.conditions:
                    self.n_equity_events_skipped_afterhours += 1
                    return None, None

                # Round timestamp to reduce duplicates
                start_timestamp = round(m.timestamp, -3)  # round to nearest second
                end_timestamp = None
                open = close = vwap = high = low = m.price

            elif tp.is_crypto:
                # Only use data from the primary exchange (Coinbase)
                if m.exchange != self.crypto_mapping['coinbase']:
                    return None, None

                # Round timestamp to reduce duplicates
                start_timestamp = round(m.received_timestamp, -3)  # round to nearest second
                end_timestamp = None
                open = close = vwap = high = low = m.price

            else:
                start_timestamp = m.start_timestamp
                end_timestamp = m.end_timestamp - 1  # prioritize a new candle's open over a previous candle's close
                open = m.open
                close = m.close
                vwap = m.vwap
                high = m.high
                low = m.low

            # Calculate lag time for this event
            now_ms = TimeUtil.now_in_millis()
            lag_ms = now_ms - start_timestamp

            # Create the first price source
            price_source1 = PriceSource(
                source=f'{POLYGON_PROVIDER_NAME}_ws',
                timespan_ms=0,
                open=open,
                close=open,
                vwap=vwap,
                high=high,
                low=low,
                start_ms=start_timestamp,
                websocket=True,
                lag_ms=lag_ms,
                bid=bid,
                ask=ask
            )

            # For point-in-time trades (equities, crypto), we don't create a second price source
            if tp.is_equities or tp.is_crypto:
                price_source2 = None
            else:
                # Create a second price source for the end of the interval (for candles)
                price_source2 = PriceSource(
                    source=f'{POLYGON_PROVIDER_NAME}_ws',
                    timespan_ms=0,
                    open=close,
                    close=close,
                    vwap=vwap,
                    high=high,
                    low=low,
                    start_ms=end_timestamp,
                    websocket=True,
                    lag_ms=now_ms - end_timestamp,
                    bid=bid,
                    ask=ask
                )

            return price_source1, price_source2

        except Exception as e:
            bt.logging.error(f"Error creating price sources: {e}")
            return None, None

    def subscribe_websockets(self, tpc: TradePairCategory = None):
        """Subscribe to websocket feeds - synchronous version"""
        for tp in TradePair:
            if tp in self.UNSUPPORTED_TRADE_PAIRS:
                continue
            if tpc and tp.trade_pair_category != tpc:
                continue

            try:
                if tp.is_crypto:
                    symbol = "XT." + tp.trade_pair.replace('/', '-')
                    print('Poly subscribe:', symbol)
                    self.websocket_clients[TradePairCategory.CRYPTO].subscribe(symbol)
                elif tp.is_forex:
                    symbol = "C." + tp.trade_pair
                    print('Poly subscribe:', symbol)
                    self.websocket_clients[TradePairCategory.FOREX].subscribe(symbol)
                elif tp.is_equities:
                    symbol = "T." + tp.trade_pair
                    print('Poly subscribe:', symbol)
                    self.websocket_clients[TradePairCategory.EQUITIES].subscribe(symbol)
                elif tp.is_indices:
                    continue
                else:
                    bt.logging.warning(f"Unknown trade pair category: {tp.trade_pair_category}")
            except Exception as e:
                bt.logging.error(f"Error subscribing to {tp.trade_pair}: {e}")

    def symbol_to_trade_pair(self, symbol: str):
        """Convert a symbol to a trade pair object"""
        # Try direct lookup first
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol)
        if tp:
            return tp

        # Try for crypto pairs which use - instead of /
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol.replace('-', '/'))
        if not tp:
            raise ValueError(f"Unknown symbol: {symbol}")

        return tp

    @exception_handler_decorator()
    async def get_closes_rest(self, pairs: List[TradePair]) -> Dict[TradePair, Any]:
        """Get prices for multiple trade pairs in parallel"""
        tasks = []
        for tp in pairs:
            task = asyncio.create_task(self.get_close_rest(tp))
            task.tp = tp  # Attach the trade pair to the task for reference
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dictionary
        all_trade_pair_closes = {}
        for task, result in zip(tasks, results):
            tp = task.tp
            if isinstance(result, Exception):
                bt.logging.error(f"{tp} generated an exception: {result}")
            else:
                all_trade_pair_closes[tp] = result if result is not None else {}

        return all_trade_pair_closes

    def agg_to_price_source(self, a, now_ms: int, timespan: str, attempting_prev_close: bool = False):
        """Convert an aggregate object to a price source"""
        p_name = f'{POLYGON_PROVIDER_NAME}_rest'
        if attempting_prev_close:
            p_name += '_prev_close'

        return PriceSource(
            source=p_name,
            timespan_ms=self.timespan_to_ms[timespan],
            open=a.open,
            close=a.close,
            vwap=a.vwap,
            high=a.high,
            low=a.low,
            start_ms=a.timestamp,
            websocket=False,
            lag_ms=now_ms - a.timestamp,
            bid=a.bid if hasattr(a, 'bid') else 0,
            ask=a.ask if hasattr(a, 'ask') else 0)

    @exception_handler_decorator()
    async def get_close_rest(
            self,
            trade_pair: TradePair,
            timestamp_ms: int = None,
            order: Order = None
    ) -> PriceSource | None:
        """Get the price for a trade pair at a specific time"""
        # Initialize REST client if needed
        if self.POLYGON_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        # Check if we're in backtesting mode with market open verification
        if self.is_backtesting:
            # Check that we are within market hours for genuine ptn orders
            if order is not None and order.src == 0:
                assert self.is_market_open(trade_pair)

        # If market is not open, return the last price before close
        if not self.is_market_open(trade_pair):
            return self.get_event_before_market_close(trade_pair)

        # Use current time if not specified
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        # Fetch recent candles
        prev_timestamp = None
        final_agg = None
        timespan = "second"

        # Define a window around the target timestamp
        start_ts = timestamp_ms - 10000  # 10 seconds before
        end_ts = timestamp_ms + 2000  # 2 seconds after

        # Fetch candles using unified fetcher
        try:
            raw = self.unified_candle_fetcher(
                trade_pair,
                start_ts,
                end_ts,
                timespan
            )

            # Process each candle
            for a in raw:
                epoch_milliseconds = a.timestamp
                price_source = self.agg_to_price_source(a, timestamp_ms, timespan)

                # Ensure timestamps are in order
                assert prev_timestamp is None or prev_timestamp < epoch_milliseconds

                final_agg = price_source
                prev_timestamp = epoch_milliseconds

            if not final_agg:
                bt.logging.warning(
                    f"Polygon failed to fetch REST data for {trade_pair.trade_pair} at time "
                    f"{TimeUtil.millis_to_formatted_date_str(timestamp_ms)}. "
                    f"If you keep seeing this warning, report it to the team ASAP"
                )
        except Exception as e:
            bt.logging.error(f"Error fetching candles for {trade_pair.trade_pair}: {e}")
            final_agg = None

        return final_agg

    def trade_pair_to_polygon_ticker(self, trade_pair: TradePair):
        """Convert a trade pair to the corresponding Polygon ticker symbol"""
        if trade_pair.is_crypto:
            return 'X:' + trade_pair.trade_pair_id
        elif trade_pair.is_forex:
            return 'C:' + trade_pair.trade_pair_id
        elif trade_pair.is_indices:
            return 'I:' + trade_pair.trade_pair_id
        elif trade_pair.is_equities:
            return trade_pair.trade_pair_id
        else:
            raise ValueError(f"Unknown trade pair category: {trade_pair.trade_pair_category}")

    def get_event_before_market_close(self, trade_pair: TradePair, end_time_ms=None) -> PriceSource | None:
        """Get the last price before market close"""
        # Check if we already have the price in cache
        if self.closed_market_prices[trade_pair] is not None:
            return self.closed_market_prices[trade_pair]
        elif trade_pair in self.UNSUPPORTED_TRADE_PAIRS:
            return None

        # Determine if we should update the cache
        write_closed_market_prices = False

        # Set time range - default to current time if not specified
        if end_time_ms is None:
            end_time_ms = TimeUtil.now_in_millis()
            write_closed_market_prices = True

        # Look back 7 days to find the last close
        start_time_ms = end_time_ms - 1000 * 60 * 60 * 24 * 7

        # Get daily candles for the period
        candles = self.get_candles_for_trade_pair(
            trade_pair,
            start_time_ms,
            end_time_ms,
            end_time_ms,
            attempting_prev_close=True,
            force_timespan='day'
        )

        # Make sure we found at least one candle
        if len(candles) == 0:
            msg = f"get_event_before_market_close: Failed to fetch market close for {trade_pair.trade_pair}"
            raise ValueError(msg)

        # Return the most recent candle
        ans = candles[-1]

        # Cache the result if requested
        if write_closed_market_prices:
            self.closed_market_prices[trade_pair] = ans

        return ans

    def get_websocket_event(self, trade_pair: TradePair) -> PriceSource | None:
        """Get the latest event from websocket data with staleness check"""
        symbol = trade_pair.trade_pair
        cur_event = self.latest_websocket_events.get(symbol)

        if not cur_event:
            return None

        # Check if the data is too stale
        timestamp_ms = cur_event.end_ms
        max_allowed_lag_s = self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
        lag_s = time.time() - timestamp_ms / 1000.0
        is_stale = lag_s > max_allowed_lag_s

        if is_stale:
            bt.logging.info(
                f"Found stale Polygon websocket data for {trade_pair.trade_pair}. Lag_s: {lag_s} "
                f"seconds. Max allowed lag for category: {max_allowed_lag_s} seconds. Ignoring this data."
            )
            return None

        return cur_event

    def unified_candle_fetcher(self, trade_pair: TradePair, start_timestamp_ms: int, end_timestamp_ms: int,
                               timespan: str = None):
        """Fetch candles with unified error handling and data cleaning"""
        # Make sure REST client is initialized
        if self.POLYGON_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        # Get the Polygon ticker
        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)

        def _fetch_raw_polygon_aggs():
            """Internal function to fetch raw aggregates from Polygon"""
            try:
                return self.POLYGON_CLIENT.list_aggs(
                    polygon_ticker,
                    1,
                    timespan,
                    start_timestamp_ms,
                    end_timestamp_ms,
                    limit=self.N_CANDLES_LIMIT
                )
            except Exception as e:
                bt.logging.error(f"Error fetching Polygon aggregates: {e}")
                return []

        def _intra_vwap_valid(agg):
            """Check if VWAP is close enough to close price"""
            if not hasattr(agg, 'vwap') or agg.vwap is None:
                return False
            return abs(agg.vwap - agg.close) / agg.close < .004

        def _consecutive_candle_spiked(a, b):
            """Check if there's an abnormal price movement between candles"""
            return abs(b.close - a.close) / a.close > .015

        def _agg_to_payload(agg, prev, nxt, last_valid_price, waiting_for_valid_payload, tp_id, verbose=False):
            """Process an aggregate and determine if it's valid"""
            # If we've been waiting too long, accept what we have
            if waiting_for_valid_payload > 20 and prev and nxt:
                return _agg_to_payload(agg, prev, nxt, None, 0, tp_id)
            elif waiting_for_valid_payload > 0:
                # Check if current price is close enough to last valid price
                delta = abs(agg.close - last_valid_price) / last_valid_price
                vv = _intra_vwap_valid(agg)
                price_valid = delta < .005 and vv  # close enough to the tether

                if verbose:
                    if price_valid:
                        print(
                            f'Breaking out. tp_id {tp_id}, waiting_for_valid_payload {waiting_for_valid_payload}, delta {delta} last_valid_price {last_valid_price} agg {agg}')
                    else:
                        print(
                            f'tp_id {tp_id}, waiting_for_valid_payload {waiting_for_valid_payload}, delta {delta} vv {vv} last_valid_price {last_valid_price} agg.close {agg.close} agg.vwap {agg.vwap}')

                return agg, price_valid

            # Forex candles are subject to spikes, particularly at the end of day
            if prev and nxt is None and _consecutive_candle_spiked(prev, agg):
                # Use the previous VWAP for this candle's close
                agg.close = prev.vwap
                return agg, True  # Don't enter the waiting_for_valid_payload state

            elif nxt and prev is None and _consecutive_candle_spiked(agg, nxt):
                # Use the next VWAP for this candle's close
                agg.close = nxt.vwap
                return agg, True  # Don't enter the waiting_for_valid_payload state

            elif _intra_vwap_valid(agg):
                # VWAP and close are close enough
                return agg, True

            else:  # spike detected
                # Try to smooth out the spike
                vv = hasattr(agg, 'low') and hasattr(agg, 'high') and agg.low <= agg.vwap <= agg.high
                smoothed_price = agg.vwap if vv else (agg.high + agg.low) / 2
                agg.close = smoothed_price

                if verbose:
                    print('--------------------------------------')
                    print(
                        f'Rejecting {tp_id}. delta {abs(agg.vwap - agg.close) / agg.close}. vv {vv} last_valid_price {last_valid_price} agg {agg}')

                return agg, False

        def _get_filtered_forex_minute_data():
            """Get and filter forex minute data to remove spikes"""
            price_info_raw = list(_fetch_raw_polygon_aggs())
            n_points = len(price_info_raw)
            last_valid_price = None
            tp_id = trade_pair.trade_pair_id
            waiting_for_valid_payload = 0  # minutes we've been ignoring data
            ans = []

            self.tp_to_mfs = {}

            for i in range(n_points):
                a = price_info_raw[i]

                prev = price_info_raw[i - 1] if i > 0 else None
                nxt = price_info_raw[i + 1] if i < n_points - 1 else None

                agg, is_valid = _agg_to_payload(a, prev, nxt, last_valid_price, waiting_for_valid_payload, tp_id)

                if is_valid:
                    last_valid_price = agg.close
                    waiting_for_valid_payload = 0
                    ans.append(agg)
                else:
                    if last_valid_price:
                        waiting_for_valid_payload += 1
                    else:
                        ans.append(agg)  # smoothed price

                    # Track the longest wait for diagnostics
                    if tp_id not in self.tp_to_mfs:
                        self.tp_to_mfs[tp_id] = waiting_for_valid_payload
                    else:
                        self.tp_to_mfs[tp_id] = max(self.tp_to_mfs[tp_id], waiting_for_valid_payload)

            return ans

        def _get_filtered_forex_second_data():
            """Get and filter forex second data from quotes"""
            ans = []
            prev_t_ms = None

            try:
                raw = self.POLYGON_CLIENT.list_quotes(
                    ticker=polygon_ticker,
                    timestamp_gte=start_timestamp_ms * 1000000,
                    timestamp_lte=end_timestamp_ms * 1000000,
                    sort='participant_timestamp',
                    order='asc',
                    limit=self.N_CANDLES_LIMIT
                )
            except Exception as e:
                bt.logging.error(f"Error fetching quotes: {e}")
                return [], 0

            n_quotes = 0
            best_delta = float('inf')

            for r in raw:
                t_ms = r.participant_timestamp // 1000000

                # Reset when timestamp changes
                if t_ms != prev_t_ms:
                    best_delta = float('inf')
                    if ans and hasattr(ans[-1], 'temp'):
                        del ans[-1].temp

                n_quotes += 1

                # Get bid/ask prices
                bid, ask, current_delta = self.parse_price_for_forex(r, stats=None)
                if bid is None:
                    continue

                midpoint_price = (bid + ask) / 2.0

                if best_delta == float('inf'):
                    # First quote for this timestamp
                    best_delta = current_delta
                    ans.append(Agg(
                        open=midpoint_price,
                        close=midpoint_price,
                        high=midpoint_price,
                        low=midpoint_price,
                        vwap=None,
                        timestamp=t_ms,
                        bid=bid,
                        ask=ask,
                        volume=0
                    ))
                    # Store bid/ask arrays for median calculation
                    ans[-1].temp = ([bid], [ask])
                else:
                    # Update existing aggregate with new quote
                    best_delta = current_delta
                    dat = ans[-1].temp
                    dat[0].append(bid)
                    dat[0].sort()
                    dat[1].append(ask)
                    dat[1].sort()

                    # Calculate median values
                    median_bid = RecentEventTracker.forex_median_price(dat[0])
                    median_ask = RecentEventTracker.forex_median_price(dat[1])
                    midpoint_price = (median_bid + median_ask) / 2.0

                    # Update the aggregate
                    ans[-1].open = ans[-1].close = ans[-1].low = ans[-1].high = midpoint_price
                    ans[-1].bid = median_bid
                    ans[-1].ask = median_ask

                prev_t_ms = t_ms

            return ans, n_quotes

        # Choose the appropriate data fetching based on asset type and timespan
        if trade_pair.is_forex:
            if timespan == 'second':
                return _get_filtered_forex_second_data()[0]
            elif timespan == 'minute':
                return _get_filtered_forex_minute_data()
            elif timespan == 'day':
                return _fetch_raw_polygon_aggs()
            else:
                raise Exception(f'Invalid timespan {timespan}')
        else:
            return _fetch_raw_polygon_aggs()

    def get_candles_for_trade_pair(
            self,
            trade_pair: TradePair,
            start_timestamp_ms: int,
            end_timestamp_ms: int,
            target_timestamp_ms: int,
            attempting_prev_close: bool = False,
            force_timespan: str = None
    ) -> list[PriceSource] | None:
        """Get candles for a trade pair with automatic timespan selection"""
        # Determine appropriate timespan based on time range
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

        # Override with forced timespan if provided
        if force_timespan:
            timespan = force_timespan

        # Fetch candles
        aggs = []
        prev_timestamp = None

        try:
            raw = self.unified_candle_fetcher(trade_pair, start_timestamp_ms, end_timestamp_ms, timespan)

            for i, a in enumerate(raw):
                epoch_miliseconds = a.timestamp

                # Ensure candles are sorted by time
                assert prev_timestamp is None or epoch_miliseconds >= prev_timestamp, (
                    'candles not sorted', prev_timestamp, epoch_miliseconds
                )

                # Convert to price source
                price_source = self.agg_to_price_source(
                    a,
                    target_timestamp_ms,
                    timespan,
                    attempting_prev_close=attempting_prev_close
                )

                aggs.append(price_source)
                prev_timestamp = epoch_miliseconds

        except Exception as e:
            bt.logging.error(f"Error fetching candles for {trade_pair.trade_pair}: {e}")

        if not aggs:
            bt.logging.trace(
                f"{POLYGON_PROVIDER_NAME} failed to fetch candle data for {trade_pair.trade_pair}. "
                f"Perhaps this trade pair was closed during the specified window."
            )

        return aggs

    def get_quote(self, trade_pair: TradePair, processed_ms: int) -> (float, float, int):
        """Get the bid and ask quote for a trade pair at a specific time"""
        # Initialize REST client if needed
        if self.POLYGON_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        if trade_pair.is_forex or trade_pair.is_equities:
            try:
                polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)

                quotes = self.POLYGON_CLIENT.list_quotes(
                    ticker=polygon_ticker,
                    timestamp_lte=processed_ms * 1_000_000,
                    sort="participant_timestamp",
                    order="desc",
                    limit=1
                )

                for q in quotes:
                    return q.bid_price, q.ask_price, int(q.participant_timestamp / 1_000_000)  # convert ns to ms
            except Exception as e:
                bt.logging.error(f"Error fetching quotes for {trade_pair.trade_pair}: {e}")

        # Default return for crypto or error cases
        return 0, 0, 0

    def get_currency_conversion(self, trade_pair: TradePair = None, base: str = None, quote: str = None) -> float:
        """Get the currency conversion rate from base currency to quote currency"""
        # Initialize REST client if needed
        if self.POLYGON_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        # Validate inputs
        if not (base and quote):
            if trade_pair and trade_pair.is_forex:
                base, quote = trade_pair.trade_pair.split("/")
            else:
                raise ValueError("Must provide either a valid forex pair or a base and quote for currency conversion")

        try:
            # Get the conversion rate
            rate = self.POLYGON_CLIENT.get_real_time_currency_conversion(
                from_=base,
                to=quote,
                precision=4,
            )

            return rate.converted
        except Exception as e:
            bt.logging.error(f"Error getting currency conversion {base}/{quote}: {e}")
            return 0.0


if __name__ == "__main__":

    secrets = ValiUtils.get_secrets()

    polygon_data_provider = PolygonDataService(api_key=secrets['polygon_apikey'], disable_ws=True)

    ans = asyncio.run(polygon_data_provider.get_close_rest(TradePair.USDJPY, 1742577204000))
    print(ans)
    #ans = asyncio.run(polygon_data_provider.get_closes_rest([TradePair.AUDCHF, TradePair.BTCUSD, TradePair.TSLA, TradePair.CADCHF]))
    time.sleep(10000)
    for tp in TradePair:
        if tp.is_indices:
            continue
        #if tp != TradePair.GBPUSD:
        #    continue

        print('PRICE BEFORE MARKET CLOSE: ', polygon_data_provider.get_event_before_market_close(tp))
        print('getting close for', tp.trade_pair_id, ':', polygon_data_provider.get_close_rest(tp))

    time.sleep(100000)

    polygon_data_provider = PolygonDataService(api_key=secrets['polygon_apikey'], disable_ws=True)
    target_timestamp_ms = 1735088280163#1715288494000

    """
    aggs = []
    for a in RESTClient(secrets['polygon_apikey']).list_aggs(
            "X:BNBUSD",
            1,
            "day",
            "2023-01-30",
            "2023-02-03",
            limit=50000,
    ):
        aggs.append(a)

    assert 0, aggs
    """


    #tp = TradePair.TSLA
    # Initialize client
    #aggs = polygon_data_provider.get_close_at_date_second(tp, target_timestamp_ms, return_aggs=True)
    import numpy as np
    #uu = {a.timestamp: [a] for a in aggs}
    for tp in [TradePair.AUDJPY]:#[x for x in TradePair if x.is_forex]:
        t0 = time.time()
        quotes = polygon_data_provider.unified_candle_fetcher(tp,
                                                              target_timestamp_ms - 1000 * 60 * 15,
                                                              target_timestamp_ms + 1000 * 60 * 15,
                                                              "minute")
        for q in quotes:
            print(TimeUtil.millis_to_formatted_date_str(q.timestamp), q)
        assert 0
        quotes = list(quotes)
        n_quotes = len(quotes)
        n_spikes = 0
        for prev, next in zip(quotes, quotes[1:]):
            delta_close = abs(prev.close - next.open) / prev.close
            if delta_close > .01:
                n_spikes += 1
                time_of_spike = TimeUtil.millis_to_verbose_formatted_date_str(next.timestamp)
                print(time_of_spike, delta_close, prev.close, next.close, prev, next)
        print(f'tp {tp.trade_pair} Found {n_spikes} spikes in {n_quotes} quotes. pct: {n_spikes / n_quotes * 100:.4f}')

        continue
        print(f'fetched data for {tp.trade_pair_id} in {time.time() - t0} s. quotes: {len(quotes)}')
        deltas = []
        n_wonky = 0
        worst_offender = None
        worst_delta = 0
        for i,  q in enumerate(quotes):
            if 1:#q.low <= q.vwap <= q.high:
                delta = abs(q.vwap - q.close) / q.close
                #assert delta > 0, q
                deltas.append(delta)
                if delta > worst_delta:
                    worst_delta = delta
                    worst_offender = q
            else:
                n_wonky += 1




        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        min_delta = np.min(deltas)
        max_delta = np.max(deltas)
        q1 = np.percentile(deltas, 25)
        median_delta = np.median(deltas)
        q3 = np.percentile(deltas, 75)

        threshold_filter = .002
        n_deltas_meeting_threshold = len([x for x in deltas if x >= threshold_filter])
        ptbf = n_deltas_meeting_threshold / len(deltas) * 100
        if ptbf > .1:
            print(f"Statistics for {tp.trade_pair_id} deltas {len(deltas)}. percent to be filtered out: {ptbf:.4f}")
            print(f"  Mean: {mean_delta:.4f}  Standard Deviation: {std_delta:.4f}")
            print(f"  Min: {min_delta:.4f}  Max: {max_delta:.4f}")
            print(f"  25th Percentile (Q1): {q1:.4f}  Median: {median_delta:.4f}  75th Percentile (Q3): {q3:.4f}")
            # print the worst offender
            print(f"  Worst offender: {TimeUtil.timestamp_ms_to_eastern_time_str(worst_offender.timestamp)} "
                  f"vwap: {worst_offender.vwap} low: {worst_offender.low} high: {worst_offender.high}"
                  f" raw: {worst_offender}")





    ##trades = polygon_data_provider.POLYGON_CLIENT.list_trades(ticker='C:CAD-JPY',
    #                                                         timestamp_gt=target_timestamp_ms * 1000000 - 1000 * 1000000 * 10,
    #                                                         timestamp_lt=target_timestamp_ms * 1000000 + 1000 * 1000000 * 10)
    #for trade in trades:
    #    print('trade', trade)

    assert 0

    # 'X:BTC-USD'
    trades = polygon_data_provider.POLYGON_CLIENT.list_trades(
        ticker='C:CAD-CHF',
        params={
        "timestamp.gte": target_timestamp_ms * 1000 - 1000 * 1000000,
        "timestamp.lte": target_timestamp_ms * 1000 + 1000 * 1000000
    })
    exchange_to_name = {1: 'Coinbase', 23: 'Kraken', 2: 'Bitfinex'}
    for trade in trades:
        participant_timestamp_ms = trade.participant_timestamp / 1000000.0
        format_date = TimeUtil.millis_to_formatted_date_str(participant_timestamp_ms)
        print(format_date, exchange_to_name.get(trade.exchange), trade.price)



    #price, time_delta = polygon_data_provider.get_close_at_date_second(trade_pair=TradePair.BTCUSD, target_timestamp_ms=1712671378202)


    print(price, time_delta)  # noqa: F821
    assert 0

    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SPX, TradePair.GBPUSD, TradePair.DJI]
    print("-----------------REST-----------------")
    ans_rest = polygon_data_provider.get_closes_rest(trade_pairs)
    for k, v in ans_rest.items():
        print(f"@@@{k}@@@: {v}")
    assert 0
    now = TimeUtil.now_in_millis()
    while True:
        # use trade_pair_to_recent_events to get the closest event to "now"
        for tp in TradePair:
            symbol = tp.trade_pair
            closest_event = polygon_data_provider.trade_pair_to_recent_events[symbol].get_closest_event(now)
            n_events = polygon_data_provider.trade_pair_to_recent_events[symbol].count_events()
            delta_time_s = (now - closest_event.start_ms) / 1000.0 if closest_event else None
            print(f"Closest event to {TimeUtil.millis_to_formatted_date_str(now)} for {tp.trade_pair_id}: {closest_event}. Total_n_events: {n_events}. Lag (s): {delta_time_s}")
        time.sleep(10)

    #time.sleep(12)
    print(polygon_data_provider.get_close_at_date_second(TradePair.BTCUSD, 1712671378202))
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
        print('main thread perspective - n_events_global:', n_events_global)  # noqa: F821
        for k, price in latest_websocket_events.items():  # noqa: F821
            t = last_websocket_ping_time_s[k]  # noqa: F821
            cached_price = cached_prices.get(k, None)
            cached_time = cached_times.get(k, None)
            if cached_price and cached_time:
                assert cached_time <= t, (cached_time, t)

            cached_prices[k] = price
            cached_times[k] = t
        debug_log()  # noqa: F821
        time.sleep(10)
