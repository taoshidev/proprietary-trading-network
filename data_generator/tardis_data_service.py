import time
import traceback
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import bittensor as bt
from tardis_client import TardisClient, Channel

from data_generator.base_data_service import BaseDataService
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
from vali_objects.vali_dataclasses.price_source import PriceSource

TARDIS_PROVIDER_NAME = "Tardis"

class TardisDataService(BaseDataService):
    def __init__(self, api_key, ipc_manager=None):
        self.init_time = time.time()
        self._api_key = api_key
        self.tardis_client = None
        
        super().__init__(provider_name=TARDIS_PROVIDER_NAME, ipc_manager=ipc_manager)
        
        # Mapping of trade pairs to exchange and symbol
        self.trade_pair_to_exchange_map = {
            TradePair.BTCUSD: ("binance", "btcusdt"),
            TradePair.ETHUSD: ("binance", "ethusdt"),
            TradePair.SOLUSD: ("binance", "solusdt"),
            TradePair.XRPUSD: ("binance", "xrpusdt"),
            TradePair.DOGEUSD: ("binance", "dogeusdt"),
        }
        
    def instantiate_not_pickleable_objects(self):
        """Instantiate the TardisClient object when needed."""
        self.tardis_client = TardisClient(api_key=self._api_key)

    async def get_market_data(self, trade_pair: TradePair, timestamp_ms: int) -> Optional[Tuple[float, float]]:
        """
        Fetch historical market data (bid/ask) for the given trade pair at the specified timestamp.
        
        Args:
            trade_pair: The cryptocurrency trade pair
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            Tuple containing (bid, ask) prices or None if data couldn't be fetched
        """
        if self.tardis_client is None:
            self.instantiate_not_pickleable_objects()
            
        if trade_pair not in self.trade_pair_to_exchange_map:
            bt.logging.warning(f"No exchange mapping found for {trade_pair.trade_pair}")
            return None
            
        exchange, symbol = self.trade_pair_to_exchange_map[trade_pair]
        
        # Convert timestamp to datetime
        timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000)
        
        # Create a time window (15 seconds before and after the timestamp)
        from_date = timestamp_dt - timedelta(seconds=15)
        to_date = timestamp_dt + timedelta(seconds=15)
        
        try:
            # Get book snapshot data from tardis
            messages = []
            
            async for local_timestamp, message in self.tardis_client.replay(
                exchange=exchange,
                from_date=from_date.strftime('%Y-%m-%dT%H:%M:%S'),
                to_date=to_date.strftime('%Y-%m-%dT%H:%M:%S'),
                filters=[Channel(name="bookTicker", symbols=[symbol])]
            ):
                messages.append(message)
                
            if not messages:
                bt.logging.warning(f"No data found for {trade_pair.trade_pair} at {from_date.strftime('%Y-%m-%d')}-{to_date.strftime('%Y-%m-%d')}")
                return None
                
            # # Find the closest message to our target timestamp
            # closest_message = None
            # min_time_diff = float('inf')
            #
            # for message in messages:
            #     if 'data' in message and 'b' in message['data'] and 'a' in message['data']:
            #         message_time = datetime.fromtimestamp(message['localTimestamp'] / 1000)
            #         time_diff = abs((message_time - timestamp_dt).total_seconds())
            #
            #         if time_diff < min_time_diff:
            #             min_time_diff = time_diff
            #             closest_message = message
            closest_message = messages[len(messages)//2]

            if closest_message and 'data' in closest_message:
                data = closest_message['data']
                
                # Extract the best bid and ask
                if 'b' in data and len(data['b']) > 0 and 'a' in data and len(data['a']) > 0:
                    best_bid = float(data['b'])
                    best_ask = float(data['a'])
                    return best_bid, best_ask
            
            return None
            
        except Exception as e:
            bt.logging.error(f"Error fetching data from Tardis for {trade_pair.trade_pair}: {e}")
            bt.logging.error(traceback.format_exc())
            return None

    def get_close_rest(self, trade_pair: TradePair, timestamp_ms: int = None) -> Optional[PriceSource]:
        """
        Get a PriceSource object with bid/ask data for the given trade pair at the specified timestamp.
        Used for REST API operations.
        
        Args:
            trade_pair: The trade pair to get data for
            timestamp_ms: The timestamp in milliseconds to get data for (defaults to now)
            
        Returns:
            PriceSource object or None if data couldn't be fetched
        """
        if not trade_pair.is_crypto:
            return None
            
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()
            
        # Run the async function to get the market data
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            bid_ask = loop.run_until_complete(self.get_market_data(trade_pair, timestamp_ms))
            loop.close()
        except Exception as e:
            bt.logging.error(f"Error running async loop for {trade_pair.trade_pair}: {e}")
            return None
        
        if not bid_ask:
            return None
            
        bid, ask = bid_ask
        mid_price = (bid + ask) / 2
        
        now_ms = TimeUtil.now_in_millis()
        return PriceSource(
            source=f'{TARDIS_PROVIDER_NAME}_rest',
            timespan_ms=0,
            open=mid_price,
            close=mid_price,
            vwap=mid_price,
            high=mid_price,
            low=mid_price,
            start_ms=timestamp_ms,
            websocket=False,
            lag_ms=now_ms - timestamp_ms,
            bid=bid,
            ask=ask
        )
        
    def get_closes_rest(self, trade_pairs: List[TradePair]) -> dict:
        """
        Get close prices for multiple trade pairs at once.
        
        Args:
            trade_pairs: List of trade pairs to get data for
            
        Returns:
            Dictionary mapping trade pairs to their PriceSource objects
        """
        result = {}
        for trade_pair in trade_pairs:
            if trade_pair.is_crypto:
                price_source = self.get_close_rest(trade_pair)
                if price_source:
                    result[trade_pair] = price_source
        return result

    def _create_websocket_client(self, tpc: TradePairCategory):
        """Not implemented for Tardis as we use REST API for historical data."""
        pass

    def _subscribe_websockets(self, tpc: TradePairCategory):
        """Not implemented for Tardis as we use REST API for historical data."""
        pass

    async def handle_msg(self, msg):
        """Not implemented for Tardis as we use REST API for historical data."""
        pass
        
    def get_quote(self, trade_pair: TradePair, processed_ms: int) -> Tuple[float, float, int]:
        """
        Get bid and ask quotes for a trade pair at a specific time.
        
        Args:
            trade_pair: The trade pair to get quotes for
            processed_ms: The timestamp in milliseconds
            
        Returns:
            Tuple of (bid, ask, timestamp_ms)
        """
        if not trade_pair.is_crypto:
            return 0, 0, 0
            
        # Run the async function to get the market data
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            bid_ask = loop.run_until_complete(self.get_market_data(trade_pair, processed_ms))
            loop.close()
        except Exception as e:
            bt.logging.error(f"Error running async loop for quote {trade_pair.trade_pair}: {e}")
            return 0, 0, 0
            
        if not bid_ask:
            return 0, 0, 0
            
        bid, ask = bid_ask
        return bid, ask, processed_ms
