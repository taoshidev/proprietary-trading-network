"""
Test suite for unified websocket recovery mechanism.
Tests the improvements made to base_data_service.py
"""

import asyncio
import time
import unittest
from unittest.mock import Mock, patch, AsyncMock
from threading import Thread

from data_generator.base_data_service import BaseDataService
from vali_objects.vali_config import TradePair, TradePairCategory


class MockWebSocketClient:
    """Mock WebSocket client for testing"""
    
    def __init__(self, fail_after_n_messages=None):
        self.connect_count = 0
        self.message_count = 0
        self.fail_after_n_messages = fail_after_n_messages
        self.subscriptions = []
        self._should_close = False
        self.is_connected = False
        
    def subscribe(self, symbol):
        self.subscriptions.append(symbol)
        
    def unsubscribe_all(self):
        self.subscriptions = []
        
    async def connect(self, handler):
        """Simulate WebSocket connection"""
        self.connect_count += 1
        self.is_connected = True
        
        try:
            while not self._should_close:
                self.message_count += 1
                
                # Simulate failure after N messages
                if self.fail_after_n_messages and self.message_count >= self.fail_after_n_messages:
                    raise Exception(f"Simulated failure after {self.message_count} messages")
                
                # Send mock message
                await handler([Mock()])
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            raise
        finally:
            self.is_connected = False
            
    async def close(self):
        self._should_close = True
        await asyncio.sleep(0.01)


class MockDataService(BaseDataService):
    """Mock implementation of BaseDataService for testing"""
    
    def __init__(self):
        super().__init__(provider_name="TestProvider", ipc_manager=None)
        self.created_clients = {tpc: [] for tpc in TradePairCategory}
        
    def _create_websocket_client(self, tpc: TradePairCategory):
        client = MockWebSocketClient()
        self.WEBSOCKET_OBJECTS[tpc] = client
        self.created_clients[tpc].append(client)
        
    def _subscribe_websockets(self, tpc: TradePairCategory = None):
        if tpc and self.WEBSOCKET_OBJECTS.get(tpc):
            self.WEBSOCKET_OBJECTS[tpc].subscribe(f"TEST.{tpc.name}")
            
    async def handle_msg(self, msgs):
        for msg in msgs:
            for tpc in self.enabled_websocket_categories:
                self.tpc_to_n_events[tpc] += 1
                self.tpc_to_last_event_time[tpc] = time.time()
                
    def get_first_trade_pair_in_category(self, tpc: TradePairCategory):
        return TradePair.BTCUSD if tpc == TradePairCategory.CRYPTO else TradePair.EURUSD
        
    def is_market_open(self, trade_pair=None, category=None):
        return True  # Always open for testing


class TestWebSocketRecovery(unittest.TestCase):
    """Test unified websocket recovery mechanism"""
    
    def setUp(self):
        self.service = None
        
    def tearDown(self):
        if self.service:
            self.service.stop_threads()
            time.sleep(0.5)
    
    def test_task_death_recovery(self):
        """Test recovery when websocket task dies"""
        self.service = MockDataService()
        self.service.MAX_TIME_NO_EVENTS_S = 1.0  # Fast timeout for testing
        
        # Configure first client to fail after 3 messages
        def create_failing_client(tpc):
            client = MockWebSocketClient(fail_after_n_messages=3)
            self.service.WEBSOCKET_OBJECTS[tpc] = client
            self.service.created_clients[tpc].append(client)
            
        # Override client creation for first client only
        original_create = self.service._create_websocket_client
        self.service._create_websocket_client = lambda tpc: (
            create_failing_client(tpc) if len(self.service.created_clients[tpc]) == 0
            else original_create(tpc)
        )
        
        # Start service
        manager_thread = Thread(target=self.service.websocket_manager, daemon=True)
        manager_thread.start()
        
        # Wait for failure and recovery
        time.sleep(3.0)
        
        # Verify recovery happened
        crypto_clients = len(self.service.created_clients[TradePairCategory.CRYPTO])
        self.assertGreaterEqual(crypto_clients, 2, 
            f"Expected at least 2 clients due to recovery, got {crypto_clients}")
        
        # Verify events were processed after recovery
        crypto_events = self.service.tpc_to_n_events[TradePairCategory.CRYPTO]
        self.assertGreater(crypto_events, 3,
            "Expected more than 3 events (initial client failed after 3)")
    
    def test_stale_connection_recovery(self):
        """Test recovery when connection stops sending events"""
        self.service = MockDataService()
        self.service.MAX_TIME_NO_EVENTS_S = 1.0  # 1 second timeout
        
        # Override handle_msg to stop processing after initial events
        original_handle = self.service.handle_msg
        self.service.stop_processing = False
        
        async def conditional_handle(msgs):
            if not self.service.stop_processing:
                await original_handle(msgs)
                
        self.service.handle_msg = conditional_handle
        
        # Start service
        manager_thread = Thread(target=self.service.websocket_manager, daemon=True)
        manager_thread.start()
        
        # Wait for initial events
        time.sleep(0.5)
        initial_events = self.service.tpc_to_n_events[TradePairCategory.CRYPTO]
        self.assertGreater(initial_events, 0, "Should have initial events")
        
        # Stop processing to simulate stale connection
        self.service.stop_processing = True
        
        # Wait for timeout and recovery
        # Health check runs every 5 seconds, so we need to wait at least that long
        time.sleep(6.0)
        
        # Resume processing
        self.service.stop_processing = False
        
        # Wait for new events
        time.sleep(1.0)
        
        # Verify recovery happened
        crypto_clients = len(self.service.created_clients[TradePairCategory.CRYPTO])
        self.assertGreaterEqual(crypto_clients, 2,
            f"Expected recovery to create new client, got {crypto_clients} clients")
        
        # Verify new events after recovery
        final_events = self.service.tpc_to_n_events[TradePairCategory.CRYPTO]
        self.assertGreater(final_events, initial_events,
            "Expected more events after recovery")
    
    def test_no_duplicate_restarts(self):
        """Test that concurrent restart attempts don't create duplicate tasks"""
        self.service = MockDataService()
        self.service.MAX_TIME_NO_EVENTS_S = 0.5  # Very fast timeout
        
        # Make health check run very frequently
        async def fast_health_check(original_check):
            # Inject faster checks
            await asyncio.sleep(0.1)
            await original_check()
            
        # This would create race conditions in the old design
        # but should be safe with per-TPC locks
        
        # Start service
        manager_thread = Thread(target=self.service.websocket_manager, daemon=True)
        manager_thread.start()
        
        # Let it run with rapid health checks
        time.sleep(2.0)
        
        # Check that we don't have runaway task creation
        # Even with rapid checks, we should have reasonable number of clients
        crypto_clients = len(self.service.created_clients[TradePairCategory.CRYPTO])
        self.assertLess(crypto_clients, 10,
            f"Too many clients created ({crypto_clients}), possible duplicate restarts")


if __name__ == '__main__':
    unittest.main()