
import unittest
from copy import deepcopy

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.mock_classes import MockLivePriceFetcher
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.limit_order_manager import LimitOrderManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order, ORDER_SRC_LIMIT_UNFILLED, ORDER_SRC_LIMIT_FILLED, ORDER_SRC_LIMIT_CANCELLED
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestLimitOrders(TestBase):

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(None, None, self.mock_metagraph)
        self.perf_ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)

        self.position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            perf_ledger_manager=self.perf_ledger_manager,
            elimination_manager=self.elimination_manager,
            running_unit_tests=True
        )

        self.position_locks = PositionLocks({})

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)

        self.limit_order_manager = LimitOrderManager(
            position_manager=self.position_manager,
            live_price_fetcher=self.live_price_fetcher,
            running_unit_tests=True
        )

        self.position_manager.clear_all_miner_positions()
        self.limit_order_manager.limit_orders.clear()

    def create_test_limit_order(self, order_type=OrderType.LONG, limit_price=49000.0,
                               trade_pair=None, leverage=0.5, order_uuid=None):
        """Helper to create test limit orders"""
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if order_uuid is None:
            order_uuid = f"test_limit_order_{TimeUtil.now_in_millis()}"

        return Order(
            trade_pair=trade_pair,
            order_uuid=order_uuid,
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=order_type,
            leverage=leverage,
            execution_type=ExecutionType.LIMIT,
            limit_price=limit_price,
            src=ORDER_SRC_LIMIT_UNFILLED
        )

    def create_test_price_sources(self, trigger_price, stable_price=None, num_sources=3):
        """Helper to create price sources for limit order testing"""
        if stable_price is None:
            stable_price = trigger_price

        now_ms = TimeUtil.now_in_millis()
        sources = []
        buffer_ms = 10 * 1000  # ValiConfig.LIMIT_ORDER_PRICE_BUFFER_MS

        # First price source triggers the order
        sources.append(PriceSource(
            source='test', timespan_ms=0, open=trigger_price, close=trigger_price,
            vwap=None, high=trigger_price, low=trigger_price,
            start_ms=now_ms, websocket=True, lag_ms=100
        ))

        # Additional price sources within buffer window
        for i in range(1, num_sources - 1):
            sources.append(PriceSource(
                source='test', timespan_ms=0, open=stable_price, close=stable_price,
                vwap=None, high=stable_price, low=stable_price,
                start_ms=now_ms + (i * 3000), websocket=True, lag_ms=100  # 3 second intervals
            ))

        # Last price source must be beyond the buffer window
        sources.append(PriceSource(
            source='test', timespan_ms=0, open=stable_price, close=stable_price,
            vwap=None, high=stable_price, low=stable_price,
            start_ms=now_ms + buffer_ms + 1000, websocket=True, lag_ms=100  # 1 second after buffer
        ))

        return sources

    def create_test_position(self, trade_pair=None, miner_hotkey=None):
        """Helper to create test positions"""
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if miner_hotkey is None:
            miner_hotkey = self.DEFAULT_MINER_HOTKEY

        return Position(
            miner_hotkey=miner_hotkey,
            position_uuid=f"pos_{TimeUtil.now_in_millis()}",
            open_ms=TimeUtil.now_in_millis(),
            trade_pair=trade_pair
        )

    def test_save_limit_order_basic(self):
        """Test basic limit order storage"""
        limit_order = self.create_test_limit_order()

        self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order_uuid, limit_order.order_uuid)
        self.assertEqual(orders[0].src, ORDER_SRC_LIMIT_UNFILLED)

    def test_save_limit_order_exceeds_maximum(self):
        """Test limit order rejection when exceeding maximum unfilled orders"""
        for i in range(ValiConfig.MAX_UNFILLED_LIMIT_ORDERS):
            limit_order = self.create_test_limit_order(
                order_uuid=f"test_order_{i}"
            )
            self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), ValiConfig.MAX_UNFILLED_LIMIT_ORDERS)

        excess_order = self.create_test_limit_order(
            order_uuid=f"test_order_excess"
        )

        with self.assertRaises(Exception) as context:
            self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, excess_order)
        self.assertIn("too many unfilled limit orders", str(context.exception))

    def test_save_flat_limit_order_no_position(self):
        """Test FLAT limit order rejection when no position exists"""
        flat_limit_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=51000.0
        )

        with self.assertRaises(SignalException) as context:
            self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, flat_limit_order)
        self.assertIn("No position found for FLAT order", str(context.exception))

    def test_save_flat_limit_order_with_position(self):
        """Test FLAT limit order acceptance when position exists"""
        position = self.create_test_position()
        self.position_manager.save_miner_position(position)

        flat_limit_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=51000.0
        )

        self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, flat_limit_order)

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order_type, OrderType.FLAT)

    def test_should_fill_long_limit_order(self):
        """Test LONG limit order fill conditions"""
        limit_order = self.create_test_limit_order(
            order_type=OrderType.LONG,
            limit_price=50500.0
        )

        price_sources = [PriceSource(
            source='test', timespan_ms=0, open=50600.0, close=50600.0,
            vwap=None, high=50600.0, low=50600.0,
            start_ms=TimeUtil.now_in_millis(), websocket=True, lag_ms=100
        )]
        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

        price_sources[0].open = 50500.0
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

        price_sources[0].open = 50400.0
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

    def test_should_fill_short_limit_order(self):
        """Test SHORT limit order fill conditions"""
        limit_order = self.create_test_limit_order(
            order_type=OrderType.SHORT,
            limit_price=49500.0
        )

        # Price below limit - should not fill
        price_sources = self.create_test_price_sources(49400.0)
        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

        # Price at limit - should fill
        price_sources = self.create_test_price_sources(49500.0)
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

        # Price above limit - should fill
        price_sources = self.create_test_price_sources(49600.0)
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

    def test_should_fill_flat_limit_order_long_position(self):
        """Test FLAT limit order fill conditions for LONG position"""
        position = self.create_test_position()
        position.position_type = OrderType.LONG

        flat_limit_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=50500.0
        )

        # Price below limit - should not fill LONG position
        price_sources = self.create_test_price_sources(50400.0)
        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(flat_limit_order, position, price_sources))

        # Price at limit - should fill
        price_sources = self.create_test_price_sources(50500.0)
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(flat_limit_order, position, price_sources))

        # Price above limit - should fill
        price_sources = self.create_test_price_sources(50600.0)
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(flat_limit_order, position, price_sources))

    def test_should_fill_flat_limit_order_short_position(self):
        """Test FLAT limit order fill conditions for SHORT position"""
        position = self.create_test_position()
        position.position_type = OrderType.SHORT

        flat_limit_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=49500.0
        )

        # Price above limit - should not fill SHORT position
        price_sources = self.create_test_price_sources(49600.0)
        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(flat_limit_order, position, price_sources))

        # Price at limit - should fill
        price_sources = self.create_test_price_sources(49500.0)
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(flat_limit_order, position, price_sources))

        # Price below limit - should fill
        price_sources = self.create_test_price_sources(49400.0)
        self.assertTrue(self.limit_order_manager._evaluate_fill_price_source(flat_limit_order, position, price_sources))

    def test_should_not_fill_already_filled_order(self):
        """Test that already filled orders are not processed"""
        limit_order = self.create_test_limit_order()
        limit_order.src = ORDER_SRC_LIMIT_FILLED

        price_sources = [PriceSource(
            source='test', timespan_ms=0, open=40000.0, close=40000.0,
            vwap=None, high=40000.0, low=40000.0,
            start_ms=TimeUtil.now_in_millis(), websocket=True, lag_ms=100
        )]

        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

    def test_should_not_fill_cancelled_order(self):
        """Test that cancelled orders are not processed"""
        limit_order = self.create_test_limit_order()
        limit_order.src = ORDER_SRC_LIMIT_CANCELLED

        price_sources = [PriceSource(
            source='test', timespan_ms=0, open=40000.0, close=40000.0,
            vwap=None, high=40000.0, low=40000.0,
            start_ms=TimeUtil.now_in_millis(), websocket=True, lag_ms=100
        )]

        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources))

    def test_should_not_fill_without_price_sources(self):
        """Test that orders are not filled without price sources"""
        limit_order = self.create_test_limit_order()

        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, None))
        self.assertFalse(self.limit_order_manager._evaluate_fill_price_source(limit_order, None, []))

    def test_limit_order_evaluation_counter(self):
        """Test that limit order evaluation counter is incremented"""
        limit_order = self.create_test_limit_order()

        price_sources = [PriceSource(
            source='test', timespan_ms=0, open=50000.0, close=50000.0,
            vwap=None, high=50000.0, low=50000.0,
            start_ms=TimeUtil.now_in_millis(), websocket=True, lag_ms=100
        )]

        initial_count = self.limit_order_manager._limit_orders_evaluated

        self.limit_order_manager._evaluate_fill_price_source(limit_order, None, price_sources)

        self.assertEqual(self.limit_order_manager._limit_orders_evaluated, initial_count + 1)

    def test_multiple_limit_orders_storage(self):
        """Test storing multiple limit orders for different trade pairs"""
        btc_order = self.create_test_limit_order(
            trade_pair=TradePair.BTCUSD,
            order_uuid="btc_order"
        )
        eth_order = self.create_test_limit_order(
            trade_pair=TradePair.ETHUSD,
            order_uuid="eth_order"
        )

        self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, btc_order)
        self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, eth_order)

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 2)

        order_uuids = [o.order_uuid for o in orders]
        self.assertIn("btc_order", order_uuids)
        self.assertIn("eth_order", order_uuids)

    def test_process_limit_orders_no_orders(self):
        """Test processing when no limit orders exist"""
        tp_to_price_sources = {
            TradePair.BTCUSD: [PriceSource(
                source='test', timespan_ms=0, open=50000.0, close=50000.0,
                vwap=None, high=50000.0, low=50000.0,
                start_ms=TimeUtil.now_in_millis(), websocket=True, lag_ms=100
            )]
        }

        self.limit_order_manager.check_limit_orders(
            self.position_locks
        )

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 0)

    def test_process_limit_orders_no_price_sources(self):
        """Test processing limit orders without matching price sources"""
        limit_order = self.create_test_limit_order(trade_pair=TradePair.BTCUSD)
        self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        tp_to_price_sources = {
            TradePair.ETHUSD: [PriceSource(
                source='test', timespan_ms=0, open=3000.0, close=3000.0,
                vwap=None, high=3000.0, low=3000.0,
                start_ms=TimeUtil.now_in_millis(), websocket=True, lag_ms=100
            )]
        }

        self.limit_order_manager.check_limit_orders(
            self.position_locks
        )

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].src, ORDER_SRC_LIMIT_UNFILLED)

    def test_limit_order_state_transitions(self):
        """Test that limit orders transition through states correctly"""
        limit_order = self.create_test_limit_order()
        self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(orders[0].src, ORDER_SRC_LIMIT_UNFILLED)

        orders[0].src = ORDER_SRC_LIMIT_FILLED
        orders[0].price = 50000.0
        orders[0].processed_ms = TimeUtil.now_in_millis()

        self.assertEqual(orders[0].src, ORDER_SRC_LIMIT_FILLED)
        self.assertIsNotNone(orders[0].price)
        self.assertIsNotNone(orders[0].processed_ms)

    def test_multiple_miners_limit_orders(self):
        """Test limit orders for multiple miners"""
        miner2_hotkey = "test_miner_2"

        self.mock_metagraph.hotkeys.append(miner2_hotkey)

        order1 = self.create_test_limit_order(order_uuid="miner1_order")
        order2 = self.create_test_limit_order(order_uuid="miner2_order")

        self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, order1)
        self.limit_order_manager.save_limit_order(miner2_hotkey, order2)

        miner1_orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        miner2_orders = self.limit_order_manager.limit_orders.get(miner2_hotkey, [])

        self.assertEqual(len(miner1_orders), 1)
        self.assertEqual(len(miner2_orders), 1)
        self.assertEqual(miner1_orders[0].order_uuid, "miner1_order")
        self.assertEqual(miner2_orders[0].order_uuid, "miner2_order")

    def test_limit_order_with_different_leverage_values(self):
        """Test limit orders with various leverage values"""
        leverages = [0.1, 0.5, 1.0, 2.0, 5.0]

        for i, leverage in enumerate(leverages):
            order = self.create_test_limit_order(
                leverage=leverage,
                order_uuid=f"leverage_order_{i}"
            )
            self.limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        orders = self.limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), len(leverages))

        stored_leverages = [o.leverage for o in orders]
        for leverage in leverages:
            self.assertIn(leverage, stored_leverages)

    def test_limit_order_initialization_from_disk(self):
        """Test that limit orders are properly initialized when LimitOrderManager starts"""
        limit_order_manager = LimitOrderManager(
            position_manager=self.position_manager,
            live_price_fetcher=self.live_price_fetcher
        )

        self.assertEqual(len(limit_order_manager.limit_orders), 0)

        limit_order = self.create_test_limit_order()
        limit_order_manager.save_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        orders = limit_order_manager.limit_orders.get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 1)


if __name__ == '__main__':
    unittest.main()
