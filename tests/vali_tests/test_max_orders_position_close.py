# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import os
import uuid
from unittest.mock import MagicMock, patch
from tests.vali_tests.mock_utils import (
    EnhancedMockMetagraph,
    EnhancedMockChallengePeriodManager,
    EnhancedMockPositionManager,
    EnhancedMockPerfLedgerManager,
    MockSubtensorWeightSetterHelper,
    MockScoring
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_8_HOURS, MS_IN_24_HOURS

# Define missing constant
MS_IN_1_MINUTE = 60000
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order, OrderSource


class TestMaxOrdersPositionClose(TestBase):
    """
    Comprehensive unit test for MAX_ORDERS_PER_POSITION_CLOSE functionality.

    Tests the complete production code path for automatic position closing when
    hitting the 200 order limit, including disk persistence verification.
    """

    def setUp(self):
        super().setUp()

        # Test miners
        self.TEST_MINER = "test_miner_max_orders"
        self.NORMAL_MINER = "normal_miner"

        self.all_miners = [self.TEST_MINER, self.NORMAL_MINER]

        # Initialize components with enhanced mocks
        self.mock_metagraph = EnhancedMockMetagraph(self.all_miners)

        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)

        self.position_locks = PositionLocks()

        # Create IPC manager for multiprocessing simulation
        self.mock_ipc_manager = MagicMock()
        self.mock_ipc_manager.list.return_value = []
        self.mock_ipc_manager.dict.return_value = {}

        # Create all managers
        self.perf_ledger_manager = EnhancedMockPerfLedgerManager(
            self.mock_metagraph,
            ipc_manager=self.mock_ipc_manager,
            running_unit_tests=True,
            perf_ledger_hks_to_invalidate={}
        )

        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.live_price_fetcher,
            None,
            running_unit_tests=True,
            ipc_manager=self.mock_ipc_manager,
            contract_manager=self.contract_manager
        )

        self.position_manager = EnhancedMockPositionManager(
            self.mock_metagraph,
            perf_ledger_manager=self.perf_ledger_manager,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )

        self.challengeperiod_manager = EnhancedMockChallengePeriodManager(
            self.mock_metagraph,
            position_manager=self.position_manager,
            perf_ledger_manager=self.perf_ledger_manager,
            contract_manager=self.contract_manager,
            running_unit_tests=True
        )

        # Set circular references
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        self.perf_ledger_manager.position_manager = self.position_manager
        self.perf_ledger_manager.elimination_manager = self.elimination_manager

        # Create weight setter
        self.weight_setter = SubtensorWeightSetter(
            self.mock_metagraph,
            self.position_manager,
            contract_manager=self.contract_manager,
            running_unit_tests=True
        )
        self.weight_setter.position_manager.challengeperiod_manager = self.challengeperiod_manager

        # Clear all data
        self.clear_all_data()

        # Set up initial state
        self._setup_environment()

    def tearDown(self):
        super().tearDown()
        self.clear_all_data()

    def clear_all_data(self):
        """Clear all test data"""
        self.position_manager.clear_all_miner_positions()
        self.perf_ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

    def _setup_environment(self):
        """Set up test environment"""
        # Set all miners to main competition
        for miner in self.all_miners:
            self.challengeperiod_manager.set_miner_bucket(miner, MinerBucket.MAINCOMP, 0)

    def _create_test_position_with_orders(self, miner_hotkey: str, trade_pair: TradePair, n_orders: int):
        """
        Create a position with specified number of orders for testing.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair: The trade pair for the position
            n_orders: Number of orders to create in the position

        Returns:
            Position object with the specified number of orders
        """
        base_time = TimeUtil.now_in_millis() - MS_IN_24_HOURS
        position_uuid = f"{miner_hotkey}_{trade_pair.trade_pair_id}_{uuid.uuid4()}"

        # Create position with first order
        position = Position(
            miner_hotkey=miner_hotkey,
            position_uuid=position_uuid,
            open_ms=base_time,
            trade_pair=trade_pair,
            is_closed_position=False,
            orders=[]
        )

        # Add the specified number of orders
        for i in range(n_orders):
            order_time = base_time + (i * MS_IN_1_MINUTE)
            order = Order(
                price=60000 + (i * 10),  # Slightly different prices
                processed_ms=order_time,
                order_uuid=f"order_{position_uuid}_{i}",
                trade_pair=trade_pair,
                order_type=OrderType.LONG if i % 2 == 0 else OrderType.SHORT,
                leverage=0.1 + (i * 0.01),  # Small leverage increments
                src=OrderSource.ORGANIC  # All orders start as organic
            )
            position.orders.append(order)

        return position

    def _simulate_validator_order_processing(self, miner_hotkey: str, position: Position, new_order: Order):
        """
        Simulate the validator's order processing logic using the actual validator code
        to test the MAX_ORDERS_PER_POSITION_CLOSE behavior.
        """
        # Create a mock validator to test the actual logic
        from neurons.validator import Validator
        from unittest.mock import Mock, MagicMock

        # Get all open positions for the miner (similar to validator logic)
        open_positions = self.position_manager.get_positions_for_one_hotkey(
            miner_hotkey, only_open_positions=True
        )

        # Create trade_pair_to_open_position dict (like validator does)
        trade_pair_to_open_position = {}
        for pos in open_positions:
            trade_pair_to_open_position[pos.trade_pair] = pos

        # Create a minimal validator instance to test the method
        mock_validator = MagicMock()
        mock_validator.position_manager = self.position_manager
        mock_validator.live_price_fetcher = self.live_price_fetcher

        # Create the actual _enforce_num_open_order_limit method bound to our mock
        from neurons.validator import Validator
        enforce_method = Validator._enforce_num_open_order_limit

        # Count orders before
        total_orders_before = sum([len(pos.orders) for pos in trade_pair_to_open_position.values()])

        try:
            # Call the actual validator method
            enforce_method(mock_validator, trade_pair_to_open_position, new_order, miner_hotkey)

            # Check if position was closed (removed from dict)
            if position.trade_pair not in trade_pair_to_open_position:
                # Position was closed and removed - this indicates successful MAX_ORDERS_PER_POSITION_CLOSE behavior
                return True  # Position was split/closed

            # If position still exists, check if it was closed
            target_position = trade_pair_to_open_position[position.trade_pair]
            if target_position.is_closed_position:
                # Position was closed but not removed from dict
                return True  # Position was split
            else:
                # Position still open, add order normally
                target_position.orders.append(new_order)
                self.position_manager.save_miner_position(target_position)
                return False  # No split occurred

        except Exception as e:
            # Exception was raised - old behavior (raise exception instead of closing)
            return False

    def test_max_orders_position_close_basic_flow(self):
        """Test basic flow of position closing when hitting 200 order limit"""

        # Create position with exactly 200 orders (at the limit)
        position = self._create_test_position_with_orders(
            self.TEST_MINER,
            TradePair.BTCUSD,
            200
        )
        self.position_manager.save_miner_position(position)

        # Verify initial state
        positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        self.assertEqual(len(positions), 1)
        self.assertEqual(len(positions[0].orders), 200)
        self.assertFalse(positions[0].is_closed_position)

        # Create the 201st order (should trigger position closing)
        order_201 = Order(
            price=62000,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"order_201_{uuid.uuid4()}",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5,
            src=OrderSource.ORGANIC
        )

        # Simulate validator processing
        position_was_split = self._simulate_validator_order_processing(
            self.TEST_MINER, position, order_201
        )

        # Verify position was split
        self.assertTrue(position_was_split, "Position should have been automatically closed when hitting 200 order limit")

        print("✅ SUCCESS: MAX_ORDERS_PER_POSITION_CLOSE validator logic triggered correctly")
        print(f"✅ Position was automatically closed when trying to add 201st order")
        print(f"✅ This prevents miners from exceeding the {ValiConfig.MAX_OPEN_ORDERS_PER_HOTKEY} order limit")

    def test_max_orders_with_flat_order_200th(self):
        """Test that FLAT order as 200th order doesn't trigger position splitting"""

        # Create position with 199 orders
        position = self._create_test_position_with_orders(
            self.TEST_MINER,
            TradePair.ETHUSD,
            199
        )
        self.position_manager.save_miner_position(position)

        # Create FLAT order as 200th order
        flat_order = Order(
            price=3000,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"flat_order_{uuid.uuid4()}",
            trade_pair=TradePair.ETHUSD,
            order_type=OrderType.FLAT,
            leverage=0.0,
            src=OrderSource.ORGANIC
        )

        # Simulate validator processing
        position_was_split = self._simulate_validator_order_processing(
            self.TEST_MINER, position, flat_order
        )

        # Verify position was NOT split (FLAT order is allowed)
        self.assertFalse(position_was_split)

        # Verify final state - position should be closed with 200 orders
        final_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        self.assertEqual(len(final_positions), 1)

        final_pos = final_positions[0]
        self.assertTrue(final_pos.is_closed_position)
        self.assertEqual(len(final_pos.orders), 200)
        self.assertEqual(final_pos.orders[-1].order_type, OrderType.FLAT)
        self.assertEqual(final_pos.orders[-1].src, OrderSource.ORGANIC)

    def test_multiple_positions_order_limit_enforcement(self):
        """Test order limit enforcement across multiple open positions"""

        # Create two positions with 100 orders each (total 200)
        position1 = self._create_test_position_with_orders(
            self.TEST_MINER, TradePair.BTCUSD, 100
        )
        position2 = self._create_test_position_with_orders(
            self.TEST_MINER, TradePair.ETHUSD, 100
        )

        self.position_manager.save_miner_position(position1)
        self.position_manager.save_miner_position(position2)

        # Verify initial state
        positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        self.assertEqual(len(positions), 2)
        total_orders = sum(len(p.orders) for p in positions)
        self.assertEqual(total_orders, 200)

        # Try to add 201st order to position1
        order_201 = Order(
            price=61000,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"order_201_{uuid.uuid4()}",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.SHORT,
            leverage=0.3,
            src=OrderSource.ORGANIC
        )

        # Simulate validator processing (should trigger split on position1)
        position_was_split = self._simulate_validator_order_processing(
            self.TEST_MINER, position1, order_201
        )

        self.assertTrue(position_was_split)

        # Verify final state
        final_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)

        # Should have 3 positions: 1 original ETHUSD (open), 1 closed BTCUSD, 1 new BTCUSD (open)
        self.assertEqual(len(final_positions), 3)

        btc_positions = [p for p in final_positions if p.trade_pair == TradePair.BTCUSD]
        eth_positions = [p for p in final_positions if p.trade_pair == TradePair.ETHUSD]

        self.assertEqual(len(btc_positions), 2)  # Original closed + new open
        self.assertEqual(len(eth_positions), 1)  # Unchanged

        # Verify BTC positions structure
        closed_btc = [p for p in btc_positions if p.is_closed_position][0]
        open_btc = [p for p in btc_positions if not p.is_closed_position][0]

        self.assertEqual(len(closed_btc.orders), 101)  # 100 + FLAT
        self.assertEqual(closed_btc.orders[-1].src, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)

        self.assertEqual(len(open_btc.orders), 1)  # Just the 201st order
        self.assertEqual(open_btc.orders[0].order_type, OrderType.SHORT)

    def test_disk_persistence_verification(self):
        """Test that positions with MAX_ORDERS_PER_POSITION_CLOSE are correctly persisted to disk"""

        # Create position and trigger max orders scenario
        position = self._create_test_position_with_orders(
            self.TEST_MINER, TradePair.GBPUSD, 199
        )
        self.position_manager.save_miner_position(position)

        # Add 200th order to trigger split
        order_200 = Order(
            price=1.26,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"trigger_order_{uuid.uuid4()}",
            trade_pair=TradePair.GBPUSD,
            order_type=OrderType.LONG,
            leverage=0.4,
            src=OrderSource.ORGANIC
        )

        self._simulate_validator_order_processing(self.TEST_MINER, position, order_200)

        # Force save to disk and reload to verify persistence
        positions_before_reload = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)

        # Clear memory and reload from disk
        self.position_manager.clear_all_miner_positions()
        self.assertEqual(len(self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)), 0)

        # Reload from disk
        self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        positions_after_reload = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)

        # Verify positions were correctly persisted and reloaded
        self.assertEqual(len(positions_after_reload), len(positions_before_reload))

        # Find the closed position with MAX_ORDERS_PER_POSITION_CLOSE
        closed_positions = [p for p in positions_after_reload if p.is_closed_position]
        self.assertEqual(len(closed_positions), 1)

        closed_pos = closed_positions[0]
        self.assertEqual(len(closed_pos.orders), 200)

        # Verify the FLAT order with correct source is preserved
        flat_order = closed_pos.orders[-1]
        self.assertEqual(flat_order.order_type, OrderType.FLAT)
        self.assertEqual(flat_order.src, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)

        # Verify all other orders maintain their original source
        organic_orders = [o for o in closed_pos.orders[:-1]]  # All except the FLAT
        for order in organic_orders:
            self.assertEqual(order.src, OrderSource.ORGANIC)

    def test_position_splitting_with_different_trade_pairs(self):
        """Test position splitting behavior across different trade pairs"""

        trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.GBPUSD]

        for i, trade_pair in enumerate(trade_pairs):
            # Create position approaching limit
            position = self._create_test_position_with_orders(
                self.TEST_MINER, trade_pair, 66 + i  # Different order counts
            )
            self.position_manager.save_miner_position(position)

        # Now we have 66 + 67 + 68 = 201 orders total
        # Next order should trigger split on the position it's added to

        positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        total_orders = sum(len(p.orders) for p in positions)
        self.assertEqual(total_orders, 201)

        # Add order to BTCUSD position (should trigger split)
        btc_position = [p for p in positions if p.trade_pair == TradePair.BTCUSD][0]

        trigger_order = Order(
            price=63000,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"trigger_{uuid.uuid4()}",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.2,
            src=OrderSource.ORGANIC
        )

        position_was_split = self._simulate_validator_order_processing(
            self.TEST_MINER, btc_position, trigger_order
        )

        self.assertTrue(position_was_split)

        # Verify final state
        final_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)

        # Should have 4 positions total: ETH (open), GBP (open), BTC (closed), BTC (new open)
        btc_positions = [p for p in final_positions if p.trade_pair == TradePair.BTCUSD]
        eth_positions = [p for p in final_positions if p.trade_pair == TradePair.ETHUSD]
        gbp_positions = [p for p in final_positions if p.trade_pair == TradePair.GBPUSD]

        self.assertEqual(len(btc_positions), 2)  # Closed + new
        self.assertEqual(len(eth_positions), 1)  # Unchanged
        self.assertEqual(len(gbp_positions), 1)  # Unchanged

        # Verify only BTC position was affected
        self.assertTrue(any(p.is_closed_position for p in btc_positions))
        self.assertTrue(all(not p.is_closed_position for p in eth_positions))
        self.assertTrue(all(not p.is_closed_position for p in gbp_positions))

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    @patch('vali_objects.utils.subtensor_weight_setter.Scoring', MockScoring)
    def test_integration_with_performance_calculations(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test that positions closed due to max orders are properly handled in performance calculations"""

        # Mock the API calls
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)

        # Create and split position
        position = self._create_test_position_with_orders(
            self.TEST_MINER, TradePair.BTCUSD, 199
        )
        self.position_manager.save_miner_position(position)

        # Trigger split
        trigger_order = Order(
            price=61000,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"trigger_{uuid.uuid4()}",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.5,
            src=OrderSource.ORGANIC
        )

        self._simulate_validator_order_processing(self.TEST_MINER, position, trigger_order)

        # Verify positions exist
        final_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        closed_positions = [p for p in final_positions if p.is_closed_position]
        self.assertEqual(len(closed_positions), 1)

        # Test that performance calculations can handle the position structure
        closed_pos = closed_positions[0]

        # Rebuild position with updated orders (should work without errors)
        try:
            closed_pos.rebuild_position_with_updated_orders(self.live_price_fetcher)
            performance_calculation_success = True
        except Exception as e:
            performance_calculation_success = False
            print(f"Performance calculation failed: {e}")

        self.assertTrue(performance_calculation_success)

        # Verify return calculations work
        self.assertIsNotNone(closed_pos.return_at_close)
        self.assertIsInstance(closed_pos.return_at_close, (int, float))

        # Test weight computation includes the miner appropriately
        current_time = TimeUtil.now_in_millis()
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(current_time)

        # Miner should be included in weight calculations
        miners_with_weights = [result[0] for result in checkpoint_results]
        self.assertIn(self.TEST_MINER, miners_with_weights)

    def test_edge_case_exact_200_orders(self):
        """Test edge case where position has exactly 200 orders"""

        # Create position with exactly 200 orders
        position = self._create_test_position_with_orders(
            self.TEST_MINER, TradePair.BTCUSD, 200
        )
        self.position_manager.save_miner_position(position)

        # Try to add 201st order
        order_201 = Order(
            price=62000,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"order_201_{uuid.uuid4()}",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.SHORT,
            leverage=0.3,
            src=OrderSource.ORGANIC
        )

        # Should trigger split since we're at the limit
        position_was_split = self._simulate_validator_order_processing(
            self.TEST_MINER, position, order_201
        )

        self.assertTrue(position_was_split)

        # Verify the closed position has 201 orders (200 original + 1 FLAT)
        final_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        closed_positions = [p for p in final_positions if p.is_closed_position]

        self.assertEqual(len(closed_positions), 1)
        closed_pos = closed_positions[0]
        self.assertEqual(len(closed_pos.orders), 201)
        self.assertEqual(closed_pos.orders[-1].src, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)

    def test_order_source_enum_values(self):
        """Test that OrderSource enum has correct values for MAX_ORDERS_PER_POSITION_CLOSE"""

        # Verify the enum value exists and has correct integer value
        self.assertEqual(OrderSource.MAX_ORDERS_PER_POSITION_CLOSE, 4)
        self.assertEqual(OrderSource.MAX_ORDERS_PER_POSITION_CLOSE.value, 4)

        # Verify backwards compatibility constants still work
        from vali_objects.vali_dataclasses.order import ORDER_SRC_MAX_ORDERS_PER_POSITION_CLOSE
        self.assertEqual(ORDER_SRC_MAX_ORDERS_PER_POSITION_CLOSE, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)

    def test_comprehensive_production_simulation(self):
        """
        Comprehensive test that simulates the complete production code path
        from signal reception to weight setting for MAX_ORDERS_PER_POSITION_CLOSE scenario
        """

        # 1. Set up realistic initial state
        base_time = TimeUtil.now_in_millis() - MS_IN_24_HOURS

        # Create position approaching limit with realistic trading pattern
        position = Position(
            miner_hotkey=self.TEST_MINER,
            position_uuid=f"production_test_{uuid.uuid4()}",
            open_ms=base_time,
            trade_pair=TradePair.BTCUSD,
            is_closed_position=False,
            orders=[]
        )

        # Add 199 orders with realistic price progression
        base_price = 60000
        for i in range(199):
            order_time = base_time + (i * MS_IN_1_MINUTE * 5)  # Every 5 minutes
            price_variation = (i % 20 - 10) * 10  # Small price variations

            order = Order(
                price=base_price + price_variation,
                processed_ms=order_time,
                order_uuid=f"prod_order_{i}_{uuid.uuid4()}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG if i % 3 == 0 else OrderType.SHORT,
                leverage=0.1 + (i % 10) * 0.05,  # Varying leverage
                src=OrderSource.ORGANIC
            )
            position.orders.append(order)

        self.position_manager.save_miner_position(position)

        # 2. Verify pre-split state
        pre_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        self.assertEqual(len(pre_positions), 1)
        self.assertEqual(len(pre_positions[0].orders), 199)

        # 3. Simulate receiving 200th signal that triggers split
        trigger_order = Order(
            price=61500,
            processed_ms=TimeUtil.now_in_millis(),
            order_uuid=f"trigger_production_{uuid.uuid4()}",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.8,
            src=OrderSource.ORGANIC
        )

        # 4. Process through validator logic
        split_occurred = self._simulate_validator_order_processing(
            self.TEST_MINER, position, trigger_order
        )

        self.assertTrue(split_occurred)

        # 5. Verify post-split state matches production expectations
        post_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)
        self.assertEqual(len(post_positions), 2)

        closed_positions = [p for p in post_positions if p.is_closed_position]
        open_positions = [p for p in post_positions if not p.is_closed_position]

        self.assertEqual(len(closed_positions), 1)
        self.assertEqual(len(open_positions), 1)

        # 6. Verify closed position structure
        closed_pos = closed_positions[0]
        self.assertEqual(len(closed_pos.orders), 200)  # 199 + 1 FLAT

        # Verify all original orders preserved
        original_orders = closed_pos.orders[:-1]
        self.assertEqual(len(original_orders), 199)
        self.assertTrue(all(o.src == OrderSource.ORGANIC for o in original_orders))

        # Verify FLAT order
        flat_order = closed_pos.orders[-1]
        self.assertEqual(flat_order.order_type, OrderType.FLAT)
        self.assertEqual(flat_order.src, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)
        self.assertEqual(flat_order.price, trigger_order.price)  # Should use trigger price

        # 7. Verify new position structure
        new_pos = open_positions[0]
        self.assertEqual(len(new_pos.orders), 1)
        self.assertEqual(new_pos.orders[0].order_uuid, trigger_order.order_uuid)
        self.assertEqual(new_pos.orders[0].src, OrderSource.ORGANIC)

        # 8. Test disk persistence across restart simulation
        # Save current state
        saved_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)

        # Simulate restart - clear memory and reload
        self.position_manager.clear_all_miner_positions()
        reloaded_positions = self.position_manager.get_positions_for_one_hotkey(self.TEST_MINER)

        # Verify state preserved across restart
        self.assertEqual(len(reloaded_positions), len(saved_positions))

        # Find reloaded closed position and verify structure
        reloaded_closed = [p for p in reloaded_positions if p.is_closed_position][0]
        self.assertEqual(len(reloaded_closed.orders), 200)
        self.assertEqual(reloaded_closed.orders[-1].src, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)

        # 9. Verify the positions can be properly processed by performance systems
        try:
            # Test return calculations
            for pos in reloaded_positions:
                if pos.is_closed_position:
                    pos.rebuild_position_with_updated_orders(self.live_price_fetcher)
                    self.assertIsNotNone(pos.return_at_close)

            # Test integration with perf ledger
            self.perf_ledger_manager.get_performance_ledger_for_miner(self.TEST_MINER)

            integration_success = True
        except Exception as e:
            integration_success = False
            print(f"Integration test failed: {e}")

        self.assertTrue(integration_success)

        print("✅ Comprehensive production simulation completed successfully")
        print(f"   - Closed position: {len(reloaded_closed.orders)} orders")
        print(f"   - FLAT order source: {reloaded_closed.orders[-1].src.name}")
        print(f"   - New position: {len([p for p in reloaded_positions if not p.is_closed_position][0].orders)} orders")