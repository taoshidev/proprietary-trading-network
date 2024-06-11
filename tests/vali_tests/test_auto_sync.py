import json
from copy import deepcopy

from vali_objects.utils.auto_sync import PositionSyncer, AUTO_SYNC_ORDER_LAG_MS
from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.order import Order


class TestPositions(TestBase):

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_OPEN_MS = 1718071209000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_order = Order(price=1, processed_ms=1718071209000, order_uuid=self.DEFAULT_ORDER_UUID, trade_pair=self.DEFAULT_TRADE_PAIR,
                                     order_type=OrderType.LONG, leverage=1)
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order]
        )
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.position_manager.init_cache_files()
        self.position_manager.clear_all_miner_positions_from_disk()

        self.position_syncer = PositionSyncer()

    def positions_to_disk_data(self, positions: list[Position]):
        return {self.DEFAULT_MINER_HOTKEY: positions}

    def positions_to_candidate_data(self, positions: list[Position]):
        mt = 0
        for p in positions:
            for o in p.orders:
                mt = max(mt, o.processed_ms)

        candidate_data = {'positions': {self.DEFAULT_MINER_HOTKEY: {'positions': positions}},
                  'eliminations': [], 'created_timestamp_ms': mt + AUTO_SYNC_ORDER_LAG_MS}
        for hk in candidate_data['positions']:
            positions_orig = candidate_data['positions'][hk]['positions']
            candidate_data['positions'][hk]['positions'] = [json.loads(str(p), cls=GeneralizedJSONDecoder) for p in positions_orig]
        return candidate_data


    def test_validate_basic_position_sync(self):
        candidate_data = self.positions_to_candidate_data([self.default_position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        assert stats['n_miners_synced'] == 1
        assert stats['n_miners_positions_deleted'] == 0
        assert stats['n_miners_positions_kept'] == 0
        assert stats['n_miners_positions_matched'] == 1
        assert stats['n_miners_positions_inserted'] == 0
        assert stats['n_miners_orders_deleted'] == 0
        assert stats['n_miners_orders_inserted'] == 0
        assert stats['n_miners_orders_matched'] == 1
        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

    def test_validate_basic_order_sync_order_match_with_heuristic_insert_one(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = self.DEFAULT_ORDER_UUID + "foobar"
        order1.processed_ms = self.default_order.processed_ms + 1000
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        assert stats['n_miners_synced'] == 1
        assert stats['n_miners_positions_deleted'] == 0
        assert stats['n_miners_positions_kept'] == 0
        assert stats['n_miners_positions_matched'] == 1
        assert stats['n_miners_positions_inserted'] == 0
        assert stats['n_miners_orders_deleted'] == 0
        assert stats['n_miners_orders_inserted'] == 1
        assert stats['n_miners_orders_matched'] == 1

        assert stats['orders_inserted'] == 1
        assert stats['orders_matched'] == 1
        assert stats['orders_deleted'] == 0
        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate['test_miner'] == order2.processed_ms

    def test_validate_basic_order_sync_order_match_with_uuid_insert_one(self):
        order1 = deepcopy(self.default_order)
        order1.processed_ms = self.default_order.processed_ms + 10000 # Purposely different to ensure match on uuid happens regardless
        order1.order_type = OrderType.SHORT# Purposely different to ensure match on uuid happens regardless
        order1.leverage = 2  # Purposely different to ensure match on uuid happens regardless
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        assert stats['n_miners_synced'] == 1
        assert stats['n_miners_positions_deleted'] == 0
        assert stats['n_miners_positions_kept'] == 0
        assert stats['n_miners_positions_matched'] == 1
        assert stats['n_miners_positions_inserted'] == 0
        assert stats['n_miners_orders_deleted'] == 0
        assert stats['n_miners_orders_inserted'] == 1
        assert stats['n_miners_orders_matched'] == 1

        assert stats['orders_inserted'] == 1
        assert stats['orders_matched'] == 1
        assert stats['orders_deleted'] == 0

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[order2.order_uuid] == order2.processed_ms

    def test_validate_basic_order_sync_no_matches(self):
        for i in range(3):
            order1 = deepcopy(self.default_order)
            order1.order_uuid = self.DEFAULT_ORDER_UUID + "foobar"
            if i == 0:
                order1.processed_ms = self.default_order.processed_ms + 10000  # Purposely different to ensure no match
            if i == 1:
                order1.order_type = OrderType.SHORT  # Purposely different to ensure no match
            if i == 2:
                order1.leverage = 2  # Purposely different to ensure no match
            order2 = deepcopy(self.default_order)
            order2.order_uuid = "test_order2"
            order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
            orders = [order1, order2]
            position = deepcopy(self.default_position)
            position.orders = orders
            candidate_data = self.positions_to_candidate_data([position])
            disk_positions = self.positions_to_disk_data([self.default_position])
            self.position_syncer.sync_positions(candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats
            assert stats['n_miners_synced'] == 1, i
            assert stats['n_miners_positions_deleted'] == 0, i
            assert stats['n_miners_positions_kept'] == 0, i
            assert stats['n_miners_positions_matched'] == 1, i
            assert stats['n_miners_positions_inserted'] == 0, i
            assert stats['n_miners_orders_deleted'] == 0, i
            assert stats['n_miners_orders_inserted'] == 0, i
            assert stats['n_miners_orders_matched'] == 1, i

            assert stats['orders_inserted'] == 0, i
            assert stats['orders_matched'] == 1, i
            assert stats['orders_deleted'] == 0, i


        # Test fragmentation especially hard. May need to do a hard snap to candidates.

