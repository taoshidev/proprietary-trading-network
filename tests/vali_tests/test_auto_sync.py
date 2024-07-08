import json
from copy import deepcopy

from vali_objects.utils.auto_sync import PositionSyncer
from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.validator_sync_base import AUTO_SYNC_ORDER_LAG_MS
from vali_objects.vali_dataclasses.order import Order


class TestPositions(TestBase):

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_OPEN_MS = 1718071209000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_order = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS, order_uuid=self.DEFAULT_ORDER_UUID, trade_pair=self.DEFAULT_TRADE_PAIR,
                                     order_type=OrderType.LONG, leverage=1)
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.LONG
        )
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.position_manager.init_cache_files()
        self.position_manager.clear_all_miner_positions_from_disk()

        self.default_open_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.LONG
        )

        self.default_closed_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.FLAT
        )
        self.default_closed_position.close_out_position(self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 6)

        self.position_syncer = PositionSyncer()

    def positions_to_disk_data(self, positions: list[Position]):
        return {self.DEFAULT_MINER_HOTKEY: positions}

    def positions_to_candidate_data(self, positions: list[Position]):
        mt = 0
        for p in positions:
            for o in p.orders:
                mt = max(mt, o.processed_ms)

        candidate_data = {'positions': {self.DEFAULT_MINER_HOTKEY: {}},
                  'eliminations': [], 'created_timestamp_ms': mt + AUTO_SYNC_ORDER_LAG_MS}
        for hk in candidate_data['positions']:
            candidate_data['positions'][hk]['positions'] = [json.loads(str(p), cls=GeneralizedJSONDecoder) for p in positions]
        return candidate_data


    def test_validate_basic_sync(self):
        candidate_data = self.positions_to_candidate_data([self.default_position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
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

        # When there are no existing positions, we should insert the new one.
        candidate_data = self.positions_to_candidate_data([self.default_position])
        disk_positions = self.positions_to_disk_data([])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 0, stats
        assert stats['n_miners_positions_kept'] == 0, stats
        assert stats['n_miners_positions_matched'] == 0, stats
        assert stats['n_miners_positions_inserted'] == 1, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 0, stats
        assert stats['n_miners_orders_matched'] == 0, stats
        assert stats['n_miners_orders_kept'] == 0, stats

        assert stats['positions_inserted'] == 1, stats
        assert stats['positions_matched'] == 0, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 0, stats
        assert stats['orders_matched'] == 0, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats
        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.DEFAULT_OPEN_MS

    def test_position_deletion(self):
        dp1 = deepcopy(self.default_closed_position)
        dp1.position_uuid = 'to_delete'
        dp1.open_ms -= 1000 * 60 * 10

        # Opened within hardsnap but closed after hardsnap. Sry doesn't matter - should have existed.
        dp2 = deepcopy(self.default_closed_position)
        dp2.position_uuid = 'to_delete'
        dp2.open_ms -= 1000 * 60 * 10
        dp2.close_ms += 1000 * 60 * 10

        dp3 = deepcopy(self.default_open_position)
        dp3.position_uuid = 'to_delete'
        dp3.open_ms -= 1000 * 60 * 10

        # One delete, one insert (from candidate)
        for i, dp in enumerate([dp1, dp2, dp3]):
            cd = self.default_open_position if i == 2 else self.default_closed_position
            candidate_data = self.positions_to_candidate_data([cd])
            disk_positions = self.positions_to_disk_data([dp])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats

            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            assert stats['n_miners_synced'] == 1, (i, stats_str)

            assert stats['n_miners_positions_deleted'] == 1, (i, stats_str)
            assert stats['n_miners_positions_kept'] == 0, (i, stats_str)
            assert stats['n_miners_positions_matched'] == 0, (i, stats_str)
            assert stats['n_miners_positions_inserted'] == 1, (i, stats_str)

            assert stats['n_miners_orders_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_inserted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_matched'] == 0, (i, stats_str)
            assert stats['n_miners_orders_kept'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 1, stats
            assert stats['positions_matched'] == 0, stats
            assert stats['positions_deleted'] == 1, stats
            assert stats['positions_kept'] == 0, stats

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
            assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == dp.open_ms


        # Same thing just no more inserting. We keep the positions since there is no candidate.
        candidate_data = self.positions_to_candidate_data([])
        for i, dp in enumerate([dp1, dp2, dp3]):
            disk_positions = self.positions_to_disk_data([dp])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats

            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            assert stats['n_miners_synced'] == 1, (i, stats_str)

            assert stats['n_miners_positions_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_positions_kept'] == 1, (i, stats_str)
            assert stats['n_miners_positions_matched'] == 0, (i, stats_str)
            assert stats['n_miners_positions_inserted'] == 0, (i, stats_str)

            assert stats['n_miners_orders_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_inserted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_matched'] == 0, (i, stats_str)
            assert stats['n_miners_orders_kept'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 0, stats
            assert stats['positions_matched'] == 0, stats
            assert stats['positions_deleted'] == 0, stats
            assert stats['positions_kept'] == 1, stats

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

    def test_split_position_with_orders_after_flat(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = self.default_order.processed_ms + 1000
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
        order2.order_type = OrderType.FLAT
        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order3"
        order3.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10 * 2
        orders = [order1, order2, order3]
        position = deepcopy(self.default_position)
        position.orders = orders
        print(position.is_open_position)
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                            disk_positions=disk_positions)
        stats = self.position_syncer.global_stats

        assert stats['n_positions_spawned_from_post_flat_orders'] == 1, stats
        assert stats['n_positions_closed_duplicate_opens_for_trade_pair'] == 0, stats

    def test_position_keep_both_insert(self):
        dp1 = deepcopy(self.default_closed_position)
        dp1.position_uuid = 'to_keep'
        # After hardsnap, we must keep it
        dp1.open_ms += 1000 * 60 * 10

        dp2 = deepcopy(self.default_closed_position)
        dp2.position_uuid = 'to_keep'
        # After hardsnap, we must keep it
        dp2.open_ms += 1000 * 60 * 10
        dp2.close_ms += 1000 * 60 * 10

        # Close the older open position, and insert the newer open position
        dp3 = deepcopy(self.default_open_position)
        dp3.position_uuid = 'double_open'
        dp3.open_ms += 1000 * 60 * 10

        for i, dp in enumerate([dp1, dp2, dp3]):
            cd = self.default_open_position if i == 2 else self.default_closed_position
            candidate_data = self.positions_to_candidate_data([cd])
            disk_positions = self.positions_to_disk_data([dp])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats

            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            assert stats['n_miners_synced'] == 1, (i, stats_str)
            # if i == 2:
            #     assert stats['blocked_keep_open_position_acked'] == 1
            assert stats['n_miners_positions_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_positions_kept'] == 1, (i, stats_str)
            assert stats['n_miners_positions_matched'] == 0, (i, stats_str)
            assert stats['n_miners_positions_inserted'] == 1, (i, stats_str)

            assert stats['n_miners_orders_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_inserted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_matched'] == 0, (i, stats_str)
            assert stats['n_miners_orders_kept'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 1, (i, stats_str)
            assert stats['positions_matched'] == 0, (i, stats_str)
            assert stats['positions_deleted'] == 0, (i, stats_str)
            assert stats['positions_kept'] == 1, (i, stats_str)

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
            assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == cd.open_ms

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
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats

        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 0, stats
        assert stats['n_miners_positions_kept'] == 0, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 0, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 1, stats
        assert stats['n_miners_orders_matched'] == 1, stats
        assert stats['n_miners_orders_kept'] == 0, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == order2.processed_ms

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
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats

        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 0, stats
        assert stats['n_miners_positions_kept'] == 0, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 0, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 1, stats
        assert stats['n_miners_orders_matched'] == 1, stats
        assert stats['n_miners_orders_kept'] == 0, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == order2.processed_ms

    def test_validate_basic_order_sync_no_matches_one_deletion(self):
        disk_positions = self.positions_to_disk_data([self.default_position])
        for i in range(3):
            order1 = deepcopy(self.default_order)
            order1.order_uuid = self.DEFAULT_ORDER_UUID + "foobar"
            if i == 0:
                order1.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10  # Purposely different to ensure no match
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
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats
            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            assert stats['n_miners_synced'] == 1, stats

            assert stats['n_miners_positions_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_positions_kept'] == 0, (i, stats_str)
            assert stats['n_miners_positions_matched'] == 1, (i, stats_str)
            assert stats['n_miners_positions_inserted'] == 0, (i, stats_str)

            assert stats['n_miners_orders_deleted'] == 1, (i, stats_str)
            assert stats['n_miners_orders_inserted'] == 1, (i, stats_str)
            assert stats['n_miners_orders_matched'] == 0, (i, stats_str)
            assert stats['n_miners_orders_kept'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 0, (i, stats_str)
            assert stats['positions_matched'] == 1, (i, stats_str)
            assert stats['positions_deleted'] == 0, (i, stats_str)
            assert stats['positions_kept'] == 0, (i, stats_str)

            assert stats['orders_inserted'] == 2, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 1, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
            assert (self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] ==
                    min(order1.processed_ms, order2.processed_ms, self.default_order.processed_ms))

    def test_validate_order_sync_all_matches_heuristic(self):
        disk_positions = self.positions_to_disk_data([self.default_position])
        for i in range(2):
            order1 = deepcopy(self.default_order)
            order1.order_uuid = self.DEFAULT_ORDER_UUID + "foobar"
            if i == 0:
                order1.processed_ms = self.default_order.processed_ms + 1000 * 60 * 2  # Slightly different. Should match
            if i == 1:
                order1.processed_ms = self.default_order.processed_ms - 1000 * 60 * 2  # Slightly different. Should match

            orders = [order1]
            position = deepcopy(self.default_position)
            position.orders = orders
            candidate_data = self.positions_to_candidate_data([position])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats
            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "


            assert stats['n_miners_synced'] == 1, stats

            assert stats['n_miners_positions_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_positions_kept'] == 0, (i, stats_str)
            assert stats['n_miners_positions_matched'] == 1, (i, stats_str)
            assert stats['n_miners_positions_inserted'] == 0, (i, stats_str)

            assert stats['n_miners_orders_deleted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_inserted'] == 0, (i, stats_str)
            assert stats['n_miners_orders_matched'] == 1, (i, stats_str)
            assert stats['n_miners_orders_kept'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 0, (i, stats_str)
            assert stats['positions_matched'] == 1, (i, stats_str)
            assert stats['positions_deleted'] == 0, (i, stats_str)
            assert stats['positions_kept'] == 0, (i, stats_str)

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 1, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

        # Test fragmentation especially hard. May need to do a hard snap to candidates.
    def test_validate_order_sync_keep_recent_orders_one_insert(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            # Ensure they are in the future
            o.order_uuid = i
            o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])


        position = deepcopy(self.default_position)
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 0, stats
        assert stats['n_miners_positions_kept'] == 0, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 0, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 1, stats
        assert stats['n_miners_orders_matched'] == 0, stats
        assert stats['n_miners_orders_kept'] == 1, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 0, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 3, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms

    def test_validate_order_sync_keep_recent_orders_match_one_by_uuid(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            # Ensure they are in the future
            o.order_uuid = o.order_uuid if i == 0 else i
            o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])

        position = deepcopy(self.default_position)
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 0, stats
        assert stats['n_miners_positions_kept'] == 0, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 0, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 0, stats
        assert stats['n_miners_orders_matched'] == 1, stats
        assert stats['n_miners_orders_kept'] == 1, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 0, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 2, stats


        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

    def test_validate_order_sync_one_of_each(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            o.order_uuid = o.order_uuid if i == 0 else i

            # Will still allow for a match
            if i == 0:
                o.processed_ms += 1000 * 60 * 2
            # Allow for a delete. mismatched attributes
            if i == 1:
                o.processed_ms = self.default_order.processed_ms - 1000 * 60 * 30
                o.order_type = OrderType.SHORT
            if i == 2:
                # Will allow for a keep as it is in the future.
                o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
                o.order_type = OrderType.SHORT
                o.leverage = -4

        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])
        position = deepcopy(self.default_position)
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = self.default_order.processed_ms + 1000 * 60 * 20
        order_to_insert.leverage = 5
        order_to_match = deepcopy(self.default_order)
        position.orders = [order_to_match, order_to_insert]
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 0, stats
        assert stats['n_miners_positions_kept'] == 0, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 0, stats

        assert stats['n_miners_orders_deleted'] == 1, stats
        assert stats['n_miners_orders_inserted'] == 1, stats
        assert stats['n_miners_orders_matched'] == 1, stats
        assert stats['n_miners_orders_kept'] == 1, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 1, stats
        assert stats['orders_kept'] == 1, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms - 1000 * 60 * 30

    def test_validate_position_sync_one_of_each_uuid_match(self):
        dp_to_keep = deepcopy(self.default_closed_position)
        dp_to_keep.position_uuid = "to_keep"
        dp_to_keep.open_ms += 1000 * 60 * 30

        dp_to_delete = deepcopy(self.default_closed_position)
        dp_to_delete.position_uuid = "to_delete"
        dp_to_delete.open_ms -= 1000 * 60 * 10
        dp_to_delete.close_ms += 1000 * 60 * 10

        dp_to_match = deepcopy(self.default_closed_position)
        dp_to_match.position_uuid = "to_match"
        dp_to_match.orders[0].processed_ms += 1000

        cp_to_insert = deepcopy(self.default_closed_position)
        cp_to_insert.position_uuid = "to_insert"
        cp_to_insert.orders[0].order_type = OrderType.SHORT
        cp_to_insert.open_ms += 1000 * 60 * 1

        # Let's do heuristic match
        cp_to_match = deepcopy(self.default_closed_position)
        cp_to_match.position_uuid = "to_match"
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = self.default_order.processed_ms + 1000 * 60 * 20
        cp_to_match.orders.append(order_to_insert)

        disk_positions = self.positions_to_disk_data([dp_to_keep, dp_to_delete, dp_to_match])
        candidate_data = self.positions_to_candidate_data([cp_to_insert, cp_to_match])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "

        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 1, stats
        assert stats['n_miners_positions_kept'] == 1, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 1, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 1, stats
        assert stats['n_miners_orders_matched'] == 1, stats
        assert stats['n_miners_orders_kept'] == 0, stats

        assert stats['positions_inserted'] == 1, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 1, stats
        assert stats['positions_kept'] == 1, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[
                   self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms - 1000 * 60 * 10


    def test_validate_position_sync_one_of_each_heuristic_match(self):
        dp_to_keep = deepcopy(self.default_closed_position)
        dp_to_keep.position_uuid = "to_keep"
        dp_to_keep.open_ms += 1000 * 60 * 30

        dp_to_delete = deepcopy(self.default_closed_position)
        dp_to_delete.position_uuid = "to_delete"
        dp_to_delete.open_ms -= 1000 * 60 * 10
        dp_to_delete.close_ms += 1000 * 60 * 10

        # Let's do heuristic match
        dp_to_match = deepcopy(self.default_closed_position)
        dp_to_match.position_uuid = "to_match"
        dp_to_match.orders[0].processed_ms += 1000

        cp_to_insert = deepcopy(self.default_closed_position)
        cp_to_insert.position_uuid = "to_insert"
        cp_to_insert.orders[0].order_type = OrderType.SHORT
        cp_to_insert.open_ms += 1000 * 60 * 1

        # Let's do heuristic match
        cp_to_match = deepcopy(self.default_closed_position)
        cp_to_match.position_uuid = "candy_to_match"
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = self.default_order.processed_ms + 1000 * 60 * 20
        cp_to_match.orders.append(order_to_insert)


        disk_positions = self.positions_to_disk_data([dp_to_keep, dp_to_delete, dp_to_match])
        candidate_data = self.positions_to_candidate_data([cp_to_insert, cp_to_match])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 1, stats
        assert stats['n_miners_positions_kept'] == 1, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 1, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 1, stats
        assert stats['n_miners_orders_matched'] == 1, stats
        assert stats['n_miners_orders_kept'] == 0, stats

        assert stats['positions_inserted'] == 1, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 1, stats
        assert stats['positions_kept'] == 1, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms - 1000 * 60 * 10


    def test_validate_order_sync_testing_hardsnap(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            o.order_uuid = o.order_uuid if i == 0 else i

            # Will still allow for a match
            if i == 0:
                o.processed_ms += 1000 * 60 * 2
            # Keeping this monstrosity only because it comes after hardsnap
            if i == 1:
                o.processed_ms = self.default_order.processed_ms + 1000 * 60 * 30
                o.order_type = OrderType.SHORT
            if i == 2:
                # Will allow for a keep as it is in the future.
                o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
                o.order_type = OrderType.SHORT
                o.leverage = -4

        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])
        position = deepcopy(self.default_position)
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = self.default_order.processed_ms + 1000 * 60 * 20
        order_to_insert.leverage = 5
        order_to_match = deepcopy(self.default_order)
        position.orders = [order_to_match, order_to_insert]
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['n_miners_synced'] == 1, stats

        assert stats['n_miners_positions_deleted'] == 0, stats
        assert stats['n_miners_positions_kept'] == 0, stats
        assert stats['n_miners_positions_matched'] == 1, stats
        assert stats['n_miners_positions_inserted'] == 0, stats

        assert stats['n_miners_orders_deleted'] == 0, stats
        assert stats['n_miners_orders_inserted'] == 1, stats
        assert stats['n_miners_orders_matched'] == 1, stats
        assert stats['n_miners_orders_kept'] == 1, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 2, stats


        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == order_to_insert.processed_ms



