import json
import time
from copy import deepcopy

from vali_objects.utils.auto_sync import PositionSyncer, AUTO_SYNC_ORDER_LAG_MS
from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.p2p_syncer import P2PSyncer
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

        self.p2p_syncer = P2PSyncer()

    def test_checkpoint_syncing_order_with_median_price(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.price = 1.0
        orders = [order1]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.price = 2.0
        orders = [order1]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.price = 3.0
        orders = [order1]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        # print(json.dumps(checkpoint1, indent=4))

        self.p2p_syncer.add_checkpoint(checkpoint1, 3)
        self.p2p_syncer.add_checkpoint(checkpoint2, 3)
        self.p2p_syncer.add_checkpoint(checkpoint3, 3)

        # print(json.dumps(self.p2p_syncer.golden, indent=4))

        assert len(self.p2p_syncer.golden[self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden[self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1
        assert self.p2p_syncer.golden[self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"][0]["price"] == 2.0

    def test_checkpoint_syncing_order_not_in_majority(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order0, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        self.p2p_syncer.add_checkpoint(checkpoint1, 3)
        self.p2p_syncer.add_checkpoint(checkpoint2, 3)
        self.p2p_syncer.add_checkpoint(checkpoint3, 3)

        assert len(self.p2p_syncer.golden[self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden[self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2

    def test_checkpoint_syncing_position_not_in_majority(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        orders = [order0]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position2"
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        self.p2p_syncer.add_checkpoint(checkpoint1, 3)
        self.p2p_syncer.add_checkpoint(checkpoint2, 3)
        self.p2p_syncer.add_checkpoint(checkpoint3, 3)

        assert len(self.p2p_syncer.golden[self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden[self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1
