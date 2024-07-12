import json
import time
from copy import deepcopy

from bittensor import Balance

from time_util.time_util import TimeUtil
from vali_objects.utils.auto_sync import PositionSyncer
from tests.shared_objects.mock_classes import MockMetagraph, MockNeuron, MockAxonInfo
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
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()  # 1718071209000
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

        self.default_neuron = MockNeuron(axon_info=MockAxonInfo("0.0.0.0"),
                                         stake=Balance(0.0))

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

    def test_get_validators(self):
        neuron1 = deepcopy(self.default_neuron)
        neuron1.stake = Balance(2000.0)
        neuron1.axon_info = MockAxonInfo(ip="test_ip1")

        neuron2 = deepcopy(self.default_neuron)
        neuron2.stake = Balance(0.0)
        neuron2.axon_info = MockAxonInfo(ip="test_ip2")

        neuron3 = deepcopy(self.default_neuron)
        neuron3.stake = Balance(2000.0)

        neuron4 = deepcopy(self.default_neuron)
        neuron4.stake = Balance(0.0)

        self.neurons = [neuron1, neuron2, neuron3, neuron4]
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY], self.neurons)
        validator_axons = self.p2p_syncer.get_validators(self.mock_metagraph.neurons)

        assert len(validator_axons) == 1

    def test_get_largest_staked_validators(self):
        neuron1 = deepcopy(self.default_neuron)
        neuron1.stake = Balance(2000.0)
        neuron1.axon_info = MockAxonInfo(ip="test_ip1")

        neuron2 = deepcopy(self.default_neuron)
        neuron2.stake = Balance(0.0)
        neuron2.axon_info = MockAxonInfo(ip="test_ip2")

        neuron3 = deepcopy(self.default_neuron)
        neuron3.stake = Balance(2000.0)

        neuron4 = deepcopy(self.default_neuron)
        neuron4.stake = Balance(0.0)

        neuron5 = deepcopy(self.default_neuron)
        neuron5.stake = Balance(3000.0)
        neuron5.axon_info = MockAxonInfo(ip="test_ip5")

        neuron6 = deepcopy(self.default_neuron)
        neuron6.stake = Balance(4000.0)
        neuron6.axon_info = MockAxonInfo(ip="test_ip6")

        self.neurons = [neuron1, neuron2, neuron3, neuron4, neuron5, neuron6]
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY], self.neurons)
        validator_axons = self.p2p_syncer.get_largest_staked_validators(2, self.mock_metagraph.neurons)

        assert len(validator_axons) == 2
        assert validator_axons[0].ip == "test_ip6"

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

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}

        self.p2p_syncer.create_golden(checkpoints)

        # print(json.dumps(self.p2p_syncer.golden, indent=4))

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1
        assert self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"][0]["price"] == 2.0

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

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2], "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2

    def test_checkpoint_syncing_order_not_in_majority_with_multiple_positions(self):
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

        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order3"
        orders = [order3]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders()

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string()), json.loads(position2.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order4"
        orders = [order3]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders()

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string()), json.loads(position2.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2], "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][1]["orders"]) == 2

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

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1

    def test_checkpoint_syncing_multiple_positions(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders()

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders()

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        orders = [order0]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders()

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string()), json.loads(position2.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        orders = [order0]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders()

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position2.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1

    def test_checkpoint_syncing_multiple_miners(self):
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

        checkpoint2 = {"positions": {"diff_miner": {"positions": [json.loads(position.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2]}
        self.p2p_syncer.create_golden(checkpoints)

        # print(json.dumps(self.p2p_syncer.golden, indent=4))

        assert len(self.p2p_syncer.golden["positions"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"]["diff_miner"]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1
        assert len(self.p2p_syncer.golden["positions"]["diff_miner"]["positions"][0]["orders"]) == 1

    def test_checkpoint_syncing_miner_not_in_majority(self):
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

        checkpoint2 = {"positions": {"diff_miner": {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders()

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        # print(json.dumps(self.p2p_syncer.golden, indent=4))

        assert len(self.p2p_syncer.golden["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1

    def test_checkpoint_syncing_checkpoint_is_stale(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = 100
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

        checkpoint2 = {"positions": {"diff_miner": {"positions": [json.loads(position.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"]["diff_miner"]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"]["diff_miner"]["positions"][0]["orders"]) == 1
