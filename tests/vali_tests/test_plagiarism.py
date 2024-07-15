# developer: jbonilla
# Copyright © 2024 Taoshi Inc
from tests.shared_objects.mock_classes import MockMetagraph, MockPlagiarismDetector
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order
from data_generator.twelvedata_service import TwelveDataService

class TestPlagiarism(TestBase):

    def setUp(self):
        super().setUp()
        self.MINER_HOTKEY1 = "test_miner1"
        self.MINER_HOTKEY2 = "test_miner2"
        self.mock_metagraph = MockMetagraph([self.MINER_HOTKEY1, self.MINER_HOTKEY2])
        self.plagiarism_detector = MockPlagiarismDetector(self.mock_metagraph)
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.eth_position1 = Position(
            miner_hotkey=self.MINER_HOTKEY1,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_eth1",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.ETHUSD,
        )
        self.eth_position2 = Position(
            miner_hotkey=self.MINER_HOTKEY2,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_eth2",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.ETHUSD,
        )
        self.btc_position1 = Position(
            miner_hotkey=self.MINER_HOTKEY1,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_btc1",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
        )
        self.btc_position2 = Position(
            miner_hotkey=self.MINER_HOTKEY2,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_btc2",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
        )

        self.plagiarism_detector.init_cache_files()
        self.plagiarism_detector.clear_eliminations_from_disk()
        self.plagiarism_detector.clear_plagiarism_scores_from_disk()
        self.position_manager.clear_all_miner_positions_from_disk()
        secrets = ValiUtils.get_secrets()
        self.tds = TwelveDataService(api_key=secrets["twelvedata_apikey"])

    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order)
        self.position_manager.save_miner_position_to_disk(position)

    def test_plagiarism_all_zero_scores(self):
        self.assertEqual({}, self.plagiarism_detector.miner_plagiarism_scores)

        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms=1000,
                order_uuid="1000")
        self.add_order_to_position_and_save_to_disk(self.eth_position1, o1)
        self.plagiarism_detector.check_plagiarism(self.eth_position1, o1)
        self.assertEqual({self.MINER_HOTKEY1: 0}, self.plagiarism_detector.miner_plagiarism_scores)

        self.add_order_to_position_and_save_to_disk(self.eth_position2, o1)
        self.plagiarism_detector.check_plagiarism(self.eth_position2, o1)
        self.assertEqual({self.MINER_HOTKEY1: 0, self.MINER_HOTKEY2: 0}, self.plagiarism_detector.miner_plagiarism_scores)

        o2 = Order(order_type=OrderType.SHORT,  # noqa: F841
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms=2000,
                order_uuid="2000")

        #TODO ...

