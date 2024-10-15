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
from vali_objects.utils.plagiarism_events import PlagiarismEvents
from vali_objects.utils.plagiarism_pipeline import PlagiarismPipeline
from vali_objects.utils.plagiarism_definitions import LagDetection
from vali_config import ValiConfig
from vali_objects.utils.position_utils import PositionUtils

import uuid

class TestPlagiarism(TestBase):

    def setUp(self):
        super().setUp()
        self.MINER_HOTKEY1 = "test_miner1"
        self.MINER_HOTKEY2 = "test_miner2"
        self.MINER_HOTKEY3 = "test_miner3"
        self.mock_metagraph = MockMetagraph([self.MINER_HOTKEY1, self.MINER_HOTKEY2, self.MINER_HOTKEY3])
        self.plagiarism_detector = MockPlagiarismDetector(self.mock_metagraph)
        self.plagiarism_detector.running_unit_tests = True
        self.current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS

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

        self.eth_position3 = Position(
            miner_hotkey=self.MINER_HOTKEY3,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_eth3",
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

        self.btc_position3 = Position(
            miner_hotkey=self.MINER_HOTKEY3,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_btc3",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
        )
        
        self.plagiarism_detector.init_cache_files()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.tds = TwelveDataService(api_key=secrets["twelvedata_apikey"])

        self.position_manager.clear_all_miner_positions_from_disk()
        self.plagiarism_detector.clear_plagiarism_from_disk()

        self.plagiarism_detector.clear_eliminations_from_disk()


    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order)
        self.position_manager.save_miner_position_to_disk(position)

    def test_plagiarism_two_miners(self):
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms=1000,
                order_uuid="1000")
        self.add_order_to_position_and_save_to_disk(self.eth_position1, o1)
        o1.processed_ms = 1005
        self.add_order_to_position_and_save_to_disk(self.eth_position2, o1)

        self.plagiarism_detector.detect()
        self.assertEqual(len(self.plagiarism_detector.plagiarism_data), 2)

        miner_one = self.plagiarism_detector.plagiarism_data[0]
        miner_two = self.plagiarism_detector.plagiarism_data[1]
        if miner_one["plagiarist"] == self.MINER_HOTKEY1:
            self.assertAlmostEqual(miner_one["overall_score"], 0)
            self.assertAlmostEqual(miner_two["overall_score"], 1)
        else:
            self.assertAlmostEqual(miner_one["overall_score"], 1)
            self.assertAlmostEqual(miner_two["overall_score"], 0)
       # self.plagiarism_detector.clear_plagiarism_from_disk()


    def test_plagiarism_three_miners(self):

        o1 = Order(order_type=OrderType.SHORT,
                leverage=-1.0,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms=1000,
                order_uuid="1000")
        o1_copy = Order(order_type=OrderType.SHORT,
                leverage=-1.0,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms=60000 * 60 * 6 + 1000, #6 hours later
                order_uuid="1001")
        

        o2 = Order(order_type=OrderType.LONG,
                leverage=0.5,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1002")
        
        o2_copy = Order(order_type=OrderType.LONG,
                leverage=0.5,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=60000 * 5 + 1000, #5 minutes later
                order_uuid="1003")
        
        #ETH
        #miner 2 following miner 1
        self.add_order_to_position_and_save_to_disk(self.eth_position1, o1)
        self.add_order_to_position_and_save_to_disk(self.eth_position2, o1_copy)
        #BTC
        #miner 3 following miner 2
        self.add_order_to_position_and_save_to_disk(self.btc_position2, o2)
        self.add_order_to_position_and_save_to_disk(self.btc_position3, o2_copy)



        self.plagiarism_detector.detect()
        self.assertEqual(len(self.plagiarism_detector.plagiarism_data), 3)

        miner_one = miner_two = miner_three = None
        for miner in self.plagiarism_detector.plagiarism_data:
            if miner["plagiarist"] == self.MINER_HOTKEY1:
                miner_one = miner
            if miner["plagiarist"] == self.MINER_HOTKEY2:
                miner_two = miner
            if miner["plagiarist"] == self.MINER_HOTKEY3:
                miner_three = miner
            
        self.assertAlmostEqual(miner_one["overall_score"], 0)
        self.assertGreaterEqual(miner_two["overall_score"], 0.95)
        self.assertGreaterEqual(miner_three["overall_score"], 0.95)

    # Test case of no plagiarism
    # Test case of 1 plagiarism (test two miners)
    # Test case of 2 plagiarists and one innocent person for different tradepairs (change 3 miner test slightly)
    # At least 3 tests for each kind of plagiarism (except maybe combo plagiarism)
    # 1 test for low score if not plagiarism (or 0)
    # 1 test for high score if plagiarism
    # 1 test for nonzero score if similar

    # Does nothing for now
    def test_lag_detection_not_similar(self):
        alternate = False
        for i in range(5):
            victim_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24),
                order_uuid=str(uuid.uuid4()))
            
            plagiarist_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24) + ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS + 1, #30 minutes after each other
                order_uuid=str(uuid.uuid4()))
            
            # Alternate who is following so that lag threshold shouldn't be passed
            if alternate:
                self.add_order_to_position_and_save_to_disk(self.eth_position2, victim_order)
                self.add_order_to_position_and_save_to_disk(self.eth_position1, plagiarist_order)
            else:
                self.add_order_to_position_and_save_to_disk(self.eth_position1, victim_order)
                self.add_order_to_position_and_save_to_disk(self.eth_position2, plagiarist_order)
            alternate = not alternate

        hotkeys = self.mock_metagraph.hotkeys
        positions = self.position_manager.get_all_miner_positions_by_hotkey(hotkeys)
        flattened_positions = PositionUtils.flatten(positions)
        positions_list_translated = PositionUtils.translate_current_leverage(flattened_positions)
        miners, trade_pairs, state_list = PositionUtils.to_state_list(positions_list_translated, current_time=self.current_time)
        state_dict = PlagiarismPipeline.state_list_to_dict(miners, trade_pairs, state_list)
        
        
    

        PlagiarismEvents.set_positions(state_dict, miners, trade_pairs)
        miner_one_lag = LagDetection(self.MINER_HOTKEY1)
        victim_key = (self.MINER_HOTKEY2, TradePair.ETHUSD.name)
        #Consider what the threshold should really be for lag score
        miner_one_score = miner_one_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key)
        self.assertAlmostEqual(miner_one_score, 1)
        victim_key = (self.MINER_HOTKEY1, TradePair.ETHUSD.name)
        miner_two_lag = LagDetection(self.MINER_HOTKEY2)
        miner_two_score = miner_two_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key)
        self.assertAlmostEqual(miner_two_score, 1)
        PlagiarismEvents.clear_plagiarism_events()



    def test_lag_detection_plagiarism(self):

        victim_order = Order(order_type=OrderType.SHORT,
                leverage=-0.03,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= 0,
                order_uuid=str(uuid.uuid4()))
        self.add_order_to_position_and_save_to_disk(self.eth_position1, victim_order)

        for i in range(1, 5):
            victim_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24),
                order_uuid=str(uuid.uuid4()))
            
            plagiarist_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24) + 1000 * 60 * 60 * 1, # 1 hour after each other
                order_uuid=str(uuid.uuid4()))
                    
            self.add_order_to_position_and_save_to_disk(self.eth_position1, victim_order)
            self.add_order_to_position_and_save_to_disk(self.eth_position2, plagiarist_order)

        hotkeys = self.mock_metagraph.hotkeys
        positions = self.position_manager.get_all_miner_positions_by_hotkey(hotkeys)
        flattened_positions = PositionUtils.flatten(positions)
        positions_list_translated = PositionUtils.translate_current_leverage(flattened_positions)
        miners, trade_pairs, state_list = PositionUtils.to_state_list(positions_list_translated, current_time=self.current_time)
        state_dict = PlagiarismPipeline.state_list_to_dict(miners, trade_pairs, state_list)
        
        
    

        PlagiarismEvents.set_positions(state_dict, miners, trade_pairs)
        miner_one_lag = LagDetection(self.MINER_HOTKEY1)
        victim_key = (self.MINER_HOTKEY2, TradePair.ETHUSD.name)
        #Consider what the threshold should really be for lag score
        miner_one_score = miner_one_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key)
        self.assertLessEqual(miner_one_score, 1)
        victim_key = (self.MINER_HOTKEY1, TradePair.ETHUSD.name)
        miner_two_lag = LagDetection(self.MINER_HOTKEY2)
        miner_two_score = miner_two_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key)
        self.assertGreater(miner_two_score, 1)

        PlagiarismEvents.clear_plagiarism_events()


    
    """
    def _test_plagiarism_all_zero_scores(self):
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
        """

