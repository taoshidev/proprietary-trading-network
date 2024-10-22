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
from vali_objects.utils.plagiarism_definitions import FollowPercentage
from vali_objects.utils.plagiarism_definitions import CopySimilarity
from vali_objects.utils.plagiarism_definitions import TwoCopySimilarity
from vali_objects.utils.plagiarism_definitions import ThreeCopySimilarity

from copy import deepcopy

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerData
from tests.shared_objects.mock_classes import (
    MockMetagraph, MockChallengePeriodManager, MockPositionManager, MockPerfLedgerManager, MockCacheController
)
from tests.vali_tests.base_objects.test_base import TestBase

from vali_config import TradePair
from vali_objects.position import Position
from vali_config import ValiConfig

import copy

from vali_config import ValiConfig
from vali_objects.utils.position_utils import PositionUtils


import uuid

class TestPlagiarism(TestBase):

    def setUp(self):

        super().setUp()

        self.ONE_DAY_MS = 1000 * 60 * 60 * 24
        self.ONE_HOUR_MS = 1000 * 60 * 60
        self.ONE_MIN_MS = 1000 * 60

        self.N_MINERS = 5
        self.MINER_NAMES = [f"test_miner{i}" for i in range(self.N_MINERS)]


        self.mock_metagraph = MockMetagraph(self.MINER_NAMES)
        self.plagiarism_detector = MockPlagiarismDetector(self.mock_metagraph)
        self.plagiarism_detector.running_unit_tests = True
        self.current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS

        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1

    
        self.plagiarism_detector.init_cache_files()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.tds = TwelveDataService(api_key=secrets["twelvedata_apikey"])

        self.position_manager.clear_all_miner_positions_from_disk()
        self.plagiarism_detector.clear_plagiarism_from_disk()

        self.plagiarism_detector.clear_eliminations_from_disk()
        self.position_counter = 0
        PlagiarismEvents.clear_plagiarism_events()

        # Set up miners with postions for btc and eth
        # This will involve setting up 6 positions that aren't plagiarism

        # One position with low leverage and orders a day apart 

        self.miner_0_btc_lev = [0.01, 0.02, -0.01]
        self.generate_one_position(hotkey=self.MINER_NAMES[0],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=self.miner_0_btc_lev,
                                   times_apart=[self.ONE_DAY_MS for _ in range(len(self.miner_0_btc_lev))],
                                   open_ms=0
                                 )
        # One position, short open time
        miner_0_eth_lev = [-0.2]
        self.generate_one_position(hotkey=self.MINER_NAMES[0],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_0_eth_lev,
                                   times_apart=[0],
                                   open_ms=self.ONE_HOUR_MS,
                                   close_ms=self.ONE_HOUR_MS * 6)
        
        #TODO investigate if FLATS get appended

        # Two positions, higher leverage each 2.5 days apart with one order
        miner_1_btc_lev_one = [0.5]
        miner_1_btc_close_one = self.ONE_HOUR_MS * 3 + (self.ONE_DAY_MS * 2.5)
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_1_btc_lev_one,
                                   times_apart=[0],
                                   open_ms=self.ONE_HOUR_MS * 3,
                                   close_ms=miner_1_btc_close_one)
        
        miner_1_btc_lev_two = [-0.5]
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_1_btc_lev_two,
                                   times_apart=[0],
                                   open_ms=miner_1_btc_close_one + (self.ONE_HOUR_MS))

        miner_1_eth_lev_one = [-0.3]
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_1_eth_lev_one,
                                   times_apart=[0],
                                   open_ms=self.ONE_HOUR_MS * 3,
                                   close_ms=miner_1_btc_close_one)
        miner_1_eth_lev_two = [0.2]
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_1_eth_lev_two,
                                   times_apart=[0],
                                   open_ms=miner_1_btc_close_one + (self.ONE_HOUR_MS),
                                   close_ms=miner_1_btc_close_one)

        # Three positions, somewhat frequent orders (6 hours apart)
        miner_2_btc_lev_one = [0.01, 0.4, -0.1, 0.1]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_2_btc_lev_one,
                                   times_apart=[self.ONE_HOUR_MS * 6 for _ in range(len(miner_2_btc_lev_one))],
                                   open_ms=self.ONE_DAY_MS,
                                   close_ms=self.ONE_DAY_MS * 2.25)
        miner_2_btc_lev_two = [0.01, 0.2, 0.1, -0.2]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_2_btc_lev_two,
                                   times_apart=[self.ONE_HOUR_MS * 6 for _ in range(len(miner_2_btc_lev_two))],
                                   open_ms=self.ONE_DAY_MS * 2.25 + self.ONE_MIN_MS,
                                   close_ms=self.ONE_DAY_MS * 3.5)
        
        miner_2_btc_lev_three = [-0.1, -0.05, 0.1, -0.1]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_2_btc_lev_three,
                                   times_apart=[self.ONE_HOUR_MS * 6 for _ in range(len(miner_2_btc_lev_three))],
                                   open_ms=self.ONE_DAY_MS * 3.5 + self.ONE_DAY_MS)

        #One Position, One Order
        miner_2_eth_lev = [-0.3]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_2_eth_lev,
                                   times_apart=[0],
                                   open_ms=0)

        # Different times apart two positions
        miner_3_btc_lev_one = [0.33, -0.05, -0.05] # 12, 8 hours apart
        self.generate_one_position(hotkey=self.MINER_NAMES[3],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_3_btc_lev_one,
                                   times_apart=[0, self.ONE_HOUR_MS * 12, self.ONE_HOUR_MS * 8],
                                   open_ms=self.ONE_HOUR_MS * 13,
                                   close_ms=self.ONE_DAY_MS * 3)
        
        miner_3_btc_lev_two = [0.05, 0.1, 0.2, -0.1] # 30 min, 6 hours, 10 min
        self.generate_one_position(hotkey=self.MINER_NAMES[3],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_3_btc_lev_two,
                                   times_apart=[0, self.ONE_MIN_MS * 30, self.ONE_HOUR_MS * 6, self.ONE_MIN_MS * 10],
                                   open_ms=self.ONE_DAY_MS * 3 + self.ONE_HOUR_MS)

        # Longer Different Times one position
        miner_3_eth_lev_one = [0.25, 0.1, -0.2] # 1 day, 2 days
        self.generate_one_position(hotkey=self.MINER_NAMES[3],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_3_eth_lev_one,
                                   times_apart=[0, self.ONE_DAY_MS, self.ONE_DAY_MS * 2],
                                   open_ms=self.ONE_MIN_MS * 10)
        # Figure out a way to then add plagiarism, I'll just have an extra miner that doesn't have any positions unless plagiarism

    def translate_positions_to_states(self):
        hotkeys = self.mock_metagraph.hotkeys
        positions = self.position_manager.get_all_miner_positions_by_hotkey(hotkeys)
        flattened_positions = PositionUtils.flatten(positions)
        positions_list_translated = PositionUtils.translate_current_leverage(flattened_positions)
        miners, trade_pairs, state_list = PositionUtils.to_state_list(positions_list_translated, current_time=self.current_time)
        state_dict = PlagiarismPipeline.state_list_to_dict(miners, trade_pairs, state_list)
        
        PlagiarismEvents.set_positions(state_dict, miners, trade_pairs, current_time=self.current_time)


    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order)
        self.position_manager.save_miner_position_to_disk(position)

    def generate_one_position(self, hotkey, trade_pair, leverages, times_apart, open_ms, close_ms=None, times_after=None):
        if times_after is None:
            times_after = [0 for _ in range(len(leverages))]
        if close_ms is None:
            close_ms = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS + 1
        
        self.position_counter += 1
        position = Position(
            miner_hotkey=hotkey,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + f"pos{self.position_counter}",
            open_ms=open_ms,
            trade_pair=trade_pair,
        )
        
        for i in range(len(leverages)):

            if leverages[i] > 0: 
                type = OrderType.LONG
            elif leverages[i] < 0:
                type = OrderType.SHORT
            else:
                type = OrderType.FLAT
        
            order = Order(order_type=type,
                leverage=leverages[i],
                price=1000,
                trade_pair=position.trade_pair,
                processed_ms= open_ms + (i * times_apart[i]) + times_after[i],
                order_uuid=str(uuid.uuid4()))
            self.add_order_to_position_and_save_to_disk(position, order)
        position.close_ms = close_ms
        position.is_closed_position = close_ms < ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        self.position_manager.save_miner_position_to_disk(position)

    def generate_plagiarism_position(self, plagiarist_key, victim_key, plagiarist_leverages, victim_leverages, times_apart, open_ms, close_ms=None,  times_after=None):

        self.generate_one_position(plagiarist_key[0], plagiarist_key[1], leverages=plagiarist_leverages, times_apart=times_apart, open_ms=open_ms, close_ms=close_ms, times_after=times_after)
        self.generate_one_position(victim_key[0], victim_key[1], leverages=victim_leverages, times_apart=times_apart, open_ms=open_ms, close_ms=close_ms)

    def test_plagiarism_two_miners(self):
        #Direct copy one order.
        self.position_manager.clear_all_miner_positions_from_disk()
        self.position_manager.get_miner_position_from_disk_using_position_in_memory
        leverages = [1.0]
        plagiarist_key = (self.MINER_NAMES[1], TradePair.ETHUSD)
        victim_key = (self.MINER_NAMES[0], TradePair.ETHUSD)
        self.generate_plagiarism_position(plagiarist_key=plagiarist_key,
                                        victim_key=victim_key,
                                        plagiarist_leverages=leverages,
                                        victim_leverages=leverages,
                                        times_apart=[1000 * 60 * 60 * 24],
                                        open_ms=0,
                                        times_after=[1000 * 60])
        self.plagiarism_detector.detect()
        self.assertEqual(len(self.plagiarism_detector.plagiarism_data), 2)

        for miner in self.plagiarism_detector.plagiarism_data:
            if miner["plagiarist"] == self.MINER_NAMES[1]:
                self.assertAlmostEqual(miner["overall_score"], 1)
            elif miner["plagiarist"] == self.MINER_NAMES[0]:
                self.assertAlmostEqual(miner["overall_score"], 0)


    def test_no_plagiarism(self):
        # There should be no false positives
        self.plagiarism_detector.detect()
        self.assertGreaterEqual(len(self.plagiarism_detector.plagiarism_data), 1)
        positions = self.position_manager.get_all_disk_positions_for_all_miners()
        for key, value in positions.items():

            self.assertAlmostEqual(value, [])

        for miner in self.plagiarism_detector.plagiarism_data:
            self.assertLess(miner["overall_score"], 0.8)

    def _test_plagiarism_scale(self):
        # Plagiarist scales the leverages of another miner with constant time lag of one hour
        # Copies Miner zero bitcoin leverages
        leverages = self.miner_0_btc_lev
        times_apart = [self.ONE_DAY_MS for _ in range(len(self.miner_0_btc_lev))] #same as for miner 0
        times_after = [self.ONE_HOUR_MS for _ in range(len(self.miner_0_btc_lev))]
        
        self.generate_one_position( hotkey=self.MINER_NAMES[4],
                                    plagiarist_leverages=leverages,
                                    times_apart=times_apart,
                                    open_ms=0,
                                    times_after=times_after)
        self.plagiarism_detector.detect()

        for miner in self.plagiarism_detector.plagiarism_data:
            if miner["plagiarist"] == self.MINER_NAMES[4]:
                self.assertGreaterEqual(miner["overall_score"], 0.95)
                trade_pairs = miner["trade_pairs"]
            elif miner["plagiarist"] == self.MINER_NAMES[0]:
                self.assertAlmostEqual(miner["overall_score"], 0)
        #TODO add more checks here to make sure it all looks good