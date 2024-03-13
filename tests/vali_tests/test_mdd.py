# developer: jbonilla
# Copyright © 2023 Taoshi Inc
from shared_objects.challenge_utils import ChallengeBase
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.MDDChecker import MDDChecker
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order
from data_generator.twelvedata_service import TwelveDataService

TEST_MINER = "test_miner"

class MockMDDChecker(MDDChecker):
    def __init__(self, metagraph):
        super().__init__(None, metagraph)

    # Lets us bypass the wait period in MDDChecker
    def get_last_update_time(self):
        return 0
class MockMetagraph():
    def __init__(self):
        self.hotkeys = [TEST_MINER]

class TestMDDChecker(TestBase):

    def setUp(self):
        super().setUp()
        self.mockMetagraph = MockMetagraph()
        self.mddChecker = MockMDDChecker(self.mockMetagraph)
        self.MINER_HOTKEY = TEST_MINER
        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.trade_pair_to_default_position = {x : Position(
            miner_hotkey=self.MINER_HOTKEY,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + str(x.trade_pair_id),
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=x,
        ) for x in TradePair}

        ValiUtils.init_cache_files(self.mockMetagraph)
        ChallengeBase.clear_eliminations_from_disk()
        ValiUtils.clear_all_miner_positions_from_disk()
        secrets = ValiUtils.get_secrets()
        self.tds = TwelveDataService(api_key=secrets["twelvedata_apikey"])

    def verify_elimination_data_in_memory_and_disk(self, expected_eliminations):
        self.mddChecker._refresh_eliminations_in_memory_and_disk()
        eliminated_hotkeys = [x['hotkey'] for x in expected_eliminations]
        expected_eliminated_hotkeys = [x['hotkey'] for x in self.mddChecker.eliminations]
        self.assertEqual(len(eliminated_hotkeys),
                         len(expected_eliminated_hotkeys),
                         "Eliminated hotkeys in memory/disk do not match expected. eliminated_hotkeys: "
                         + str(eliminated_hotkeys) + " expected_eliminated_hotkeys: " + str(expected_eliminated_hotkeys))
        self.assertEqual(set(eliminated_hotkeys), set(expected_eliminated_hotkeys))
        for v1, v2 in zip(expected_eliminations, self.mddChecker.eliminations):
            self.assertEqual(v1['hotkey'], v2['hotkey'])
            self.assertEqual(v1['reason'], v2['reason'])
            self.assertAlmostEquals(v1['elimination_initiated_time'], v2['elimination_initiated_time'], places=1)

    def test_mdd_failure_with_open_position(self):
        self.verify_elimination_data_in_memory_and_disk([])
        self.position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mddChecker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        relevant_position.add_order(o1)
        self.assertEqual(relevant_position.is_closed_position, False)
        ValiUtils.save_miner_position(self.MINER_HOTKEY, self.DEFAULT_TEST_POSITION_UUID, relevant_position)
        self.mddChecker.mdd_check()
        failure_row = ChallengeBase.generate_elimination_row(TEST_MINER, 0, MDDChecker.MAX_TOTAL_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])


    def test_mdd_failure_with_closed_position(self):
        self.verify_elimination_data_in_memory_and_disk([])
        self.position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_price = self.tds.get_close(trade_pair=TradePair.BTCUSD)[TradePair.BTCUSD]
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        o2 = Order(order_type=OrderType.FLAT,
                leverage=0,
                price=live_price * 100,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mddChecker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        relevant_position.add_order(o1)
        self.mddChecker.mdd_check()
        self.assertEqual(relevant_position.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])

        relevant_position.add_order(o2)
        self.assertEqual(relevant_position.is_closed_position, True)
        ValiUtils.save_miner_position(self.MINER_HOTKEY, self.DEFAULT_TEST_POSITION_UUID, relevant_position)
        self.mddChecker.mdd_check()
        failure_row = ChallengeBase.generate_elimination_row(TEST_MINER, 0, MDDChecker.MAX_TOTAL_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])


if __name__ == '__main__':
    import unittest
    unittest.main()