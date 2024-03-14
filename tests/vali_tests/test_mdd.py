# developer: jbonilla
# Copyright © 2023 Taoshi Inc
from shared_objects.challenge_utils import ChallengeBase
from tests.shared_objects.mock_classes import MockMetagraph, MockMDDChecker
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.MDDChecker import MDDChecker
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order
from data_generator.twelvedata_service import TwelveDataService

class TestMDDChecker(TestBase):

    def setUp(self):
        super().setUp()
        self.MINER_HOTKEY = "test_miner"
        self.mockMetagraph = MockMetagraph([self.MINER_HOTKEY])
        self.mddChecker = MockMDDChecker(self.mockMetagraph)
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
            self.assertAlmostEquals(v1['dd'], v2['dd'], places=2)

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
        failure_row = ChallengeBase.generate_elimination_row(self.MINER_HOTKEY, 0, MDDChecker.MAX_TOTAL_DRAWDOWN)
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
        failure_row = ChallengeBase.generate_elimination_row(self.MINER_HOTKEY, 0, MDDChecker.MAX_TOTAL_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])

    def test_mdd_failure_with_two_open_orders_different_trade_pairs(self):
        self.verify_elimination_data_in_memory_and_disk([])
        position_btc = self.trade_pair_to_default_position[TradePair.BTCUSD]
        position_eth = self.trade_pair_to_default_position[TradePair.ETHUSD]

        position_eth.position_uuid = self.DEFAULT_TEST_POSITION_UUID + '_eth'
        position_btc.position_uuid = self.DEFAULT_TEST_POSITION_UUID + '_btc'

        live_btc_price = self.tds.get_close(trade_pair=TradePair.BTCUSD)[TradePair.BTCUSD]
        live_eth_price = self.tds.get_close(trade_pair=TradePair.ETHUSD)[TradePair.ETHUSD]

        o1 = Order(order_type=OrderType.LONG,
                leverage=1.0,
                price=live_btc_price * (1 + (1 - ValiConfig.MAX_TOTAL_DRAWDOWN)), # Barely above the threshold for elimination
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        o2 = Order(order_type=OrderType.LONG,
                leverage=1.0,
                price=live_eth_price * (1 + (1 - ValiConfig.MAX_TOTAL_DRAWDOWN)), # Barely above the threshold for elimination
                trade_pair=TradePair.ETHUSD,
                processed_ms=2000,
                order_uuid="2000")

        self.mddChecker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        position_btc.add_order(o1)
        ValiUtils.save_miner_position(self.MINER_HOTKEY, self.DEFAULT_TEST_POSITION_UUID + '_btc', position_btc)
        self.mddChecker.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])

        position_eth.add_order(o2)
        ValiUtils.save_miner_position(self.MINER_HOTKEY, self.DEFAULT_TEST_POSITION_UUID + '_eth', position_eth)
        self.mddChecker.mdd_check()
        failure_row = ChallengeBase.generate_elimination_row(self.MINER_HOTKEY, .826, MDDChecker.MAX_TOTAL_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])

    def test_no_mdd_failures(self):
        self.verify_elimination_data_in_memory_and_disk([])
        self.position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_price = self.tds.get_close(trade_pair=TradePair.BTCUSD)[TradePair.BTCUSD]
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=live_price,
                order_uuid="1000")

        o2 = Order(order_type=OrderType.LONG,
                leverage=.5,
                price=live_price,
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
        self.assertEqual(relevant_position.is_closed_position, False)
        ValiUtils.save_miner_position(self.MINER_HOTKEY, self.DEFAULT_TEST_POSITION_UUID, relevant_position)
        self.mddChecker.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])

if __name__ == '__main__':
    import unittest
    unittest.main()