# developer: jbonilla
# Copyright © 2024 Taoshi Inc
from shared_objects.cache_controller import CacheController
from tests.shared_objects.mock_classes import MockMetagraph, MockMDDChecker
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order
from data_generator.twelvedata_service import TwelveDataService

class TestMDDChecker(TestBase):

    def setUp(self):
        super().setUp()
        self.MINER_HOTKEY = "test_miner"
        self.mock_metagraph = MockMetagraph([self.MINER_HOTKEY])
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.mdd_checker = MockMDDChecker(self.mock_metagraph, self.position_manager)
        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.trade_pair_to_default_position = {x: Position(
            miner_hotkey=self.MINER_HOTKEY,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + str(x.trade_pair_id),
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=x,
        ) for x in TradePair}

        self.mdd_checker.init_cache_files()
        self.mdd_checker.clear_eliminations_from_disk()
        self.position_manager.clear_all_miner_positions_from_disk()
        secrets = ValiUtils.get_secrets()
        secrets["twelvedata_apikey"] = secrets["twelvedata_apikey2"]
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets)

    def verify_elimination_data_in_memory_and_disk(self, expected_eliminations):
        self.mdd_checker._refresh_eliminations_in_memory_and_disk()
        expected_eliminated_hotkeys = [x['hotkey'] for x in expected_eliminations]
        eliminated_hotkeys = [x['hotkey'] for x in self.mdd_checker.eliminations]
        self.assertEqual(len(eliminated_hotkeys),
                         len(expected_eliminated_hotkeys),
                         "Eliminated hotkeys in memory/disk do not match expected. eliminated_hotkeys: "
                         + str(eliminated_hotkeys) + " expected_eliminated_hotkeys: " + str(expected_eliminated_hotkeys))
        self.assertEqual(set(eliminated_hotkeys), set(expected_eliminated_hotkeys))
        for v1, v2 in zip(expected_eliminations, self.mdd_checker.eliminations):
            self.assertEqual(v1['hotkey'], v2['hotkey'])
            self.assertEqual(v1['reason'], v2['reason'])
            self.assertAlmostEquals(v1['elimination_initiated_time_ms'] / 1000.0, v2['elimination_initiated_time_ms'] / 1000.0, places=1)
            self.assertAlmostEquals(v1['dd'], v2['dd'], places=2)

    def verify_positions_on_disk(self, in_memory_positions, assert_all_closed=None, assert_all_open=None):
        positions_from_disk = self.position_manager.get_all_miner_positions(self.MINER_HOTKEY, only_open_positions=False)
        self.assertEqual(len(positions_from_disk), len(in_memory_positions),
                         f"Mismatched number of positions. Positions on disk: {positions_from_disk}"
                         f" Positions in memory: {in_memory_positions}")
        for position in in_memory_positions:
            matching_disk_position = next((x for x in positions_from_disk if x.position_uuid == position.position_uuid), None)
            self.position_manager.positions_are_the_same(position, matching_disk_position)
            if assert_all_closed:
                self.assertTrue(matching_disk_position.is_closed_position, f"Position in memory: {position} Position on disk: {matching_disk_position}")
            if assert_all_open:
                self.assertFalse(matching_disk_position.is_closed_position)


    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order)
        self.position_manager.save_miner_position_to_disk(position)

    def test_mdd_failure_max_total_drawdown(self):
        self.verify_elimination_data_in_memory_and_disk([])
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=111111111,
                order_uuid="1000")

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.assertFalse(relevant_position.is_closed_position)
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)
        self.mdd_checker.mdd_check()
        failure_row = CacheController.generate_elimination_row(relevant_position.miner_hotkey, 0, MDDChecker.MAX_TOTAL_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])
        self.verify_positions_on_disk([relevant_position], assert_all_closed=True)


    def test_mdd_failure_with_closed_position_daily_drawdown(self):
        self.verify_elimination_data_in_memory_and_disk([])
        live_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        # o2 has the timestamp used for determining if this is a daily failure
        o2 = Order(order_type=OrderType.FLAT,
                leverage=0,
                price=live_price * 100,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.assertFalse(relevant_position.is_closed_position)
        self.verify_positions_on_disk([relevant_position])
        self.mdd_checker.mdd_check()
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o2)
        self.assertTrue(relevant_position.is_closed_position)
        self.mdd_checker.mdd_check()
        failure_row = CacheController.generate_elimination_row(relevant_position.miner_hotkey, 0, MDDChecker.MAX_DAILY_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])
        self.verify_positions_on_disk([relevant_position], assert_all_closed=True)

    def test_mdd_failure_with_closed_position_total_drawdown(self):
        self.verify_elimination_data_in_memory_and_disk([])
        live_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        # o2 has the timestamp used for determining if this is a daily failure
        o2 = Order(order_type=OrderType.FLAT,
                leverage=0,
                price=live_price * 100,
                trade_pair=TradePair.BTCUSD,
                processed_ms=11111111,
                order_uuid="2000")

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.assertFalse(relevant_position.is_closed_position)
        self.verify_positions_on_disk([relevant_position])
        self.mdd_checker.mdd_check()
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o2)
        self.assertTrue(relevant_position.is_closed_position)
        self.mdd_checker.mdd_check()
        failure_row = CacheController.generate_elimination_row(relevant_position.miner_hotkey, 0, MDDChecker.MAX_TOTAL_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])
        self.verify_positions_on_disk([relevant_position], assert_all_closed=True)


    def test_mdd_failure_with_two_open_orders_different_trade_pairs(self):
        self.verify_elimination_data_in_memory_and_disk([])
        position_btc = self.trade_pair_to_default_position[TradePair.BTCUSD]
        position_eth = self.trade_pair_to_default_position[TradePair.ETHUSD]

        position_eth.position_uuid = self.DEFAULT_TEST_POSITION_UUID + '_eth'
        position_btc.position_uuid = self.DEFAULT_TEST_POSITION_UUID + '_btc'

        live_btc_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        live_eth_price = self.live_price_fetcher.get_close(trade_pair=TradePair.ETHUSD)

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

        self.mdd_checker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(position_btc, o1)
        self.mdd_checker.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([position_btc], assert_all_open=True)

        self.add_order_to_position_and_save_to_disk(position_eth, o2)
        self.mdd_checker.mdd_check()
        failure_row = CacheController.generate_elimination_row(position_eth.miner_hotkey, .826, MDDChecker.MAX_TOTAL_DRAWDOWN)
        self.verify_elimination_data_in_memory_and_disk([failure_row])
        self.verify_positions_on_disk([position_btc, position_eth])

    def test_no_mdd_failures(self):
        self.verify_elimination_data_in_memory_and_disk([])
        self.position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
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
        self.mdd_checker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.mdd_checker.mdd_check()
        self.assertEqual(relevant_position.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)

        self.add_order_to_position_and_save_to_disk(relevant_position, o2)
        self.assertEqual(relevant_position.is_closed_position, False)
        self.mdd_checker.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)


    def test_no_mdd_failures_high_leverage_one_order(self):
        self.verify_elimination_data_in_memory_and_disk([])
        position_btc = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_btc_price = self.live_price_fetcher.get_close(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.LONG,
                leverage=20.0,
                price=live_btc_price *1.001, # Down 0.1%
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")


        self.mdd_checker.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(position_btc, o1)
        self.mdd_checker.mdd_check()
        self.assertEqual(position_btc.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([position_btc], assert_all_open=True)

        btc_position_from_disk = self.position_manager.get_all_miner_positions(self.MINER_HOTKEY, only_open_positions=False)[0]
        print("Position return on BTC after mdd_check:", btc_position_from_disk.current_return)
        print("Max MDD for closed positions:", self.mdd_checker.portfolio_max_dd_closed_positions)
        print("Max MDD for all positions:", self.mdd_checker.portfolio_max_dd_all_positions)


        print("Adding ETH position")
        position_eth = self.trade_pair_to_default_position[TradePair.ETHUSD]
        live_eth_price = self.live_price_fetcher.get_close(trade_pair=TradePair.ETHUSD)
        o2 = Order(order_type=OrderType.LONG,
                   leverage=20.0,
                   price=live_eth_price * 1.001,  # Down 0.1%
                   trade_pair=TradePair.ETHUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        self.add_order_to_position_and_save_to_disk(position_eth, o2)
        self.mdd_checker.mdd_check()
        positions_from_disk = self.position_manager.get_all_miner_positions(self.MINER_HOTKEY, only_open_positions=False)
        for p in positions_from_disk:
            print('individual position return', p.trade_pair, p.current_return)
        print("Max MDD for closed positions:", self.mdd_checker.portfolio_max_dd_closed_positions)
        print("Max MDD for all position:", self.mdd_checker.portfolio_max_dd_all_positions)


if __name__ == '__main__':
    import unittest
    unittest.main()