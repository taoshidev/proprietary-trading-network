# developer: jbonilla
import json
import datetime
from copy import deepcopy

from vali_objects.position import CRYPTO_CARRY_FEE_PER_INTERVAL, FOREX_CARRY_FEE_PER_INTERVAL, \
    INDICES_CARRY_FEE_PER_INTERVAL
from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import TradePair, ValiConfig
from vali_objects.utils.leverage_utils import LEVERAGE_BOUNDS_V2_START_TIME_MS, get_position_leverage_bounds
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position, FEE_V6_TIME_MS
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS


class TestPositions(TestBase):

    def setUp(self):
        super().setUp()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        secrets["twelvedata_apikey"] = secrets["twelvedata_apikey"]
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.position_manager.init_cache_files()
        self.position_manager.clear_all_miner_positions_from_disk()

    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order)
        self.position_manager.save_miner_position_to_disk(position)

    def _find_disk_position_from_memory_position(self, position):
        for disk_position in self.position_manager.get_all_miner_positions(position.miner_hotkey):
            if disk_position.position_uuid == position.position_uuid:
                return disk_position
        raise ValueError(f"Could not find position {position.position_uuid} in disk")

    def validate_intermediate_position_state(self, in_memory_position, expected_state):
        disk_position = self._find_disk_position_from_memory_position(in_memory_position)
        success, reason = PositionManager.positions_are_the_same(in_memory_position, expected_state)
        self.assertTrue(success, "In memory position is not as expected. " + reason)
        success, reason = PositionManager.positions_are_the_same(disk_position, expected_state)
        self.assertTrue(success, "Disc position is not as expected. " + reason)

    def test_maximum_leverage_in_interval_monotone_increasing(self):
        position = deepcopy(self.default_position)
        position.orders = []
        for i in range(10):
            o = Order(order_type=OrderType.LONG,
                      leverage=.1 + i / 10,
                      price=100,
                      trade_pair=TradePair.BTCUSD,
                      processed_ms=1000 + i * 10,
                      order_uuid=str(i))
            position.orders.append(o)
        position.rebuild_position_with_updated_orders()

        # Test various intervals
        test_intervals = [
            (1000, 1001, 0.1),  # 0.1
            (1000, 1010, 0.3),  # 0.1 + 0.2
            (1000, 1020, 0.6),  # 0.1 + 0.2 + 0.3
            (1000, 1030, 1.0),  # 0.1 + 0.2 + 0.3 + 0.4
            (1000, 1040, 1.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5
            (1000, 1050, 2.1),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6
            (1000, 1060, 2.8),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7
            (1000, 1070, 3.6),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8
            (1000, 1080, 4.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 + 0.9
            (1000, 1090, 5.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 + 0.9 + 1.0
            (1000, 1100, 5.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 + 0.9 + 1.0

            (1085, 1085, 4.5),  # zero interval length is still valid since we check inclusive.
            (1090, 1090, 5.5),  # zero interval length is still valid since we check inclusive.

            (1010, 1020, 0.6),  # max at 1020 unchanged
            (1020, 1030, 1.0),  # max at ... unchanged
            (1030, 1040, 1.5),  # max at ... unchanged
            (1040, 1050, 2.1),  # max at ... unchanged
            (1050, 1060, 2.8),  # max at ... unchanged
            (1060, 1070, 3.6),  # max at ... unchanged
            (1070, 1080, 4.5),  # max at ... unchanged
            (1080, 1090, 5.5),  # 1090 unchanged
            (1090, 1100, 5.5),  # 1090 unchanged

            (1100, 1150, 5.5),
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage
            (1500, 1500, 5.5)
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage

        ]

        for start, end, expected_leverage in test_intervals:
            msg = f"start: {start}, end: {end}, expected_leverage: {expected_leverage}"
            self.assertAlmostEquals(position.max_leverage_seen_in_interval(start, end), expected_leverage, 7, msg)

        # throw an exception for invalid interval
        invalid_intervals = [
            (900, 950),  # Interval before any order timestamps
            (1050, 1000),  # End timestamp smaller than start timestamp
            (-100, 1000),  # Negative start timestamp
            (1000, -100),  # Negative end timestamp
        ]
        for start, end in invalid_intervals:
            msg = f"start: {start}, end: {end}"
            with self.assertRaises(ValueError, msg=msg):
                _ = position.max_leverage_seen_in_interval(start, end)

    def test_maximum_leverage_in_interval_ups_and_downs(self):
        position = deepcopy(self.default_position)
        position.orders = []
        for i in range(10):
            if i % 2 == 0:
                lev = -1
            else:
                lev = 0.5

            ot = OrderType.LONG if lev > 0 else OrderType.SHORT
            o = Order(order_type=ot,
                      leverage=lev,
                      price=100,
                      trade_pair=TradePair.BTCUSD,
                      processed_ms=1000 + i * 10,
                      order_uuid=str(i))
            position.orders.append(o)
        position.rebuild_position_with_updated_orders()

        # Test various intervals
        test_intervals = [
            (1000, 1001, 1),  # -1 = -1
            (1000, 1010, 1),  # -1 + 0.5 = -0.5
            (1000, 1020, 1.5),  # -1 + 0.5 - 1 = -1.5
            (1000, 1030, 1.5),  # -1 + 0.5 - 1 + 0.5 = -1
            (1000, 1040, 2.0),  # -1 + 0.5 - 1 + 0.5 - 1 = -2
            (1000, 1050, 2.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -1.5
            (1000, 1060, 2.5),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 = -2.5
            (1000, 1070, 2.5),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -2
            (1000, 1080, 3.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 = -3
            (1000, 1090, 3.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -2.5
            (1000, 1100, 3.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -2.5

            (1085, 1085, 3.0),  # zero interval length is still valid since we check inclusive.
            (1090, 1090, 2.5),  # zero interval length is still valid since we check inclusive.

            (1010, 1020, 1.5),  # max at 1020 unchanged
            (1020, 1030, 1.5),  # max at ... unchanged
            (1030, 1040, 2.0),  # max at ... unchanged
            (1040, 1050, 2.0),  # max at ... unchanged
            (1050, 1060, 2.5),  # max at ... unchanged
            (1060, 1070, 2.5),  # max at ... unchanged
            (1070, 1080, 3.0),  # max at ... unchanged
            (1080, 1090, 3.0),  # 1090 unchanged
            (1090, 1100, 2.5),  # 1090 unchanged

            (1100, 1150, 2.5),
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage
            (1500, 1500, 2.5)
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage

        ]

        for start, end, expected_leverage in test_intervals:
            msg = f"start: {start}, end: {end}, expected_leverage: {expected_leverage}"
            self.assertAlmostEquals(position.max_leverage_seen_in_interval(start, end), expected_leverage, 7, msg)

    def test_simple_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=110,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + MS_IN_8_HOURS + 1000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.9995,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0987836209351618,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

        self.assertEqual(position.get_carry_fee(o2.processed_ms)[0], CRYPTO_CARRY_FEE_PER_INTERVAL)

    def test_carry_fee_edge_case(self):
        """
        assert next_update_time_ms > current_time_ms,
        [TimeUtil.millis_to_verbose_formatted_date_str(x) for x in
         (self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms)] + [carry_fee, position]
        AssertionError: ['2024-07-24 04:00:00.000', '2024-07-24 04:00:00.000', '2024-07-24 04:00:00.000', 0.9999979876994218,
         Position(miner_hotkey='5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X',
          position_uuid='6955409a-031e-47df-8614-4488208497a6',
          open_ms=1721228840870,
          trade_pair=<TradePair.BTCUSD: ['BTCUSD', 'BTC/USD', 0.001, 0.001, 20, <TradePairCategory.CRYPTO: 'crypto'>]>,
          orders=[Order(trade_pair=<TradePair.BTCUSD: ['BTCUSD', 'BTC/USD', 0.001, 0.001, 20, <TradePairCategory.CRYPTO: 'crypto'>]>,
                    order_type=<OrderType.LONG: 'LONG'>, leverage=0.001, price=64900.41, processed_ms=1721228840870,
                    order_uuid='6955409a-031e-47df-8614-4488208497a6', price_sources=[])],
           current_return=1.0000177396413983,
           close_ms=None,
           return_at_close=1.0000152272972591,
            net_leverage=0.001,
            average_entry_price=64900.41,
            position_type=<OrderType.LONG: 'LONG'>, is_closed_position=False)]
        """
        timestamp_ms_july_24_2024_4am = datetime.datetime(2024, 7, 24, 4, 0, 0,
                                                          tzinfo=datetime.timezone.utc).timestamp() * 1000
        timestamp_ms_july_24_2024_4am = int(timestamp_ms_july_24_2024_4am)
        t0 = 1721228840870  # Wednesday, July 17, 2024 3:07:20.870 PM
        position = Position(
            open_ms=t0,
            miner_hotkey='5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X',
            position_uuid='6955409a-031e-47df-8614-4488208497a6',
            trade_pair=TradePair.BTCUSD,
        )
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=t0,
                   order_uuid="1000")
        position.add_order(o1)
        carry_fee, next_update_time_ms = position.get_carry_fee(timestamp_ms_july_24_2024_4am)
        self.assertNotEqual(next_update_time_ms, timestamp_ms_july_24_2024_4am)
        self.assertEqual(next_update_time_ms, timestamp_ms_july_24_2024_4am + MS_IN_8_HOURS,
                         msg=next_update_time_ms - timestamp_ms_july_24_2024_4am)

    def test_simple_long_position_with_implicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=2.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + 10 * MS_IN_8_HOURS,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'close_ms': None,
            'return_at_close': 0.9995,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.9958850251380311,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

        self.assertAlmostEqual(position.get_carry_fee(o2.processed_ms)[0], CRYPTO_CARRY_FEE_PER_INTERVAL ** 10, 7)

    def test_simple_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + 1,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=90,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + 3 * MS_IN_8_HOURS + 1000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.assertAlmostEqual(position.get_carry_fee(o1.processed_ms)[0], 1.0)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.9995,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0985508997795737,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)
        self.assertAlmostEqual(position.get_carry_fee(o2.processed_ms)[0], CRYPTO_CARRY_FEE_PER_INTERVAL ** 3)

    def test_liquidated_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 10.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.99,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 10.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 10.0)
        self.assertEqual(position.get_cumulative_leverage(), 20.0)
        self.assertAlmostEqual(position.get_carry_fee(o2.processed_ms)[0], 1.0)

    def test_liquidated_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

    def test_liquidated_short_position_with_no_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=.1,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.LONG,
                   leverage=.1,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        # Orders post-liquidation are ignored
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 1.1)

    def test_liquidated_long_position_with_no_FLAT(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=.1,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=.1,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 10,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': None,
            'return_at_close': 0.99,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 10,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        # Orders post-liquidation are ignored
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 10,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), 10.0)
        self.assertEqual(position.get_cumulative_leverage(), 10.1)

    def test_simple_short_position_with_implicit_FLAT(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=2.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.4985,
            'current_return': 1.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

    def test_invalid_leverage_order(self):
        position_v1 = deepcopy(self.default_position)
        min_allowed_leverage_v1, max_allowed_leverage_v1 = get_position_leverage_bounds(TradePair.BTCUSD,
                                                                                        LEVERAGE_BOUNDS_V2_START_TIME_MS - 1)
        position_v2 = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.LONG,
                                        leverage=ValiConfig.ORDER_MIN_LEVERAGE * .999,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=1000,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.LONG,
                                        leverage=ValiConfig.ORDER_MAX_LEVERAGE * 1.0001,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=1000,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.SHORT,
                                        leverage=ValiConfig.ORDER_MIN_LEVERAGE * .999,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=1000,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.SHORT,
                                        leverage=ValiConfig.ORDER_MAX_LEVERAGE * 1.0001,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=1000,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.LONG,
                                        leverage=0.0,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=1000,
                                        order_uuid="1000"))
        with self.assertRaises(ValueError):
            position_v2.add_order(Order(order_type=OrderType.LONG,
                                        leverage=0.0,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.SHORT,
                                        leverage=min_allowed_leverage_v1 * .999,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS - 1,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v2.add_order(Order(order_type=OrderType.SHORT,
                                        leverage=TradePair.BTCUSD.min_leverage * .999,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.LONG,
                                        leverage=-1.0,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=1000,
                                        order_uuid="1000"))
        with self.assertRaises(ValueError):
            position_v2.add_order(Order(order_type=OrderType.LONG,
                                        leverage=-1.0,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v1.add_order(Order(order_type=OrderType.LONG,
                                        leverage=min_allowed_leverage_v1 * .999,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS - 1,
                                        order_uuid="1000"))

        with self.assertRaises(ValueError):
            position_v2.add_order(Order(order_type=OrderType.LONG,
                                        leverage=TradePair.BTCUSD.min_leverage * .999,
                                        price=100,
                                        trade_pair=TradePair.BTCUSD,
                                        processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                                        order_uuid="1000"))

    def test_invalid_prices_zero(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=0,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        with self.assertRaises(ValueError):
            position.add_order(o1)

    def test_invalid_prices_negative(self):
        with self.assertRaises(ValueError):
            o1 = Order(order_type=OrderType.LONG,  # noqa: F841
                       leverage=1.0,
                       price=-1,
                       trade_pair=TradePair.BTCUSD,
                       processed_ms=1000,
                       order_uuid="1000")

    def test_three_orders_with_longs_no_drawdown(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.1,
                   price=2000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=2000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.1,
            'initial_entry_price': 1000,
            'average_entry_price': 1090.9090909090908,
            'close_ms': None,
            'return_at_close': 1.9978,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1090.9090909090908,
            'close_ms': 5000,
            'return_at_close': 1.9978,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.1)
        self.assertEqual(position.get_cumulative_leverage(), 2.2)

    def test_two_orders_with_a_loss(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 24,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 12,
                   order_uuid="2000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': 0.9995,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.4995,
            'current_return': 0.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)
        self.assertEqual(position.get_spread_fee(), 1.0 - position.trade_pair.fees)
        self.assertEqual(position.get_carry_fee(o2.processed_ms)[0], 1.0)

    def test_three_orders_with_a_loss_and_then_a_gain(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 24,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.1,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 12,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=0.1,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 4,
                   order_uuid="5000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save_to_disk(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': 0.9995,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.1,
            'initial_entry_price': 1000,
            'average_entry_price': 954.5454545454545,
            'close_ms': None,
            'return_at_close': 0.499725,
            'current_return': 0.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 950.0,
            'close_ms': None,
            'return_at_close': 1.04937,
            'current_return': 1.05,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), 1.1)
        self.assertAlmostEquals(position.get_cumulative_leverage(), 1.2, 8)

    def test_returns_on_large_price_increase(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.1,
                   price=2000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.LONG,
                   leverage=.01,
                   price=39000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")
        o4 = Order(order_type=OrderType.LONG,
                   leverage=.01,
                   price=40000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=4000,
                   order_uuid="4000")
        o5 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=40000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")
        position = deepcopy(self.default_position)

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.add_order_to_position_and_save_to_disk(position, o2)
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.add_order_to_position_and_save_to_disk(position, o4)
        self.add_order_to_position_and_save_to_disk(position, o5)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3, o4, o5],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1776.7857142857142,
            'close_ms': 5000,
            'return_at_close': 43.7609328,
            'current_return': 43.81,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.12)
        self.assertEqual(position.get_cumulative_leverage(), 2.24)

    def test_returns_on_many_shorts(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=0.1,
                   price=900,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=.01,
                   price=800,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")
        o4 = Order(order_type=OrderType.SHORT,
                   leverage=.01,
                   price=700,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=4000,
                   order_uuid="4000")
        o5 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=600,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")
        position = deepcopy(self.default_position)

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.add_order_to_position_and_save_to_disk(position, o2)
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.add_order_to_position_and_save_to_disk(position, o4)
        self.add_order_to_position_and_save_to_disk(position, o5)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3, o4, o5],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 986.6071428571428,
            'close_ms': 5000,
            'return_at_close': 1.4313950399999997,
            'current_return': 1.4329999999999998,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 1.12)
        self.assertEqual(position.get_cumulative_leverage(), 2.24)

    def test_returns_on_alternating_long_short(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.5,
                   price=900,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=2.0,
                   price=800,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")
        o4 = Order(order_type=OrderType.LONG,
                   leverage=2.1,
                   price=700,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=4000,
                   order_uuid="4000")
        o5 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=600,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")
        position = deepcopy(self.default_position)

        self.add_order_to_position_and_save_to_disk(position, o1)
        self.add_order_to_position_and_save_to_disk(position, o2)
        self.add_order_to_position_and_save_to_disk(position, o3)
        self.add_order_to_position_and_save_to_disk(position, o4)
        self.add_order_to_position_and_save_to_disk(position, o5)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3, o4, o5],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1700.0000000000005,
            'close_ms': 5000,
            'return_at_close': 1.4364000000000001,
            'current_return': 1.44,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), 2.5)
        # -1 +.5 - 2.0 + 2.1 = 1.44 (abs 5.6) , (flat from -.4) -> 6.0
        self.assertEqual(position.get_cumulative_leverage(), 6.0)

    def test_error_adding_mismatched_trade_pair(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.EURNZD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=3.0,
                   price=500,
                   trade_pair=TradePair.CADCHF,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.FLAT,
                   leverage=5.0,
                   price=500,
                   trade_pair=TradePair.SPX,
                   processed_ms=3000,
                   order_uuid="3000")

        for order in [o1, o2, o3]:
            with self.assertRaises(ValueError):
                position.add_order(order)

    def test_two_positions_no_collisions(self):
        weekday_time_ms = FEE_V6_TIME_MS + 1000 * 60 * 60 * 24 * 3
        trade_pair1 = TradePair.SPX
        hotkey1 = self.DEFAULT_MINER_HOTKEY
        position1 = Position(
            miner_hotkey=hotkey1,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=weekday_time_ms,
            trade_pair=trade_pair1
        )
        trade_pair2 = TradePair.EURJPY
        hotkey2 = self.DEFAULT_MINER_HOTKEY + '_2'
        position2 = Position(
            miner_hotkey=hotkey2,
            position_uuid=self.DEFAULT_POSITION_UUID + '_2',
            open_ms=weekday_time_ms,
            trade_pair=trade_pair2
        )

        o1 = Order(order_type=OrderType.SHORT,
                   leverage=0.4,
                   price=1000,
                   trade_pair=trade_pair1,
                   processed_ms=weekday_time_ms,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=0.4,
                   price=500,
                   trade_pair=trade_pair2,
                   processed_ms=weekday_time_ms,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position1, o1)
        self.validate_intermediate_position_state(position1, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -0.4,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'close_ms': None,
            'return_at_close': 0.999982,
            'current_return': 1.0,
            'miner_hotkey': position1.miner_hotkey,
            'open_ms': weekday_time_ms,
            'trade_pair': trade_pair1,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.add_order_to_position_and_save_to_disk(position2, o2)
        self.validate_intermediate_position_state(position2, {
            'orders': [o2],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -0.4,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'close_ms': None,
            'return_at_close': .999986,
            'current_return': 1.0,
            'miner_hotkey': position2.miner_hotkey,
            'open_ms': weekday_time_ms,
            'trade_pair': trade_pair2,
            'position_uuid': self.DEFAULT_POSITION_UUID + '_2'
        })

        self.assertEqual(position1.max_leverage_seen(), 0.4)
        self.assertEqual(position2.max_leverage_seen(), 0.4)
        self.assertEqual(position1.get_cumulative_leverage(), 0.4)
        self.assertEqual(position2.get_cumulative_leverage(), 0.4)

        self.assertEqual(position1.get_carry_fee(o1.processed_ms + MS_IN_24_HOURS)[0],
                         INDICES_CARRY_FEE_PER_INTERVAL ** position1.max_leverage_seen())
        self.assertEqual(position2.get_carry_fee(o2.processed_ms + MS_IN_24_HOURS)[0],
                         FOREX_CARRY_FEE_PER_INTERVAL ** position2.max_leverage_seen())

    def test_transition_to_positional_leverage_v2_high_positive_leverage(self):
        position = deepcopy(self.default_position)
        live_price = 69000
        min_allowed_leverage, max_allowed_leverage = get_position_leverage_bounds(TradePair.BTCUSD,
                                                                                  LEVERAGE_BOUNDS_V2_START_TIME_MS + 1)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=max_allowed_leverage * 2,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 1,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': o1.leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 1.0)

        o3 = Order(order_type=OrderType.SHORT,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 10,
                   order_uuid="3000")

        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o3],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': o1.leverage - ValiConfig.ORDER_MIN_LEVERAGE,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_transition_to_positional_leverage_v2_small_positive_leverage(self):
        position = deepcopy(self.default_position)
        live_price = 69000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE * 2,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 1,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save_to_disk(position, o2)

        self.assertEqual(position.max_leverage_seen(), ValiConfig.ORDER_MIN_LEVERAGE * 2)
        self.assertEqual(position.get_cumulative_leverage(), ValiConfig.ORDER_MIN_LEVERAGE * 2)

        o3 = Order(order_type=OrderType.LONG,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 10,
                   order_uuid="3000")

        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o3],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': o1.leverage + o3.leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_transition_to_positional_leverage_v2_small_negative_leverage(self):
        position = deepcopy(self.default_position)
        live_price = 69000
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE * 2,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 1,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save_to_disk(position, o2)

        self.assertEqual(position.max_leverage_seen(), ValiConfig.ORDER_MIN_LEVERAGE * 2)
        self.assertEqual(position.get_cumulative_leverage(), ValiConfig.ORDER_MIN_LEVERAGE * 2)

        o3 = Order(order_type=OrderType.SHORT,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 10,
                   order_uuid="3000")

        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o3],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': o1.leverage + o3.leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_transition_to_positional_leverage_v2_high_negative_leverage(self):
        position = deepcopy(self.default_position)
        live_price = 69000
        min_allowed_leverage, max_allowed_leverage = get_position_leverage_bounds(TradePair.BTCUSD,
                                                                                  LEVERAGE_BOUNDS_V2_START_TIME_MS + 1)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=max_allowed_leverage * 2,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 1,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': o1.leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 1.0)

        o3 = Order(order_type=OrderType.LONG,
                   leverage=ValiConfig.ORDER_MIN_LEVERAGE,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 10,
                   order_uuid="3000")

        self.add_order_to_position_and_save_to_disk(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o3],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -abs(o1.leverage) + ValiConfig.ORDER_MIN_LEVERAGE,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

    def test_leverage_clamping_v1_long(self):
        position = deepcopy(self.default_position)
        live_price = 69000
        min_allowed_leverage, max_allowed_leverage = get_position_leverage_bounds(TradePair.BTCUSD,
                                                                                  LEVERAGE_BOUNDS_V2_START_TIME_MS - 1)
        self.assertEqual(min_allowed_leverage, 0.001)
        self.assertEqual(max_allowed_leverage, 20)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=max_allowed_leverage - 2,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=max_allowed_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        o2_clamped = deepcopy(o2)
        o2_clamped.leverage = 2

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2_clamped],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': max_allowed_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), 20.0)
        self.assertEqual(position.get_cumulative_leverage(), 20.0)

    def test_leverage_clamping_v2_long(self):
        position = deepcopy(self.default_position)
        live_price = 69000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage / 2,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 2000,
                   order_uuid="2000")

        o2_clamped = deepcopy(o2)
        o2_clamped.leverage = TradePair.BTCUSD.max_leverage / 2

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2_clamped],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': LEVERAGE_BOUNDS_V2_START_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage)

    def test_leverage_clamping_v1_skip_long_order(self):
        position = deepcopy(self.default_position)
        min_allowed_leverage, max_allowed_leverage = get_position_leverage_bounds(TradePair.BTCUSD,
                                                                                  LEVERAGE_BOUNDS_V2_START_TIME_MS - 1)

        live_price = 100000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=max_allowed_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': max_allowed_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': 1000,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), max_allowed_leverage)
        self.assertEqual(position.get_cumulative_leverage(), max_allowed_leverage)

    def test_leverage_clamping_v2_skip_long_order(self):
        position = deepcopy(self.default_position)
        live_price = 100000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage / 10,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 2000,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': LEVERAGE_BOUNDS_V2_START_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage)

    def test_leverage_clamping_short_v1(self):
        position = deepcopy(self.default_position)
        min_allowed_leverage, max_allowed_leverage = get_position_leverage_bounds(TradePair.BTCUSD,
                                                                                  LEVERAGE_BOUNDS_V2_START_TIME_MS - 1)
        live_price = 4444
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-10.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=-11.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        o2_clamped = deepcopy(o2)
        o2_clamped.leverage = -1.0 * (max_allowed_leverage - abs(o1.leverage))

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2_clamped],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1.0 * max_allowed_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), max_allowed_leverage)
        self.assertEqual(position.get_cumulative_leverage(), max_allowed_leverage)

    def test_leverage_clamping_short_v2(self):
        position = deepcopy(self.default_position)
        live_price = 4444
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-TradePair.BTCUSD.max_leverage * .80,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 2000,
                   order_uuid="2000")

        o2_clamped = deepcopy(o2)
        o2_clamped.leverage = -1.0 * (TradePair.BTCUSD.max_leverage - abs(o1.leverage))

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2_clamped],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })
        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage)

    def test_leverage_clamping_v2_clamps_leverage_to_small(self):
        position = deepcopy(self.default_position)
        live_price = 4444
        o1 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.min_leverage * 1.5,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=TradePair.BTCUSD.min_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save_to_disk(position, o1)
        # Ensure valueError is thrown. This position's leverage is too small to be conisdered valid.
        # Instead of clamping, this order should cause an error

        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save_to_disk(position, o2)

    def test_leverage_v1_clamping_skip_short_order(self):
        position = deepcopy(self.default_position)
        live_price = 999
        min_allowed_leverage, max_allowed_leverage = get_position_leverage_bounds(TradePair.BTCUSD,
                                                                                  LEVERAGE_BOUNDS_V2_START_TIME_MS - 1)
        self.assertEqual(min_allowed_leverage, 0.001)
        self.assertEqual(max_allowed_leverage, 20)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-max_allowed_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -max_allowed_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), max_allowed_leverage)
        self.assertEqual(position.get_cumulative_leverage(), max_allowed_leverage)

    def test_leverage_v2_clamping_skip_short_order(self):
        position = deepcopy(self.default_position)
        live_price = 999
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=TradePair.BTCUSD.max_leverage / 10,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=LEVERAGE_BOUNDS_V2_START_TIME_MS + 2000,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -TradePair.BTCUSD.max_leverage,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID
        })

        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage)

    def test_position_json(self):
        position = deepcopy(self.default_position)
        live_price = 100000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=5.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save_to_disk(position, order)

        #self.assertEqual(position_json, {})
        dict_repr = position.to_dict()  # Make sure no side effects in the recreated object...
        for x in dict_repr['orders']:
            self.assertFalse('trade_pair' in x, dict_repr)

        position_json = position.to_json_string()
        recreated_object = Position(**json.loads(position_json))
        for x in recreated_object.orders:
            self.assertTrue(hasattr(x, 'trade_pair'), recreated_object)

        recreated_object_json = json.loads(position_json)
        for x in recreated_object_json['orders']:
            self.assertFalse('trade_pair' in x, recreated_object_json)

        #print(f"position json: {position_json}")
        dict_repr = position.to_dict()  # Make sure no side effects in the recreated object...

        recreated_object = Position.parse_raw(position_json)  #Position(**json.loads(position_json))
        #print(f"recreated object str repr: {recreated_object}")
        #print("recreated object:", recreated_object)
        self.assertTrue(PositionManager.positions_are_the_same(position, recreated_object))
        for x in dict_repr['orders']:
            self.assertFalse('trade_pair' in x, dict_repr)

        for x in recreated_object.orders:
            self.assertTrue(hasattr(x, 'trade_pair'), recreated_object)


if __name__ == '__main__':
    import unittest

    unittest.main()
