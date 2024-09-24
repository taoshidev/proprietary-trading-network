# developer: trdougherty
from copy import deepcopy

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerData
from tests.shared_objects.mock_classes import (
    MockMetagraph, MockChallengePeriodManager, MockPositionManager, MockPerfLedgerManager, MockCacheController
)
from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.test_utilities import generate_ledger

from vali_config import TradePair
from vali_objects.position import Position
from vali_config import ValiConfig

import copy


class TestChallengePeriodIntegration(TestBase):

    def setUp(self):
        super().setUp()
        self.N_MINERS = 20

        # Time configurations
        self.START_TIME = 0
        self.END_TIME = ValiConfig.CHALLENGE_PERIOD_MS - 1

        # For time management
        self.CURRENTLY_IN_CHALLENGE = ValiConfig.CHALLENGE_PERIOD_MS  # Evaluation time when inside the challenge period
        self.OUTSIDE_OF_CHALLENGE = ValiConfig.CHALLENGE_PERIOD_MS + 1  # Evaluation time when the challenge period is over

        self.N_POSITIONS_BOUNDS = ValiConfig.CHALLENGE_PERIOD_MIN_POSITIONS + 1
        self.N_POSITIONS = self.N_POSITIONS_BOUNDS - 1

        self.EVEN_TIME_DISTRIBUTION = [
            int(self.START_TIME + (self.END_TIME - self.START_TIME) * i / self.N_POSITIONS_BOUNDS)
            for i
            in range(self.N_POSITIONS_BOUNDS)
        ]

        # Define miner categories
        self.SUCCESS_MINER_NAMES = [f"test_miner{i}" for i in range(1, self.N_MINERS // 2)]
        self.TESTING_MINER_NAMES = [f"test_miner{i}" for i in range(self.N_MINERS // 2, self.N_MINERS // 2 + self.N_MINERS // 4)]
        self.FAILING_MINER_NAMES = [f"test_miner{i}" for i in range(self.N_MINERS // 2 + self.N_MINERS // 4, self.N_MINERS)]

        self.MINER_NAMES = self.SUCCESS_MINER_NAMES + self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES

        # Default characteristics
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_CLOSE_MS = 2000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        # Default positions
        self.DEFAULT_POSITION = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            close_ms=self.DEFAULT_CLOSE_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            is_closed_position=True,
            return_at_close=1.00,
            orders=[Order(price=60000, processed_ms=self.START_TIME, order_uuid="initial_order",
                          trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)]

        )

        # Generate a positions list with N_POSITIONS positions
        self.DEFAULT_POSITIONS = []
        for i in range(self.N_POSITIONS):
            position = deepcopy(self.DEFAULT_POSITION)
            position.open_ms = self.EVEN_TIME_DISTRIBUTION[i]
            position.close_ms = self.EVEN_TIME_DISTRIBUTION[i + 1]
            position.is_closed_position = True
            position.return_at_close = 1.0
            position.orders[0] = Order(price=60000, processed_ms=int(position.open_ms), order_uuid="order" + str(i),
                                       trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)

            self.DEFAULT_POSITIONS.append(position)

        self.WINNING_POSITIONS = copy.deepcopy(self.DEFAULT_POSITIONS)
        for position in self.WINNING_POSITIONS:
            position.return_at_close = 1.1

        self.LOSING_POSITIONS = copy.deepcopy(self.DEFAULT_POSITIONS)
        for position in self.LOSING_POSITIONS:
            position.return_at_close = 0.9

        # Ledgers
        self.PROFITABLE_LEDGER = generate_ledger(gain=0.2, loss=-0.1, start_time=self.START_TIME, end_time=self.END_TIME)
        self.LOSING_LEDGER = generate_ledger(
            gain=0.1,
            loss=-0.2,
            mdd=1 - (ValiConfig.CHALLENGE_PERIOD_MAX_DRAWDOWN_PERCENT / 100),
            start_time=self.START_TIME,
            end_time=self.END_TIME
        )

        self.UNDETERMINED_LEDGER = generate_ledger(
            value=0.1,
            start_time=self.START_TIME,
            end_time=self.END_TIME,
        )
        self.LEDGERS = {}

        # Testing information
        self.TESTING_INFORMATION = {x: self.START_TIME for x in self.MINER_NAMES}

        # Initialize system components
        self.mock_metagraph = MockMetagraph(self.MINER_NAMES)
        self.challengeperiod_manager = MockChallengePeriodManager(self.mock_metagraph)
        self.position_manager = MockPositionManager(self.mock_metagraph)
        self.ledger_manager = MockPerfLedgerManager(self.mock_metagraph)
        self.cache_controller = MockCacheController(self.mock_metagraph)

        # Build base ledgers and positions
        self.LEDGERS = {
            miner: copy.deepcopy(self.PROFITABLE_LEDGER) for miner in self.SUCCESS_MINER_NAMES
        }
        self.LEDGERS.update({
            miner: copy.deepcopy(self.LOSING_LEDGER) for miner in self.FAILING_MINER_NAMES
        })
        self.LEDGERS.update({
            miner: copy.deepcopy(self.UNDETERMINED_LEDGER) for miner in self.TESTING_MINER_NAMES
        })
        self.ledger_manager.save_perf_ledgers_to_disk(self.LEDGERS)

        # Build base positions
        self.POSITIONS = {}
        for miner in self.SUCCESS_MINER_NAMES:
            positions = deepcopy(self.WINNING_POSITIONS)
            for position in positions:
                position.miner_hotkey = miner

            self.POSITIONS[miner] = positions

        for miner in self.FAILING_MINER_NAMES:
            positions = deepcopy(self.LOSING_POSITIONS)
            for position in positions:
                position.miner_hotkey = miner

            self.POSITIONS[miner] = positions

        for miner in self.TESTING_MINER_NAMES:
            positions = deepcopy(self.DEFAULT_POSITIONS)
            for position in positions:
                position.miner_hotkey = miner

            self.POSITIONS[miner] = positions

        for miner, positions in self.POSITIONS.items():
            for position in positions:
                position.position_uuid = f"{miner}_position_{position.open_ms}_{position.close_ms}"
                self.position_manager.save_miner_position_to_disk(position)

        # Finally update the challenge period to default state
        self.challengeperiod_manager.init_cache_files()
        self.challengeperiod_manager._clear_eliminations_in_memory_and_disk()

        # Add all the miners with a start time of 0
        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            self.MINER_NAMES,
            eliminations=[],
            current_time=self.START_TIME
        )

    def tearDown(self):
        super().tearDown()
        # Cleanup and setup
        self.position_manager.clear_all_miner_positions_from_disk()
        self.ledger_manager.clear_perf_ledgers_from_disk()

        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.challengeperiod_manager._clear_eliminations_in_memory_and_disk()

    def test_refresh_populations(self):
        self.challengeperiod_manager.refresh(current_time=self.CURRENTLY_IN_CHALLENGE)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_testing), len(self.TESTING_MINER_NAMES))
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_success), len(self.SUCCESS_MINER_NAMES))
        self.assertEqual(len(self.challengeperiod_manager.eliminations), len(self.FAILING_MINER_NAMES))

    def test_full_refresh(self):
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_testing), len(self.MINER_NAMES))
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_success), 0)
        self.assertEqual(len(self.challengeperiod_manager.eliminations), 0)

        inspection_hotkeys = self.challengeperiod_manager.challengeperiod_testing

        for hotkey, inspection_time in inspection_hotkeys.items():

            time_criteria = self.CURRENTLY_IN_CHALLENGE - inspection_time <= ValiConfig.CHALLENGE_PERIOD_MS
            self.assertTrue(time_criteria, f"Time criteria failed for {hotkey}")

        self.challengeperiod_manager.refresh(current_time=self.CURRENTLY_IN_CHALLENGE)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        self.challengeperiod_manager.write_eliminations_to_disk(self.challengeperiod_manager.eliminations)

        del self.challengeperiod_manager.eliminations
        self.assertTrue(len(self.cache_controller.eliminations) == 0)
        self.cache_controller._refresh_eliminations_in_memory()

        elimination_hotkeys = [x['hotkey'] for x in self.cache_controller.eliminations]

        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, elimination_hotkeys)

        for miner in self.SUCCESS_MINER_NAMES:
            self.assertNotIn(miner, elimination_hotkeys)

        for miner in self.TESTING_MINER_NAMES:
            self.assertNotIn(miner, elimination_hotkeys)

    def test_failing_mechanics(self):
        # Add all the challenge period miners
        self.assertListEqual(sorted(self.MINER_NAMES), sorted(self.mock_metagraph.hotkeys))
        self.assertListEqual(sorted(self.MINER_NAMES), sorted(list(self.challengeperiod_manager.challengeperiod_testing.keys())))
        # Let's check the initial state of the challenge period
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_success), 0)
        self.assertEqual(len(self.challengeperiod_manager.eliminations), 0)
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_testing), len(self.MINER_NAMES))

        eliminations = self.challengeperiod_manager.get_filtered_eliminations_from_disk()
        self.assertEqual(len(eliminations), 0)

        self.challengeperiod_manager._refresh_challengeperiod_in_memory_and_disk(eliminations=eliminations)
        self.assertEqual(len(self.challengeperiod_manager.eliminations), 0)

        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_testing), len(self.MINER_NAMES))

        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.challengeperiod_manager.metagraph.hotkeys,
            eliminations=self.challengeperiod_manager.eliminations,
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_testing), len(self.MINER_NAMES))

        self.challengeperiod_manager.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)
        elimination_keys = [x['hotkey'] for x in self.challengeperiod_manager.eliminations]

        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, self.mock_metagraph.hotkeys)
            self.assertIn(miner, elimination_keys)

        eliminations = self.challengeperiod_manager.get_eliminated_hotkeys()

        self.assertListEqual(sorted(list(eliminations)), sorted(elimination_keys))

    def test_single_position_no_ledger(self):
        # Cleanup all positions first
        self.position_manager.clear_all_miner_positions_from_disk()
        self.ledger_manager.clear_perf_ledgers_from_disk()

        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.challengeperiod_manager._clear_eliminations_in_memory_and_disk()

        self.challengeperiod_manager.challengeperiod_testing = {}
        self.challengeperiod_manager.challengeperiod_success = {}

        position = deepcopy(self.DEFAULT_POSITION)
        position.is_closed_position = False
        position.close_ms = None

        self.position_manager.save_miner_position_to_disk(position)
        self.challengeperiod_manager.challengeperiod_testing = {self.DEFAULT_MINER_HOTKEY: self.DEFAULT_OPEN_MS}
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        # Now loading the data
        positions = self.position_manager.get_all_miner_positions_by_hotkey(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        ledgers = self.ledger_manager.load_perf_ledgers_from_disk()

        # First check that there is nothing on the miner
        self.assertEqual(ledgers.get(self.DEFAULT_MINER_HOTKEY, PerfLedgerData().cps), PerfLedgerData().cps)
        self.assertEqual(ledgers.get(self.DEFAULT_MINER_HOTKEY, len(PerfLedgerData().cps)), 0)

        # Check the failing criteria initially
        failing_criteria = self.challengeperiod_manager.screen_failing_criteria(
            ledger_element=ledgers.get(self.DEFAULT_MINER_HOTKEY)
        )

        self.assertFalse(failing_criteria)

        # Now check the inspect to see where the key went
        challenge_success, challenge_eliminations = self.challengeperiod_manager.inspect(
            positions=positions,
            ledger=ledgers,
        )

        # There should be no promotion or demotion
        self.assertListEqual(challenge_success, [])
        self.assertListEqual(challenge_eliminations, [])

    def test_failing_miner_screen(self):
        # Add all the challenge period miners
        self.challengeperiod_manager.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)

        ledger = self.ledger_manager.load_perf_ledgers_from_disk()
        positions = self.position_manager.get_all_miner_positions_by_hotkey(
            self.MINER_NAMES,
            self.challengeperiod_manager.eliminations
        )

        # Should accurately fail all the miners identified as failing
        for miner in self.FAILING_MINER_NAMES:
            passing_criteria = self.challengeperiod_manager.screen_passing_criteria(
                position_elements=positions.get(miner),
                ledger_element=ledger.get(miner),
                current_time=self.OUTSIDE_OF_CHALLENGE
            )

            # self.assertListEqual([miner], self.challengeperiod_manager.eliminations)
            self.assertFalse(passing_criteria)

    def test_promote_testing_miner(self):
        # Add all the challenge period miners
        self.challengeperiod_manager.refresh(current_time=self.CURRENTLY_IN_CHALLENGE)

        testing_hotkeys = list(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        self.assertIn(self.TESTING_MINER_NAMES[0], testing_hotkeys)
        self.assertNotIn(self.TESTING_MINER_NAMES[0], success_hotkeys)

        self.challengeperiod_manager._promote_challengeperiod_in_memory(
            hotkeys=[self.TESTING_MINER_NAMES[0]],
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        testing_hotkeys = list(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        self.assertNotIn(self.TESTING_MINER_NAMES[0], testing_hotkeys)
        self.assertIn(self.TESTING_MINER_NAMES[0], success_hotkeys)

        # Check that the timestamp of the success is the current time of evaluation
        self.assertEqual(
            self.challengeperiod_manager.challengeperiod_success[self.TESTING_MINER_NAMES[0]],
            self.CURRENTLY_IN_CHALLENGE
        )

    def test_refresh_elimination_disk(self):
        # At this point, all the miners should be in testing
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.MINER_NAMES))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)

        # Check one of the failing miners, to see if they are screened
        failing_miner = self.FAILING_MINER_NAMES[0]
        failing_screen = self.challengeperiod_manager.screen_failing_criteria(
            ledger_element=self.LEDGERS[failing_miner]
        )

        self.assertEqual(failing_screen, True)

        # Now inspect all the hotkeys
        challenge_success, challenge_eliminations = self.challengeperiod_manager.inspect(
            positions=self.POSITIONS,
            ledger=self.LEDGERS,
            inspection_hotkeys=self.challengeperiod_manager.challengeperiod_testing,
            current_time=self.OUTSIDE_OF_CHALLENGE
        )

        # self.assertListEqual(challenge_success, self.SUCCESS_MINER_NAMES)
        self.assertListEqual(sorted(challenge_eliminations), sorted(self.FAILING_MINER_NAMES + self.TESTING_MINER_NAMES))

        self.challengeperiod_manager.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)
        self.assertTrue(len(self.cache_controller.eliminations) == 0)
        self.cache_controller._refresh_eliminations_in_memory()

        self.assertTrue(len(self.cache_controller.eliminations) > 0)
        cached_eliminations = [x['hotkey'] for x in self.cache_controller.eliminations]

        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, cached_eliminations)

    def test_no_positions_miner_filtered(self):
        self.challengeperiod_manager.challengeperiod_testing = {}
        self.challengeperiod_manager.challengeperiod_success = {}
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)

        # Now going to remove the positions of the miners
        miners_without_positions = self.TESTING_MINER_NAMES[:2]

        # Redeploy the positions
        for miner, positions in self.POSITIONS.items():
            if miner in miners_without_positions:
                for position in positions:
                    self.position_manager.delete_position_from_disk(position)

        self.challengeperiod_manager.refresh(current_time=self.CURRENTLY_IN_CHALLENGE)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        for miner in miners_without_positions:
            self.assertIn(miner, self.mock_metagraph.hotkeys)
            self.assertNotIn(miner, self.challengeperiod_manager.challengeperiod_testing)
            self.assertNotIn(miner, self.challengeperiod_manager.challengeperiod_success)

    def test_disjoint_testing_success(self):
        self.challengeperiod_manager.refresh(current_time=self.CURRENTLY_IN_CHALLENGE)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        testing_set = set(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_set = set(self.challengeperiod_manager.challengeperiod_success.keys())

        self.assertTrue(testing_set.isdisjoint(success_set))

    def test_addition(self):
        self.challengeperiod_manager.refresh(current_time=self.CURRENTLY_IN_CHALLENGE)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.MINER_NAMES,
            eliminations=[],
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        testing_set = set(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_set = set(self.challengeperiod_manager.challengeperiod_success.keys())

        self.assertTrue(testing_set.isdisjoint(success_set))

    def test_add_miner_no_positions(self):
        self.challengeperiod_manager.challengeperiod_testing = {}
        self.challengeperiod_manager.challengeperiod_success = {}

        # Check if it still stores the miners with no perf ledger
        self.ledger_manager.clear_perf_ledgers_from_disk()

        new_miners = ["miner_no_positions1", "miner_no_positions2"]

        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()
        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=new_miners,
            eliminations=[],
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == 0)
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)

        # Now add perf ledgers to check that adding miners without positions still doesn't add them
        self.ledger_manager.save_perf_ledgers_to_disk(self.LEDGERS)
        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=new_miners,
            eliminations=[],
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == 0)
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)

        all_miners_positions = self.challengeperiod_manager.get_all_miner_positions_by_hotkey(self.MINER_NAMES)
        self.assertListEqual(list(all_miners_positions.keys()), self.MINER_NAMES)

        miners_with_one_position = self.challengeperiod_manager.get_all_miner_hotkeys_with_at_least_one_position()
        miners_with_one_position_sorted = sorted(list(miners_with_one_position))

        self.assertListEqual(miners_with_one_position_sorted, sorted(self.MINER_NAMES))
        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.MINER_NAMES,
            eliminations=[],
            current_time=self.CURRENTLY_IN_CHALLENGE
        )

        # Refresh the challenge period
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        # All the miners should be passed to testing now
        self.assertListEqual(
            sorted(list(self.challengeperiod_manager.challengeperiod_testing.keys())),
            sorted(self.MINER_NAMES)
        )

        self.assertListEqual(
            sorted(list(self.challengeperiod_manager.challengeperiod_testing.values())),
            [self.CURRENTLY_IN_CHALLENGE] * len(self.MINER_NAMES)
        )

        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_success), 0)

    def test_refresh_all_eliminated(self):
        for miner in self.MINER_NAMES:
            self.challengeperiod_manager.append_elimination_row(miner, -1, "FAILED_CHALLENGE_PERIOD")

        self.challengeperiod_manager._write_eliminations_from_memory_to_disk()
        self.challengeperiod_manager.eliminations = []

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.MINER_NAMES))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)

        self.challengeperiod_manager.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == 0)
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == len(self.MINER_NAMES))

    def test_clear_challengeperiod_in_memory_and_disk(self):
        self.challengeperiod_manager.challengeperiod_testing = {
            "test_miner1": 1, "test_miner2": 1, "test_miner3": 1, "test_miner4": 1
        }
        self.challengeperiod_manager.challengeperiod_success = {
            "test_miner5": 1, "test_miner6": 1, "test_miner7": 1, "test_miner8": 1
        }

        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()

        testing_keys = list(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_keys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        self.assertEqual(testing_keys, [])
        self.assertEqual(success_keys, [])
