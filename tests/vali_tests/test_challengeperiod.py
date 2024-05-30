# developer: trdougherty
# Copyright © 2024 Taoshi Inc
import functools
import random
from copy import deepcopy

from tests.shared_objects.mock_classes import MockMetagraph, MockChallengePeriodManager, MockPositionManager, MockPerfLedgerManager, MockCacheController
from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.test_utilities import generate_ledger

from vali_config import TradePair
from vali_objects.position import Position
from vali_config import ValiConfig

class TestChallengePriod(TestBase):

    def setUp(self):
        super().setUp()
        self.n_miners = 20
        self.miner_success_names = [ f"test_miner{i}" for i in range(1, self.n_miners//2) ]
        self.miner_testing_names = [ f"test_miner{i}" for i in range(self.n_miners//2, self.n_miners) ]

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

        self.miner_nopositions = [ "test_miner4", "test_miner12" ] # one success, one testing

        self.miner_positions = [ x for x in self.miner_success_names if x not in self.miner_nopositions ]
        self.miner_positions_success = [ x for x in self.miner_success_names if x in self.miner_positions ]
        self.miner_positions_testing = [ x for x in self.miner_testing_names if x in self.miner_positions ]

        self.miner_names = self.miner_success_names + self.miner_testing_names

        self.mock_metagraph = MockMetagraph(self.miner_names)
        self.challengeperiod_manager = MockChallengePeriodManager(metagraph=self.mock_metagraph)
        self.position_manager = MockPositionManager(metagraph=self.mock_metagraph)
        self.ledger_manager = MockPerfLedgerManager(metagraph=self.mock_metagraph)
        self.cache_controller = MockCacheController(metagraph=self.mock_metagraph)

        self.small_gain = 1e-3
        self.small_loss = -1e-3 + 1e-5

        self.ledgers = {}
        self.failing_miners = []
        self.default_ledger_checkpoints = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_VOLUME_CHECKPOINTS

        self.start_time = 0
        self.end_time = 10 * 24 * 60 * 60 * 1000 # 5 days
        self.midpoint_time = self.start_time + (self.end_time - self.start_time) // 2
        self.midpoint_time_end = self.midpoint_time + (self.end_time - self.midpoint_time) // 2
        self.current_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS
        self.elimination_time = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS + 1

        self.challengeperiod_manager.init_cache_files()
        self.position_manager.clear_all_miner_positions_from_disk()
        self.challengeperiod_manager.clear_eliminations_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()

        # clear the ledger
        MockPerfLedgerManager.save_perf_ledgers_to_disk({})

    def tearDown(self) -> None:
        self.challengeperiod_manager.clear_eliminations_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        MockPerfLedgerManager.save_perf_ledgers_to_disk({})

        del self.challengeperiod_manager
        del self.ledger_manager
        del self.mock_metagraph
        del self.miner_names
        del self.ledgers
        return super().tearDown()
    
    def generate_ledgers(self):
        self.position_manager.clear_all_miner_positions_from_disk()
        self.challengeperiod_manager.clear_challengeperiod_from_disk()
        self.challengeperiod_manager.clear_eliminations_from_disk()

        self.ledgers = {}
        for miner in self.miner_success_names:
            self.ledgers[miner] = generate_ledger(nterms=self.default_ledger_checkpoints, value=0.01, start_time=self.start_time, end_time=self.end_time, gain=0.2, loss=-0.1)

        for miner in self.miner_testing_names:
            ## these positions should still be within the challenge period time window, they opened a bit later than the rest
            self.ledgers[miner] = generate_ledger(nterms=self.default_ledger_checkpoints, value=0.01, start_time=self.midpoint_time, end_time=self.midpoint_time_end, gain=self.small_gain, loss=self.small_loss) # not enough time to meet criteria, or to pass

        # All of these miners should be subject to elimination, as the time is expired
        self.ledgers["test_miner1"] = generate_ledger(nterms=self.default_ledger_checkpoints, value=0.01, start_time=self.start_time, end_time=self.end_time, gain=2.2, loss=-2.1, mdd=0.8) # losing due to high mdd, eliminated
        self.ledgers["test_miner2"] = generate_ledger(nterms=self.default_ledger_checkpoints, value=0.01, start_time=self.start_time, end_time=self.end_time, gain=0.1, loss=-0.3) # losing due to high omega, eliminated
        self.ledgers["test_miner3"] = generate_ledger(nterms=self.default_ledger_checkpoints, value=0.01, start_time=self.start_time, end_time=self.end_time, gain=0.5, loss=-0.499) # losing due to not enough returns

        copied_ledger = deepcopy(self.ledgers["test_miner4"])
        copied_ledger.cps[-2].mdd = 0.8 # just one element which is failing
        self.ledgers["test_miner4"] = copied_ledger
        # self.ledgers["test_miner4"] = generate_ledger(nterms=self.default_ledger_checkpoints, value=0.01, start_time=self.start_time, end_time=self.end_time, gain=0.5, loss=-0.4, open_ms=30 * 1000) # losing due to time duration condition

        self.failing_miners = ["test_miner1", "test_miner2", "test_miner3", "test_miner4"]
        MockPerfLedgerManager.save_perf_ledgers_to_disk(self.ledgers)

        self.challengeperiod_manager.challengeperiod_success = {}

        full_miner_starttimes = { minername: self.start_time for minername in self.miner_names }
        miner_partial_starttimes = { minername: self.midpoint_time for minername in self.miner_testing_names }

        self.challengeperiod_manager.challengeperiod_testing = { **full_miner_starttimes, **miner_partial_starttimes }

        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

    def generate_positions(self):
        ## now create positions for all of the miners in the ledger
        num_positions = 100
        open_time_start = 1000
        open_time_end = 2000

        for miner in self.miner_names:
            if miner in self.miner_nopositions:
                continue
            # Generate and save positions
            for i in range(num_positions):
                open_ms = random.randint(open_time_start, open_time_end)
                close_ms = open_ms + random.randint(1, 1000)
                position = deepcopy(self.default_position)
                position.miner_hotkey = miner
                # Get a random trade pair
                position.trade_pair = random.choice(list(TradePair))
                position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}"
                position.open_ms = open_ms
                if close_ms:
                    position.close_out_position(close_ms)
                self.position_manager.save_miner_position_to_disk(position)
    
    def setup_ledger(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.generate_ledgers()
            self.generate_positions()
            result = func(self, *args, **kwargs)
            # Clear ledger after creation
            MockPerfLedgerManager.save_perf_ledgers_to_disk({})
            self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
            self.position_manager.clear_all_miner_positions_from_disk()
            return result
        return wrapper
    
    @setup_ledger
    def test_refresh(self):
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_testing), len(self.miner_names))
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_success), 0)

        self.challengeperiod_manager.refresh(current_time=self.current_time)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        elimination_hotkeys = [ x['hotkey'] for x in self.challengeperiod_manager.eliminations ]
        testing_hotkeys = list(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        for miner in self.miner_testing_names:
            self.assertIn(miner, testing_hotkeys)

        for miner in self.miner_success_names:
            if miner not in self.failing_miners:
                self.assertIn(miner, success_hotkeys)

        for miner in self.failing_miners:
            self.assertNotIn(miner, success_hotkeys)
            self.assertIn(miner, elimination_hotkeys)

        self.challengeperiod_manager.write_eliminations_to_disk(self.challengeperiod_manager.eliminations)

        ## get challengeperiod eliminations from disk and save to memory
        del self.challengeperiod_manager.eliminations
        
        self.assertTrue(len(self.cache_controller.eliminations) == 0)
        self.cache_controller._refresh_eliminations_in_memory()

        cache_elimination_hotkeys = [ x['hotkey'] for x in self.cache_controller.eliminations ]
        for miner in self.failing_miners:
            self.assertIn(miner, cache_elimination_hotkeys)

        self.cache_controller.eliminations = []
        self.cache_controller.clear_eliminations_from_disk()

    @setup_ledger
    def test_promote_testing_miner(self):
        ## test that the miners are promoted to success - writes to disk
        testing_hotkeys = list(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        ## miner_testing_names[0] should be promoted
        self.assertIn(self.miner_testing_names[0], testing_hotkeys)
        self.assertNotIn(self.miner_testing_names[0], success_hotkeys)

        self.challengeperiod_manager._promote_challengeperiod_in_memory(
            hotkeys=[ self.miner_testing_names[0] ],
            current_time=self.current_time
        )

        testing_hotkeys = list(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        self.assertNotIn(self.miner_testing_names[0], testing_hotkeys)
        self.assertIn(self.miner_testing_names[0], success_hotkeys)

        ## check that the time is stored correctly
        self.assertEqual(self.challengeperiod_manager.challengeperiod_success[self.miner_testing_names[0]], self.current_time)

    @setup_ledger
    def test_refresh_populations(self):
        ## first refresh the challengeperiod - all of the successful miners were eliminated for some reason
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)
        
        self.challengeperiod_manager.refresh(current_time=self.current_time)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        ## Check that we still don't have successful miners, as they were all in the eliminations
        self.assertFalse(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertFalse(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertFalse(len(self.challengeperiod_manager.eliminations) == 0)

    @setup_ledger
    def test_refresh_elimination_disk(self):
        ## first refresh the challengeperiod - all of the successful miners were eliminated for some reason
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)
        
        # this should write to disk
        self.challengeperiod_manager.refresh(current_time=self.current_time)

        ## Check using the cache controller
        self.assertTrue(len(self.cache_controller.eliminations) == 0)
        self.cache_controller._refresh_eliminations_in_memory()

        self.assertTrue(len(self.cache_controller.eliminations) > 0)
        cached_eliminations = [ x['hotkey'] for x in self.cache_controller.eliminations ]

        for miner in self.failing_miners:
            self.assertIn(miner, cached_eliminations)

    @setup_ledger
    def test_nopositions_miner_filtered(self):
        ## clear the challengeperiod in memory and disk
        self.challengeperiod_manager.challengeperiod_testing = {}
        self.challengeperiod_manager.challengeperiod_success = {}
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)

        # this should write to disk
        self.challengeperiod_manager.refresh(current_time=self.current_time)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        ## Refresh should add the miners to testing and success dictionaries, but only if there are positions on record for them
        for miner in self.miner_nopositions:
            self.assertIn(miner, self.mock_metagraph.hotkeys) # check that it's in the hotkeys
            self.assertNotIn(miner, self.challengeperiod_manager.challengeperiod_testing)
            self.assertNotIn(miner, self.challengeperiod_manager.challengeperiod_success)

    @setup_ledger
    def test_disjoint_testing_success(self):
        ## first refresh the challengeperiod - all of the successful miners were eliminated for some reason
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)
        
        # this should write to disk
        self.challengeperiod_manager.refresh(current_time=self.current_time)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        ## Check using the cache controller
        testing_set = set(list(self.challengeperiod_manager.challengeperiod_testing.keys()))
        success_set = set(list(self.challengeperiod_manager.challengeperiod_success.keys()))

        self.assertTrue(testing_set.isdisjoint(success_set))

    @setup_ledger
    def test_addition(self):
        ## first refresh the challengeperiod - all of the successful miners were eliminated for some reason
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)
        
        # this should write to disk
        self.challengeperiod_manager.refresh(current_time=self.current_time)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(new_hotkeys=self.miner_names, eliminations=[], current_time=self.current_time)

        ## Check using the cache controller
        testing_set = set(list(self.challengeperiod_manager.challengeperiod_testing.keys()))
        success_set = set(list(self.challengeperiod_manager.challengeperiod_success.keys()))

        self.assertTrue(testing_set.isdisjoint(success_set))

    @setup_ledger
    def test_add_miner(self):
        ## first refresh the challengeperiod - all of the successful miners were eliminated for some reason
        self.challengeperiod_manager.challengeperiod_testing = {}
        self.challengeperiod_manager.challengeperiod_success = {}
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(new_hotkeys=self.miner_nopositions, eliminations=[], current_time=self.current_time)

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == 0) # no additions should be made
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0) # no additions should be made

        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(new_hotkeys=self.miner_positions, eliminations=[], current_time=self.current_time)

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_positions_testing)) # all of the miners with positions on file should be added to testing
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0) # no additions should be made to success yet

    @setup_ledger
    def test_refresh_all_eliminated(self):
        ## first refresh the challengeperiod - all of the successful miners were eliminated for some reason
        # eliminations = [ { "hotkey": x, "elimination_time": self.current_time, "reason": "FAILED_CHALLENGE_PERIOD" } for x in self.miner_names ]

        for miner in self.miner_names:
            self.challengeperiod_manager.append_elimination_row(miner, -1, "FAILED_CHALLENGE_PERIOD")

        self.challengeperiod_manager._write_eliminations_from_memory_to_disk()
        self.challengeperiod_manager.eliminations = [] # clear the eliminations from memory, to be sure they're read from disk

        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)
        
        ## we have added all of the miners to elimination
        self.challengeperiod_manager.refresh(current_time=self.current_time)
        self.challengeperiod_manager._refresh_challengeperiod_in_memory()

        ## Check that we still don't have successful miners, as they were all in the eliminations
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == 0) # all of the miners were eliminated, none should be in testing
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0) # none of the miners passed
        self.assertFalse(len(self.challengeperiod_manager.eliminations) == 0) # all of the miners were eliminated
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == len(self.miner_names))

    def test_clear_challengeperiod_in_memory_and_disk(self):
        ## build out a sample for testing
        self.assertEqual(self.challengeperiod_manager.challengeperiod_testing, {})
        self.challengeperiod_manager.challengeperiod_testing = { "test_miner1": 1, "test_miner2": 1, "test_miner3": 1, "test_miner4": 1 }
        self.challengeperiod_manager.challengeperiod_success = { "test_miner1": 1, "test_miner2": 1, "test_miner3": 1, "test_miner4": 1 }
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        ## now clear the challengeperiod in memory and disk
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        testing_keys = list(self.challengeperiod_manager.challengeperiod_testing.keys())
        success_keys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        ## check that the clear function worked
        self.assertEqual(testing_keys, [])
        self.assertEqual(success_keys, [])

        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()

    @setup_ledger
    def test_failing_miner_criteria_with_drawdown(self):
        """
        Test that miners fail the challenge period due to drawdown criteria.
        """
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)

        failing_miner_times = { miner: self.challengeperiod_manager.challengeperiod_testing[miner] for miner in self.failing_miners }

        self.challengeperiod_manager.refresh(current_time=self.current_time)
        elimination_hotkeys = [x['hotkey'] for x in self.challengeperiod_manager.eliminations]

        # check that all eliminations in memory are correct
        for miner in self.failing_miners:
            passing_criteria = self.challengeperiod_manager.screen_ledger(ledger_element=self.ledgers[miner])
            drawdown_criteria = self.challengeperiod_manager.screen_ledger_drawdown(ledger_element=self.ledgers[miner])
            time_criteria = self.current_time - failing_miner_times[miner] < ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS
            self.assertIn(miner, elimination_hotkeys)
            self.assertTrue(not drawdown_criteria or not time_criteria)

    @setup_ledger
    def test_passing_miner_criteria_with_drawdown(self):
        """
        Test that miners pass the challenge period meeting both passing and drawdown criteria.
        """
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)

        self.challengeperiod_manager.refresh(current_time=self.current_time)
        passing_hotkeys = list(self.challengeperiod_manager.challengeperiod_success.keys())

        # check that all passing miners are correct
        for miner in passing_hotkeys:
            passing_criteria = self.challengeperiod_manager.screen_ledger(ledger_element=self.ledgers[miner])
            drawdown_criteria = self.challengeperiod_manager.screen_ledger_drawdown(ledger_element=self.ledgers[miner])
            self.assertIn(miner, passing_hotkeys)
            self.assertTrue(passing_criteria)
            self.assertTrue(drawdown_criteria)

    @setup_ledger
    def test_miner_passing_with_all_criteria(self):
        """
        Test that miners pass the challenge period meeting all criteria.
        """
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_testing) == len(self.miner_names))
        self.assertTrue(len(self.challengeperiod_manager.challengeperiod_success) == 0)
        self.assertTrue(len(self.challengeperiod_manager.eliminations) == 0)

        self.challengeperiod_manager.refresh(current_time=self.current_time)

        for miner, inspection_time in self.challengeperiod_manager.challengeperiod_testing.items():
            passing_criteria = self.challengeperiod_manager.screen_ledger(ledger_element=self.ledgers[miner])
            drawdown_criteria = self.challengeperiod_manager.screen_ledger_drawdown(ledger_element=self.ledgers[miner])

            if miner in self.failing_miners:
                self.assertIn(miner, self.challengeperiod_manager.eliminations)
                self.assertTrue(not passing_criteria or not drawdown_criteria)
            else:
                self.assertTrue(drawdown_criteria)
