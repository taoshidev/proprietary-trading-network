from tests.shared_objects.mock_classes import MockMetagraph, MockPositionManager
from tests.shared_objects.test_utilities import generate_losing_ledger, generate_winning_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

class TestEliminationManager(TestBase):
    def setUp(self):
        super().setUp()

        self.MDD_MINER = "miner_1"
        self.REGULAR_MINER = "miner_2"

        # Initialize system components
        self.mock_metagraph = MockMetagraph([self.MDD_MINER])

        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        self.ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)
        self.position_manager = MockPositionManager(self.mock_metagraph,
                                                    perf_ledger_manager=self.ledger_manager,
                                                    elimination_manager=self.elimination_manager)
        self.position_manager.perf_ledger_manager = self.ledger_manager
        self.elimination_manager.position_manager = self.position_manager

        self.challengeperiod_manager = ChallengePeriodManager(self.mock_metagraph,
                                                              position_manager=self.position_manager,
                                                              perf_ledger_manager=self.ledger_manager,
                                                              running_unit_tests=True)
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager

        self.position_manager.clear_all_miner_positions()
        self.position_locks = PositionLocks()

        self.LEDGERS = {}
        self.LEDGERS[self.MDD_MINER] = generate_losing_ledger(0, ValiConfig.CHALLENGE_PERIOD_MS)
        self.LEDGERS[self.REGULAR_MINER] = generate_winning_ledger(0, ValiConfig.CHALLENGE_PERIOD_MS)
        self.ledger_manager.save_perf_ledgers(self.LEDGERS)

        self.challengeperiod_manager.challengeperiod_success[self.MDD_MINER] = 0
        self.challengeperiod_manager.challengeperiod_success[self.REGULAR_MINER] = 0

    def tearDown(self):
        super().tearDown()
        # Cleanup and setup
        self.position_manager.clear_all_miner_positions()
        self.ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

    def test_elimination_for_mdd(self):
        # Neither miner has been eliminated
        self.assertEqual(len(self.challengeperiod_manager.challengeperiod_success), 2)

        self.elimination_manager.process_eliminations(self.position_locks)

        eliminations = self.elimination_manager.get_eliminations_from_disk()
        self.assertEqual(len(eliminations), 1)
        for elimination in eliminations:
            self.assertEqual(elimination["hotkey"], self.MDD_MINER)
            self.assertEqual(elimination["reason"], "MAX_TOTAL_DRAWDOWN")

        #TODO Does a miner have to be popped from challenge period?
        #self.assertEqual(len(self.challengeperiod_manager.challengeperiod_success), 1)


