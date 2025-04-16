from shared_objects.cache_controller import CacheController
from tests.shared_objects.mock_classes import MockMetagraph, MockPositionManager
from tests.shared_objects.test_utilities import generate_losing_ledger, generate_winning_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig, TradePair
from vali_objects.utils.elimination_manager import EliminationManager, EliminationReason
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

class TestEliminationManager(TestBase):
    def setUp(self):
        super().setUp()

        self.MDD_MINER = "miner_mdd"
        self.REGULAR_MINER = "miner_regular"

        # Initialize system components
        self.mock_metagraph = MockMetagraph([self.MDD_MINER, self.REGULAR_MINER])

        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        self.ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)
        self.position_manager = MockPositionManager(self.mock_metagraph,
                                                    perf_ledger_manager=self.ledger_manager,
                                                    elimination_manager=self.elimination_manager)
        self.position_manager.clear_all_miner_positions()
        for hk in self.mock_metagraph.hotkeys:
            mock_position = Position(
                miner_hotkey=hk,
                position_uuid=hk,
                open_ms=1,
                close_ms=2,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=False,
                return_at_close=1.00,
                orders=[Order(price=60000, processed_ms=1, order_uuid="initial_order",
                              trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)]

            )
            self.position_manager.save_miner_position(mock_position)

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        files = CacheController.get_directory_names(all_miners_dir)
        assert len(files) == len(self.mock_metagraph.hotkeys), (all_miners_dir, files, self.mock_metagraph.hotkeys)

        self.position_manager.perf_ledger_manager = self.ledger_manager
        self.elimination_manager.position_manager = self.position_manager

        self.challengeperiod_manager = ChallengePeriodManager(self.mock_metagraph,
                                                              position_manager=self.position_manager,
                                                              perf_ledger_manager=self.ledger_manager,
                                                              running_unit_tests=True)
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager

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
            self.assertEqual(elimination["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

        # test_zombie_eliminations
        self.mock_metagraph.hotkeys = []
        self.elimination_manager.process_eliminations(self.position_locks)
        eliminations = self.elimination_manager.get_eliminations_from_disk()
        for elimination in eliminations:
            if elimination["hotkey"] == self.MDD_MINER:
                assert elimination["reason"] == EliminationReason.MAX_TOTAL_DRAWDOWN.value, eliminations
            elif elimination["hotkey"] == self.REGULAR_MINER:
                assert elimination["reason"] == EliminationReason.ZOMBIE.value, eliminations
            else:
                raise Exception(f"Unexpected hotkey in eliminations: {elimination['hotkey']}")



