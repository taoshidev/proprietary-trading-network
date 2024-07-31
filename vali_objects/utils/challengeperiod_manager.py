# developer: trdougherty

import numpy as np
import time
from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.scoring import Scoring, ScoringUnit
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfLedger, PerfLedgerData
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.position import Position


class ChallengePeriodManager(CacheController):
    def __init__(self, config, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.perf_manager = PerfLedgerManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)

    def refresh(self, current_time: int = None):
        if not self.refresh_allowed(ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS):
            time.sleep(1)
            return

        # The refresh should just read the current eliminations
        self.eliminations = self.get_filtered_eliminations_from_disk()

        # Collect challenge period and update with new eliminations criteria
        self._refresh_challengeperiod_in_memory_and_disk(eliminations=self.eliminations)

        # challenge period adds to testing if not in eliminated, already in the challenge period, or in the new eliminations list from disk
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph.hotkeys,
            eliminations=self.eliminations,
            current_time=current_time
        )

        ledger = self.perf_manager.load_perf_ledgers_from_disk()
        positions = self.position_manager.get_all_miner_positions_by_hotkey(
            self.metagraph.hotkeys,
            sort_positions=True
        )

        challengeperiod_success, challengeperiod_eliminations = self.inspect(
            positions=positions,
            ledger=ledger,
            inspection_hotkeys=self.challengeperiod_testing,
            current_time=current_time
        )

        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(hotkeys=challengeperiod_success, current_time=current_time)
        self._demote_challengeperiod_in_memory(hotkeys=challengeperiod_eliminations)

        # Now sync challenge period with the disk
        self._write_challengeperiod_from_memory_to_disk()
        self._write_eliminations_from_memory_to_disk()

    def inspect(
        self,
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedgerData],
        inspection_hotkeys: dict[str, int] = None,
        current_time: int = None,
        eliminations: list[str] = None,
        log: bool = False,
    ):
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period. Does not modify the challenge period in memory.
        """
        if inspection_hotkeys is None:
            return [], []  # no hotkeys to inspect

        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        if eliminations is None:
            eliminations = self.eliminations

        passing_miners = []
        failing_miners = []
        for hotkey, inspection_time in inspection_hotkeys.items():
            # Check the criteria for passing the challenge period
            if hotkey not in ledger:
                passing_criteria = False
            else:
                if log:
                    print(f"Inspecting hotkey: {hotkey}")
                passing_criteria = ChallengePeriodManager.screen(
                    position_elements=positions.get(hotkey, []),
                    ledger_element=ledger[hotkey],
                    current_time=current_time,
                    log=log
                )

            time_criteria = current_time - inspection_time < ValiConfig.CHALLENGE_PERIOD_MS

            # if the miner meets the criteria for passing, they are added to the passing list
            if passing_criteria:
                passing_miners.append(hotkey)
                continue

            # if the miner does not meet the criteria for passing, they are added to the failing list
            if not time_criteria:
                failing_miners.append(hotkey)
                continue

        return passing_miners, failing_miners

    @staticmethod
    def screen(
        position_elements: list[Position],
        ledger_element: PerfLedgerData,
        current_time: int,
        log: bool = False
    ) -> bool:
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period.
        """
        if ledger_element is None:
            return False

        if len(ledger_element.cps) == 0:
            return False

        if len(position_elements) <= 1:
            # We need at least more than 1 position to evaluate the challenge period
            return False

        minimum_return = ValiConfig.CHALLENGE_PERIOD_RETURN
        minimum_number_of_positions = ValiConfig.CHALLENGE_PERIOD_MIN_POSITIONS
        maximum_positional_returns_ratio = ValiConfig.CHALLENGE_PERIOD_MAX_POSITIONAL_RETURNS_RATIO
        maximum_drawdown = ValiConfig.CHALLENGE_PERIOD_MAX_DRAWDOWN_PERCENT

        # Check the closed positions from the miner
        filtered_positions = PositionUtils.filter_single_miner(
            position_elements,
            evaluation_time_ms=current_time,
            lookback_time_ms=ValiConfig.CHALLENGE_PERIOD_MS
        )

        # Drawdown Criteria
        max_drawdown = LedgerUtils.recent_drawdown(ledger_element.cps, restricted=False)
        max_drawdown_percentage = LedgerUtils.drawdown_percentage(max_drawdown)

        # Closed Positions Criteria
        closed_positions = [position for position in filtered_positions if position.is_closed_position]
        n_closed_positions = len(closed_positions)

        # Returns Ratio Criteria
        max_returns_ratio = PositionUtils.returns_ratio(filtered_positions)
        base_return = np.exp(Scoring.base_return(filtered_positions))

        # Evaluation
        return_criteria = base_return >= minimum_return
        closed_positions_criteria = n_closed_positions >= minimum_number_of_positions
        max_returns_ratio_criteria = max_returns_ratio <= maximum_positional_returns_ratio
        max_drawdown_criteria = max_drawdown_percentage >= maximum_drawdown

        if log:
            viewable_return = 100 * (base_return - 1)
            viewable_minimum_return = 100 * (minimum_return - 1)
            print(f"Return: {viewable_return:.4f}% >= {viewable_minimum_return:.2f}%: {return_criteria}")
            print()

        return return_criteria and closed_positions_criteria and max_returns_ratio_criteria and max_drawdown_criteria
