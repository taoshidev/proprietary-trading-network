# developer: trdougherty

import time
import bittensor as bt
from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.scoring import Scoring
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfLedgerData
from vali_objects.utils.position_filtering import PositionFiltering
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.position import Position


class ChallengePeriodManager(CacheController):
    def __init__(self, config, metagraph, running_unit_tests=False, position_manager=None):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.perf_manager = PerfLedgerManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests) if position_manager is None else position_manager

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

        challenge_period_miners = list(self.challengeperiod_testing.keys())

        # Check that our miners are in challenge period - don't need to get all of them
        positions = self.position_manager.get_all_miner_positions_by_hotkey(
            challenge_period_miners,
            sort_positions=True
        )
        ledger = self.perf_manager.load_perf_ledgers_from_disk()
        ledger = {hotkey: ledger.get(hotkey, None) for hotkey in challenge_period_miners}

        challengeperiod_success, challengeperiod_eliminations = self.inspect(
            positions=positions,
            ledger=ledger,
            inspection_hotkeys=self.challengeperiod_testing,
            current_time=current_time
        )

        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(hotkeys=challengeperiod_success, current_time=current_time)
        self._demote_challengeperiod_in_memory(hotkeys=challengeperiod_eliminations)

        # Now remove any miners who are no longer in the metagraph
        self._prune_deregistered_metagraph()

        # Now sync challenge period with the disk

        self._write_challengeperiod_from_memory_to_disk()
        if challengeperiod_eliminations:
            for hotkey in challengeperiod_eliminations:
                self.position_manager.handle_eliminated_miner(hotkey, {})
            self._write_eliminations_from_memory_to_disk()
        self.set_last_update_time()

    def _prune_deregistered_metagraph(self, hotkeys=None):
        """
        Prune the challenge period of all miners who are no longer in the metagraph
        """
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        for hotkey in list(self.challengeperiod_testing.keys()):
            if hotkey not in hotkeys:
                self.challengeperiod_testing.pop(hotkey)

        for hotkey in list(self.challengeperiod_success.keys()):
            if hotkey not in hotkeys:
                self.challengeperiod_success.pop(hotkey)


    def is_recently_re_registered(self, ledger, positions, hotkey):
        """
        A miner can re-register and their perf ledger may still be old.
        This function checks for that condition and blocks challenge period failure so that
        the perf ledger can rebuild.
        """
        if ledger:
            time_of_ledger_start = ledger.start_time_ms
        else:
            # No ledger? No edge case.
            return False
        if positions and all(p.orders for p in positions):
            time_of_first_order = min(p.orders[0].processed_ms for p in positions)
        else:
            # No positions? Perf ledger must be stale.
            msg = f'No positions for hotkey {hotkey} - ledger start time: {time_of_ledger_start}'
            print(msg)
            return True

        # A perf ledger can never begin before the first order. Edge case confirmed.
        ans = time_of_ledger_start < time_of_first_order
        if ans:
            msg = f'Hotkey {hotkey} has a ledger start time of {TimeUtil.millis_to_formatted_date_str(time_of_ledger_start)} and a first order time of {TimeUtil.millis_to_formatted_date_str(time_of_first_order)}'
            print(msg)
        return ans

    def inspect(
        self,
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedgerData],
        inspection_hotkeys: dict[str, int] = None,
        current_time: int = None,
        log: bool = False,
    ):
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period. Does not modify the challenge period in memory.
        """
        if inspection_hotkeys is None:
            return [], []  # no hotkeys to inspect

        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        passing_miners = []
        failing_miners = []
        miners_rrr = set()
        for hotkey, inspection_time in inspection_hotkeys.items():
            if self.is_recently_re_registered(ledger.get(hotkey), positions.get(hotkey), hotkey):
                miners_rrr.add(hotkey)
                continue
            # Default starts as true
            passing_criteria = True

            # We want to know if the miner still has time, as we know the criteria to pass is not met
            time_criteria = current_time - inspection_time <= ValiConfig.CHALLENGE_PERIOD_MS

            # Check if hotkey is in ledger and has checkpoints (cps)
            if hotkey not in ledger:
                passing_criteria = False

            # Check if hotkey is in positions and has at least one position
            if hotkey not in positions:
                passing_criteria = False

            # This step is meant to ensure no positions or ledgers reference missing hotkeys, we need them to evaluate
            if not passing_criteria:
                if not time_criteria:
                    # If the miner registers, never interacts
                    bt.logging.info(f'Hotkey {hotkey} has no positions or ledger, and has not interacted since registration. cp_failed')
                    failing_miners.append(hotkey)

                continue  # Moving on, as the miner is already failing

            # This step we want to check their failure criteria. If they fail, we can move on.
            failing_criteria, recorded_drawdown_percentage = ChallengePeriodManager.screen_failing_criteria(ledger_element=ledger[hotkey])
            if failing_criteria:
                bt.logging.info(f'Hotkey {hotkey} has failed the challenge period due to drawdown {recorded_drawdown_percentage}. cp_failed')
                failing_miners.append(hotkey)
                continue

            # The main logic loop. They are in the competition but haven't passed yet, need to check the time after.
            passing_criteria = ChallengePeriodManager.screen_passing_criteria(
                position_elements=positions[hotkey],
                ledger_element=ledger[hotkey],
                current_time=current_time,
                log=log
            )

            # If they pass here, then they meet the criteria for passing within the challenge period
            if passing_criteria:
                passing_miners.append(hotkey)
                continue

            # If their time is ever up, they fail
            if not time_criteria:
                bt.logging.info(f'Hotkey {hotkey} has failed the challenge period due to time. cp_failed')
                failing_miners.append(hotkey)
                continue

        bt.logging.info(f'Challenge Period - n_miners_passing: {len(passing_miners)}'
                        f' n_miners_failing: {len(failing_miners)} '
                        f'recently_re_registered: {miners_rrr} '
                        f'n_miners_inspected {len(inspection_hotkeys)}')
        return passing_miners, failing_miners

    @staticmethod
    def screen_passing_criteria(
        position_elements: list[Position],
        ledger_element: PerfLedgerData,
        current_time: int,
        log: bool = False
    ) -> bool:
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period.
        """
        if position_elements is None:
            return False

        if ledger_element is None:
            return False

        if len(ledger_element.cps) == 0:
            return False

        if len(position_elements) <= 1:
            # We need at least more than 1 position to evaluate the challenge period
            return False

        minimum_return = ValiConfig.CHALLENGE_PERIOD_RETURN_LOG
        minimum_number_of_positions = ValiConfig.CHALLENGE_PERIOD_MIN_POSITIONS
        maximum_positional_returns_ratio = ValiConfig.CHALLENGE_PERIOD_MAX_POSITIONAL_RETURNS_RATIO
        maximum_unrealized_return_ratio = ValiConfig.CHALLENGE_PERIOD_MAX_UNREALIZED_RETURNS_RATIO

        # Check the closed positions from the miner
        filtered_positions = PositionFiltering.filter_single_miner(
            position_elements,
            evaluation_time_ms=current_time,
            lookback_time_ms=current_time  # Setting to current time will filter for all positions through all time
        )

        # Closed Positions Criteria
        closed_positions = [position for position in filtered_positions if position.is_closed_position]
        recorded_n_closed_positions = len(closed_positions)

        # Returns Ratio Criteria
        recorded_returns_ratio = PositionPenalties.returns_ratio(filtered_positions)
        recorded_return = Scoring.base_return(filtered_positions)

        # Unrealized Gains Ratio Criteria
        recorded_unrealized_ratio = LedgerUtils.daily_consistency_ratio(ledger_element.cps)

        # Evaluation
        # Recorded returns are greater than minimum returns - log
        return_criteria = recorded_return >= minimum_return

        # Recorded number of closed positions are greater than minimum number of positions
        closed_positions_criteria = recorded_n_closed_positions >= minimum_number_of_positions

        # Ratio of largest to overall is less than our maximum permitted ratio
        max_returns_ratio_criteria = recorded_returns_ratio < maximum_positional_returns_ratio

        # Ratio of unrealized gains to total returns is less than our maximum permitted ratio
        max_unrealized_return_ratio_criteria = recorded_unrealized_ratio <= maximum_unrealized_return_ratio

        if log:
            viewable_return = 100 * (recorded_return - 1)
            viewable_minimum_return = 100 * (minimum_return - 1)
            print(f"Return: {viewable_return:.4f}% >= {viewable_minimum_return:.2f}%: {return_criteria}")
            print()

        return return_criteria and closed_positions_criteria and max_returns_ratio_criteria and max_unrealized_return_ratio_criteria

    @staticmethod
    def screen_failing_criteria(
        ledger_element: PerfLedgerData
    ) -> (bool, float):
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period. Returns True if they fail.
        """
        if ledger_element is None:
            return False, 0

        if len(ledger_element.cps) == 0:
            return False, 0

        maximum_drawdown_percent = ValiConfig.CHALLENGE_PERIOD_MAX_DRAWDOWN_PERCENT

        max_drawdown = LedgerUtils.recent_drawdown(ledger_element.cps, restricted=False)
        recorded_drawdown_percentage = LedgerUtils.drawdown_percentage(max_drawdown)

        # Drawdown is less than our maximum permitted drawdown
        max_drawdown_criteria = recorded_drawdown_percentage >= maximum_drawdown_percent

        return max_drawdown_criteria, recorded_drawdown_percentage

