# developer: trdougherty
import threading
import time
import bittensor as bt
import copy

from setproctitle import setproctitle

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.scoring import Scoring
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfLedger
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.position import Position


class ChallengePeriodManager(CacheController):
    def __init__(self, metagraph, perf_ledger_manager : PerfLedgerManager =None, running_unit_tests=False,
                 position_manager: PositionManager =None):
        super().__init__(metagraph, running_unit_tests=running_unit_tests)
        self.perf_ledger_manager = perf_ledger_manager if perf_ledger_manager else \
            PerfLedgerManager(metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        self.elimination_manager = self.position_manager.elimination_manager
        self.challengeperiod_testing = {}
        self.challengeperiod_success = {}
        if len(self.get_challengeperiod_testing()) == 0 and len(self.get_challengeperiod_success()) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests),
                {"testing": {}, "success": {}}
            )

    def _add_challengeperiod_testing_in_memory_and_disk(
            self,
            new_hotkeys: list[str],
            eliminations: list[dict] = None,
            current_time: int = None
    ):
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        if eliminations is None:
            eliminations = self.elimination_manager.get_eliminations_from_memory()

        elimination_hotkeys = [x['hotkey'] for x in eliminations]

        # check all hotkeys which have at least one position
        miners_with_positions = self.position_manager.get_miner_hotkeys_with_at_least_one_position()

        for hotkey in new_hotkeys:
            if hotkey in miners_with_positions:
                if hotkey not in elimination_hotkeys:
                    if hotkey not in self.challengeperiod_testing:
                        if hotkey not in self.challengeperiod_success:
                            bt.logging.info(f"Adding hotkey {hotkey} to challengeperiod miners.")
                            self.challengeperiod_testing[hotkey] = current_time

        self._write_challengeperiod_from_memory_to_disk()

    def refresh(self, current_time: int = None):
        if not self.refresh_allowed(ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS):
            time.sleep(1)
            return
        setproctitle(f"vali_{self.__class__.__name__}")
        # The refresh should just read the current eliminations
        eliminations = self.elimination_manager.get_eliminations_from_memory()

        # Collect challenge period and update with new eliminations criteria
        self._refresh_challengeperiod_in_memory_and_disk(eliminations=eliminations)

        # challenge period adds to testing if not in eliminated, already in the challenge period, or in the new eliminations list from disk
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph.hotkeys,
            eliminations=eliminations,
            current_time=current_time
        )
        challengeperiod_success_hotkeys = list(self.challengeperiod_success.keys())
        challengeperiod_testing_hotkeys = list(self.challengeperiod_testing.keys())

        all_miners = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys

        # Check that our miners are in challenge period - don't need to get all of them
        positions = self.position_manager.get_positions_for_hotkeys(
            all_miners,
            sort_positions=True
        )
        ledger = self.perf_ledger_manager.get_perf_ledgers_from_memory()
        ledger = {hotkey: ledger.get(hotkey, None) for hotkey in all_miners}

        challengeperiod_success, challengeperiod_eliminations = self.inspect(
            positions=positions,
            ledger=ledger,
            success_hotkeys=challengeperiod_success_hotkeys,
            inspection_hotkeys=self.challengeperiod_testing,
            current_time=current_time
        )

        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(hotkeys=challengeperiod_success, current_time=current_time)
        self._demote_challengeperiod(hotkeys=challengeperiod_eliminations)

        # Now remove any miners who are no longer in the metagraph
        self._prune_deregistered_metagraph()

        # Now sync challenge period with the disk
        self._write_challengeperiod_from_memory_to_disk()
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
            msg = (f'Hotkey {hotkey} has a ledger start time of {TimeUtil.millis_to_formatted_date_str(time_of_ledger_start)},'
                   f' a first order time of {TimeUtil.millis_to_formatted_date_str(time_of_first_order)}, and an'
                   f' initialization time of {TimeUtil.millis_to_formatted_date_str(ledger.initialization_time_ms)}.')
            print(msg)
        return ans

    def inspect(
        self,
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedger],
        success_hotkeys: list[str],
        inspection_hotkeys: dict[str, int] = None,
        current_time: int = None,
        success_scores_dict: dict[str, dict] = None,
        inspection_scores_dict: dict[str, dict] = None
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

        # If success_scoring_dict is already calculated, no need to calculate scores. Useful for testing
        if success_scores_dict is None:

            success_positions = dict((hotkey, miner_positions) for hotkey, miner_positions in positions.items() if hotkey in success_hotkeys)
            success_ledger = dict((hotkey, ledger_data) for hotkey, ledger_data in ledger.items() if hotkey in success_hotkeys)

            # Get the penalized scores of all successful miners
            success_scores_dict = Scoring.score_miners(ledger_dict=success_ledger,
                                                            positions=success_positions,
                                                            evaluation_time_ms=current_time)
        

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
                positions=positions,
                ledger=ledger,
                inspection_hotkey=hotkey,
                success_scores_dict=success_scores_dict,
                current_time=current_time,
                inspection_scores_dict=inspection_scores_dict
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
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedger],
        success_scores_dict: dict[str, dict],
        inspection_hotkey: str,
        current_time: int,
        inspection_scores_dict = None
    ) -> bool:
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period.
        Args:
            success_scores_dict: a dictionary with a similar structure to config with keys being
            function names of metrics and values having "scores" (scores of miners that passed challenge)
            and "weight" which is the weight of the metric
        """
        # inspection_scores_dict is used to bypass running scoring when testing
        if inspection_scores_dict is None:

            if positions is None or len(positions) == 0:
                return False

            positions_list = positions.get(inspection_hotkey, None)

            if positions_list is None:
                return False

            if len(positions_list) <= 1:
                # We need at least more than 1 position to evaluate the challenge period
                return False

            inspection_positions = {inspection_hotkey: positions_list}
            # Get individual scoring dict for inspection
            inspection_ledger = {inspection_hotkey: ledger.get(inspection_hotkey, None)}

            if inspection_ledger.get(inspection_hotkey) is None:
                return False


            # Get penalized scores of inspection miner
            inspection_scores_dict = Scoring.score_miners(
                ledger_dict=inspection_ledger,
                positions=inspection_positions,
                evaluation_time_ms=current_time)
            
        trial_scores_dict = copy.deepcopy(success_scores_dict)

        for config_name, config in trial_scores_dict["metrics"].items():

            miner_scores = config["scores"]
            miner_scores += inspection_scores_dict["metrics"][config_name]["scores"]

        trial_scores_dict["penalties"].update(inspection_scores_dict["penalties"])

        combined_scores = Scoring.combine_scores(scoring_dict=trial_scores_dict)

        percentiles = Scoring.miner_scores_percentiles(list(combined_scores.items()))

        percentile_dict = dict(percentiles)
        inspection_percentile = percentile_dict.get(inspection_hotkey, 0)

        passed = inspection_percentile >= ValiConfig.CHALLENGE_PERIOD_PERCENTILE_THRESHOLD

        return passed

    @staticmethod
    def screen_failing_criteria(
        ledger_element: PerfLedger
    ) -> (bool, float):
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period. Returns True if they fail.
        """
        if ledger_element is None:
            return False, 0

        if len(ledger_element.cps) == 0:
            return False, 0

        maximum_drawdown_percent = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE

        max_drawdown = LedgerUtils.recent_drawdown(ledger_element.cps, restricted=False)
        recorded_drawdown_percentage = LedgerUtils.drawdown_percentage(max_drawdown)

        # Drawdown is less than our maximum permitted drawdown
        max_drawdown_criteria = recorded_drawdown_percentage >= maximum_drawdown_percent

        return max_drawdown_criteria, recorded_drawdown_percentage


    def get_challengeperiod_testing(self):
        return ValiUtils.get_vali_json_file_dict(
            ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
        ).get('testing', {})

    def get_challengeperiod_success(self):
        return ValiUtils.get_vali_json_file_dict(
            ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
        ).get('success', {})

    def _refresh_challengeperiod_in_memory(self, eliminations: list[dict] = None):
        if eliminations is None:
            eliminations_hotkeys = self.elimination_manager.get_eliminated_hotkeys()
        else:
            eliminations_hotkeys = set([x['hotkey'] for x in eliminations])

        location = ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
        existing_challengeperiod = ValiUtils.get_vali_json_file_dict(location)
        existing_challengeperiod_testing = existing_challengeperiod.get('testing', {})
        existing_challengeperiod_success = existing_challengeperiod.get('success', {})

        self.challengeperiod_testing = {k: v for k, v in existing_challengeperiod_testing.items() if k not in eliminations_hotkeys}
        self.challengeperiod_success = {k: v for k, v in existing_challengeperiod_success.items() if k not in eliminations_hotkeys}

    def _refresh_challengeperiod_in_memory_and_disk(self, eliminations=None):
        if eliminations is None:
            eliminations = []

        self._refresh_challengeperiod_in_memory(eliminations=eliminations)
        self._write_challengeperiod_from_memory_to_disk()

    def clear_challengeperiod_from_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_challengeperiod_file_location(
            running_unit_tests=self.running_unit_tests),
            {"testing": {}, "success": {}}
        )

    def _clear_challengeperiod_in_memory_and_disk(self):
        self.challengeperiod_testing = {}
        self.challengeperiod_success = {}
        self.clear_challengeperiod_from_disk()

    def _promote_challengeperiod_in_memory(self, hotkeys: list[str], current_time: int):
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting hotkeys {hotkeys} to challengeperiod success.")

        new_success = {hotkey: current_time for hotkey in hotkeys}
        self.challengeperiod_success = {
            **self.challengeperiod_success,
            **new_success
        }

        for hotkey in hotkeys:
            if hotkey in self.challengeperiod_testing:
                self.challengeperiod_testing.pop(hotkey)
            else:
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but promotion to success was attempted.")

    def _demote_challengeperiod(self, hotkeys: list[str]):
        for hotkey in hotkeys:
            bt.logging.info(f"Removing hotkeys {hotkey} from challenge period.")
            if hotkey in self.challengeperiod_testing:
                self.challengeperiod_testing.pop(hotkey)
            else:
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but demotion to failure was attempted.")

        for hotkey in hotkeys:
            bt.logging.info(f"Eliminating hotkey {hotkey}.")

            # This will also add the hotkey to the in memory self.eliminations list
            self.elimination_manager.append_elimination_row(hotkey, -1, 'FAILED_CHALLENGE_PERIOD')

    def _write_challengeperiod_from_memory_to_disk(self):
        challengeperiod_data = {
            "testing": self.challengeperiod_testing,
            "success": self.challengeperiod_success
        }
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_challengeperiod_file_location(
                running_unit_tests=self.running_unit_tests
            ),
            challengeperiod_data
        )


