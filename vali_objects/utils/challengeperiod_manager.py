# developer: trdougherty
import time
import bittensor as bt
import copy

from datetime import datetime
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
from vali_objects.utils.elimination_manager import EliminationReason

class ChallengePeriodManager(CacheController):
    # Debug hotkey for detailed logging
    DEBUG_HOTKEY = "5FCPYqbYEq2y7NwQTCLxNApP2UjUE86J8QnhdWTHFkzzFWL1"
    def __init__(self, metagraph, perf_ledger_manager : PerfLedgerManager =None, running_unit_tests=False,
                 position_manager: PositionManager =None, ipc_manager=None, is_backtesting=False):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.perf_ledger_manager = perf_ledger_manager if perf_ledger_manager else \
            PerfLedgerManager(metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        self.elimination_manager = self.position_manager.elimination_manager
        self.eliminations_with_reasons: dict[str, tuple[str, float]] = {}
        if self.is_backtesting:
            initial_challenegeperiod_testing = {}
            initial_challenegeperiod_success = {}
        else:
            initial_challenegeperiod_testing = self.get_challengeperiod_testing(from_disk=True)
            initial_challenegeperiod_success = self.get_challengeperiod_success(from_disk=True)
        self.using_ipc = bool(ipc_manager)
        if ipc_manager:
            self.challengeperiod_testing = ipc_manager.dict()
            self.challengeperiod_success = ipc_manager.dict()
            for k, v in initial_challenegeperiod_testing.items():
                self.challengeperiod_testing[k] = v
            for k, v in initial_challenegeperiod_success.items():
                self.challengeperiod_success[k] = v
        else:
            self.challengeperiod_testing = initial_challenegeperiod_testing
            self.challengeperiod_success = initial_challenegeperiod_success
        if not self.is_backtesting and len(self.get_challengeperiod_testing()) == 0 and len(self.get_challengeperiod_success()) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests),
                {"testing": {}, "success": {}}
            )
        self.refreshed_challengeperiod_start_time = False


    #Used to bypass running challenge period, but still adds miners to success for statistics
    def add_all_miners_to_success(self, current_time_ms, run_elimination=True):
        assert self.is_backtesting, "This function is only for backtesting"
        eliminations = []
        if run_elimination:
            # The refresh should just read the current eliminations
            eliminations = self.elimination_manager.get_eliminations_from_memory()

            # Collect challenge period and update with new eliminations criteria
            self.remove_eliminated(eliminations=eliminations)

        challenge_hk_to_positions, challenge_hk_to_first_order_time = self.position_manager.filtered_positions_for_scoring(
            hotkeys=self.metagraph.hotkeys)

        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph.hotkeys,
            eliminations=eliminations,
            hk_to_first_order_time=challenge_hk_to_first_order_time
        )

        miners_to_promote = list(self.challengeperiod_testing.keys())

        #Finally promote all testing miners to success
        self._promote_challengeperiod_in_memory(hotkeys=miners_to_promote, current_time=current_time_ms)

    def _add_challengeperiod_testing_in_memory_and_disk(
            self,
            new_hotkeys: list[str],
            eliminations: list[dict] = None,
            hk_to_first_order_time: dict[str, int] = None
    ):

        if eliminations is None:
            eliminations = self.elimination_manager.get_eliminations_from_memory()

        elimination_hotkeys = set(x['hotkey'] for x in eliminations)

        any_changes = False
        for hotkey in new_hotkeys:
            if hotkey not in hk_to_first_order_time:  # miner has no positions
                continue

            start_time_ms = hk_to_first_order_time[hotkey]

            if start_time_ms is None:
                bt.logging.warning(f"Hotkey {hotkey} has invalid first order time. Skipping.")
                continue

            if hotkey in elimination_hotkeys:
                continue

            if hotkey in self.challengeperiod_success:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Already in challengeperiod_success, skipping")
                continue

            if hotkey in self.challengeperiod_testing:
                if self.challengeperiod_testing[hotkey] is None:
                    if hotkey == self.DEBUG_HOTKEY:
                        bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Fixing None start time to {start_time_ms}")
                    bt.logging.info(f"Fixing challengeperiod hotkey {hotkey} to to valid start time {start_time_ms}")
                    any_changes = True
                    self.challengeperiod_testing[hotkey] = start_time_ms
                elif hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Already in challengeperiod_testing with time {self.challengeperiod_testing[hotkey]}")
            else:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Adding to challengeperiod_testing with start time {start_time_ms}")
                bt.logging.info(f"Adding hotkey {hotkey} to challengeperiod miners with start time {start_time_ms}")
                any_changes = True
                self.challengeperiod_testing[hotkey] = start_time_ms

        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def _refresh_challengeperiod_start_time(self, hk_to_first_order_time_ms: dict[str, int]):
        """
        retroactively update the challengeperiod_testing start time based on time of first order.
        used when a miner is un-eliminated, and positions are preserved.
        """
        bt.logging.info("Refreshing challengeperiod start times")
        any_changes = False
        for hotkey, start_time_ms in self.challengeperiod_testing.items():
            if hotkey not in hk_to_first_order_time_ms:
                bt.logging.warning(f"Hotkey {hotkey} in challenge period has no first order time. Skipping for now.")
                continue
            first_order_time_ms = hk_to_first_order_time_ms[hotkey]
            if start_time_ms != first_order_time_ms:
                bt.logging.info(f"Challengeperiod start time for {hotkey} updated from: {datetime.utcfromtimestamp(start_time_ms/1000)} "
                                f"to: {datetime.utcfromtimestamp(first_order_time_ms/1000)}, {(start_time_ms-first_order_time_ms)/1000}s delta")
                self.challengeperiod_testing[hotkey] = first_order_time_ms
                any_changes = True
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()
        bt.logging.info("All challengeperiod start times up to date")

    def refresh(self, current_time: int = None):
        if not self.refresh_allowed(ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS):
            time.sleep(1)
            return
        bt.logging.info(f"Refreshing challenge period. invalidation data {self.perf_ledger_manager.perf_ledger_hks_to_invalidate}")
        
        # Debug logging for specific hotkey
        if self.DEBUG_HOTKEY in self.challengeperiod_testing:
            bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Found in testing dict at refresh start, testing_time={self.challengeperiod_testing[self.DEBUG_HOTKEY]}")
        if self.DEBUG_HOTKEY in self.challengeperiod_success:
            bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Found in success dict at refresh start, success_time={self.challengeperiod_success[self.DEBUG_HOTKEY]}")
        # The refresh should just read the current eliminations
        eliminations = self.elimination_manager.get_eliminations_from_memory()

        # Collect challenge period and update with new eliminations criteria
        self.remove_eliminated(eliminations=eliminations)
        challengeperiod_success_hotkeys = list(self.challengeperiod_success.keys())
        challengeperiod_testing_hotkeys = list(self.challengeperiod_testing.keys())

        all_miners = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys

        hk_to_positions, hk_to_first_order_time = self.position_manager.filtered_positions_for_scoring(hotkeys=self.metagraph.hotkeys)

        # challenge period adds to testing if not in eliminated, already in the challenge period, or in the new eliminations list from disk
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph.hotkeys,
            eliminations=eliminations,
            hk_to_first_order_time=hk_to_first_order_time
        )

        if not self.refreshed_challengeperiod_start_time:
            self.refreshed_challengeperiod_start_time = True
            self._refresh_challengeperiod_start_time(hk_to_first_order_time)

        ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=all_miners)
        ledger = {hotkey: ledger.get(hotkey, None) for hotkey in all_miners}

        challengeperiod_success, challengeperiod_eliminations = self.inspect(
            positions=hk_to_positions,
            ledger=ledger,
            success_hotkeys=challengeperiod_success_hotkeys,
            inspection_hotkeys=self.challengeperiod_testing,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time
        )
        self.eliminations_with_reasons = challengeperiod_eliminations

        any_changes = bool(challengeperiod_success) or bool(challengeperiod_eliminations)

        # Debug logging before promotion/demotion
        if self.DEBUG_HOTKEY in challengeperiod_success:
            bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: About to be PROMOTED to success")
        if self.DEBUG_HOTKEY in challengeperiod_eliminations:
            bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: About to be ELIMINATED, reason={challengeperiod_eliminations[self.DEBUG_HOTKEY]}")
        
        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(hotkeys=challengeperiod_success, current_time=current_time)
        self._demote_challengeperiod_in_memory(eliminations_with_reasons=challengeperiod_eliminations)

        # Now remove any miners who are no longer in the metagraph
        any_changes |= self._prune_deregistered_metagraph()

        # Now sync challenge period with the disk
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()
        self.set_last_update_time()

    def _prune_deregistered_metagraph(self, hotkeys=None) -> bool:
        """
        Prune the challenge period of all miners who are no longer in the metagraph
        """
        any_changes = False
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        for hotkey in list(self.challengeperiod_testing.keys()):
            if hotkey not in hotkeys:
                any_changes = True
                del self.challengeperiod_testing[hotkey]

        for hotkey in list(self.challengeperiod_success.keys()):
            if hotkey not in hotkeys:
                any_changes = True
                del self.challengeperiod_success[hotkey]

        return any_changes


    def is_recently_re_registered(self, ledger, hotkey, hk_to_first_order_time):
        """
        A miner can re-register and their perf ledger may still be old.
        This function checks for that condition and blocks challenge period failure so that
        the perf ledger can rebuild.
        """
        if not hk_to_first_order_time:
            return False
        if ledger:
            time_of_ledger_start = ledger.start_time_ms
        else:
            # No ledger? No edge case.
            return False

        first_order_time = hk_to_first_order_time.get(hotkey, None)
        if first_order_time is None:
            # No positions? Perf ledger must be stale.
            msg = f'No positions for hotkey {hotkey} - ledger start time: {time_of_ledger_start}'
            print(msg)
            return True

        # A perf ledger can never begin before the first order. Edge case confirmed.
        ans = time_of_ledger_start < first_order_time
        if ans:
            msg = (f'Hotkey {hotkey} has a ledger start time of {TimeUtil.millis_to_formatted_date_str(time_of_ledger_start)},'
                   f' a first order time of {TimeUtil.millis_to_formatted_date_str(first_order_time)}, and an'
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
        inspection_scores_dict: dict[str, dict] = None,
        hk_to_first_order_time: dict[str, int] = None

    ) -> tuple[list[str], dict[str, tuple[str, float]]]:
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period. Does not modify the challenge period in memory.

        Args:
            success_scores_dict (dict[str, dict]) - a dictionary with a similar structure to config with keys being
            function names of metrics and values having "scores" (scores of miners that passed challenge)
            and "weight" which is the weight of the metric. Only provided if running tests

            inspection_scores_dict (dict[str, dict]) - identical to success_scores_dict in structure, but only has data
            for one inspection hotkey. Only provided if running tests

        Returns:
            passing_miners - list of miners that passed the challenge period.
            failing_miner - dictionary of hotkey to a tuple of the form (reason failed challenge period, maximum drawdown)
        """
        if inspection_hotkeys is None:
            return [], {}  # no hotkeys to inspect

        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        passing_miners = []
        failing_miners = {}
        miners_rrr = set()        

        # If success_scoring_dict is already calculated, no need to calculate scores. Useful for testing
        if success_scores_dict is None:

            success_positions = dict((hotkey, miner_positions) for hotkey, miner_positions in positions.items() if hotkey in success_hotkeys)
            success_ledger = dict((hotkey, ledger_data) for hotkey, ledger_data in ledger.items() if hotkey in success_hotkeys)

            # Get the penalized scores of all successful miners
            success_scores_dict = Scoring.score_miners(ledger_dict=success_ledger,
                                                            positions=success_positions,
                                                            evaluation_time_ms=current_time,
                                                            weighting=True)
        
        miners_not_enough_positions = []
        for hotkey, inspection_time in inspection_hotkeys.items():
            # Debug logging for specific hotkey
            if hotkey == self.DEBUG_HOTKEY:
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Starting inspection - inspection_time={inspection_time}, current_time={current_time}")
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Time since testing started: {(current_time - inspection_time) / 1000 / 60:.1f} minutes")
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Challenge period limit: {ValiConfig.CHALLENGE_PERIOD_MS / 1000 / 60:.1f} minutes")
            if self.is_recently_re_registered(ledger.get(hotkey), hotkey, hk_to_first_order_time):
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Marked as recently re-registered, skipping inspection")
                miners_rrr.add(hotkey)
                continue
            elif inspection_time is None:
                bt.logging.warning(f'Hotkey {hotkey} has no inspection time. Unexpected.')
                continue
            # Default starts as true
            passing_criteria = True

            # We want to know if the miner still has time, as we know the criteria to pass is not met
            time_criteria = current_time - inspection_time <= ValiConfig.CHALLENGE_PERIOD_MS

            # Get hotkey to positions dict that only includes the inspection miner
            has_minimum_positions, inspection_positions = ChallengePeriodManager.screen_minimum_positions(positions=positions, inspection_hotkey=hotkey)
            if not has_minimum_positions:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: FAILED minimum positions check - positions={positions.get(hotkey, [])}")
                miners_not_enough_positions.append((hotkey, positions.get(hotkey, [])))
                passing_criteria = False
            elif hotkey == self.DEBUG_HOTKEY:
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: PASSED minimum positions check - position_count={len(positions.get(hotkey, []))}")

            # Get hotkey to ledger dict that only includes the inspection miner
            has_minimum_ledger, inspection_ledger = ChallengePeriodManager.screen_minimum_ledger(ledger=ledger, inspection_hotkey=hotkey)
            if not has_minimum_ledger:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: FAILED minimum ledger check - ledger={ledger.get(hotkey)}")
                passing_criteria = False
            elif hotkey == self.DEBUG_HOTKEY:
                ledger_element = ledger.get(hotkey)
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: PASSED minimum ledger check - cps_count={len(ledger_element.cps) if ledger_element else 0}")

            # This step is meant to ensure no positions or ledgers reference missing hotkeys, we need them to evaluate
            if not passing_criteria:
                if not time_criteria:
                    # If the miner registers, never interacts
                    bt.logging.info(f'Hotkey {hotkey} has no positions or ledger, and has not interacted since registration. cp_failed')
                    failing_miners[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value, -1)

                continue  # Moving on, as the miner is already failing
            # This step we want to check their drawdown. If they fail, we can move on.
            failing_criteria, recorded_drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element=ledger[hotkey])

            if failing_criteria:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: FAILED drawdown check - drawdown={recorded_drawdown_percentage}%")
                bt.logging.info(f'Hotkey {hotkey} has failed the challenge period due to drawdown {recorded_drawdown_percentage}. cp_failed')
                failing_miners[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value, recorded_drawdown_percentage)
                continue
            elif hotkey == self.DEBUG_HOTKEY:
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: PASSED drawdown check - drawdown={recorded_drawdown_percentage}%")
            

            # The main logic loop. They are in the competition but haven't passed yet, need to check the time after.
            if hotkey == self.DEBUG_HOTKEY:
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: About to check passing criteria")
            passing_criteria = ChallengePeriodManager.screen_passing_criteria(
                inspection_positions=inspection_positions,
                inspection_ledger=inspection_ledger,
                inspection_hotkey=hotkey,
                success_scores_dict=success_scores_dict,
                current_time=current_time,
                inspection_scores_dict=inspection_scores_dict
            )
            if hotkey == self.DEBUG_HOTKEY:
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Passing criteria result: {passing_criteria}")

            # If they pass here, then they meet the criteria for passing within the challenge period
            if passing_criteria:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: PASSED all criteria - will be promoted!")
                passing_miners.append(hotkey)
                continue

            # If their time is ever up, they fail
            if not time_criteria:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: FAILED time criteria - time is up!")
                bt.logging.info(f'Hotkey {hotkey} has failed the challenge period due to time. cp_failed')
                failing_miners[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value, recorded_drawdown_percentage)
                continue
            elif hotkey == self.DEBUG_HOTKEY:
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Still has time remaining - continuing in challenge period")

        if miners_not_enough_positions:
            bt.logging.info(f'Challenge Period - miners with not enough positions: {miners_not_enough_positions}')
        bt.logging.info(f'Challenge Period - n_miners_passing: {len(passing_miners)}'
                        f' n_miners_failing: {len(failing_miners)} '
                        f'recently_re_registered: {miners_rrr} '
                        f'n_miners_inspected {len(inspection_hotkeys)}')
        return passing_miners, failing_miners
    
    @staticmethod
    def screen_passing_criteria(
        inspection_positions: dict[str, list[Position]],
        inspection_ledger: dict[str, PerfLedger],
        success_scores_dict: dict[str, dict],
        inspection_hotkey: str,
        current_time: int,
        inspection_scores_dict = None
    ) -> bool:
        DEBUG_HOTKEY = "5FCPYqbYEq2y7NwQTCLxNApP2UjUE86J8QnhdWTHFkzzFWL1"
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period.
        Args:
            success_scores_dict: a dictionary with a similar structure to config with keys being
            function names of metrics and values having "scores" (scores of miners that passed challenge)
            and "weight" which is the weight of the metric
        """
        # Before scoring, check that the miner has enough trading days to be promoted
        min_interaction_criteria = ChallengePeriodManager.screen_minimum_interaction(
            ledger_element=inspection_ledger.get(inspection_hotkey))
        if not min_interaction_criteria:
            if inspection_hotkey == DEBUG_HOTKEY:
                ledger_element = inspection_ledger.get(inspection_hotkey)
                if ledger_element:
                    from vali_objects.utils.ledger_utils import LedgerUtils
                    miner_returns = LedgerUtils.daily_return_log(ledger_element)
                    bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: FAILED minimum interaction - daily_returns_count={len(miner_returns)}, required={ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS.value()}")
                else:
                    bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: FAILED minimum interaction - no ledger element")
            return False
        elif inspection_hotkey == DEBUG_HOTKEY:
            ledger_element = inspection_ledger.get(inspection_hotkey)
            from vali_objects.utils.ledger_utils import LedgerUtils
            miner_returns = LedgerUtils.daily_return_log(ledger_element)
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: PASSED minimum interaction - daily_returns_count={len(miner_returns)}, required={ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS.value()}")

        # inspection_scores_dict is used to bypass running scoring when testing
        if inspection_scores_dict is None:
            # Get scores of inspection miner and penalties
            inspection_scores_dict = Scoring.score_miners(
                ledger_dict=inspection_ledger,
                positions=inspection_positions,
                evaluation_time_ms=current_time,
                weighting=True)
        
        # Debug: Show inspection miner's individual metric scores before combining
        if inspection_hotkey == DEBUG_HOTKEY:
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: === INSPECTION MINER INDIVIDUAL SCORES ===")
            for metric_name, metric_data in inspection_scores_dict["metrics"].items():
                scores = metric_data["scores"]
                weight = metric_data["weight"]
                if scores:  # Should only have one score for the inspection miner
                    miner, score = scores[0]
                    bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - {metric_name.upper()}: raw_score={score:.6f}, weight={weight}")
                else:
                    bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - {metric_name.upper()}: NO SCORE")
            
            penalty = inspection_scores_dict["penalties"].get(inspection_hotkey, 1.0)
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - PENALTY: {penalty}")
            
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
        
        if inspection_hotkey == DEBUG_HOTKEY:
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: === DETAILED SCORING BREAKDOWN ===")
            
            # Show individual metric scores and percentiles
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: Individual Metric Performance:")
            for metric_name, metric_config in trial_scores_dict["metrics"].items():
                metric_scores = metric_config["scores"]
                weight = metric_config["weight"]
                
                # Find this miner's score for this metric
                miner_score = None
                for miner, score in metric_scores:
                    if miner == inspection_hotkey:
                        miner_score = score
                        break
                
                if miner_score is not None:
                    # Calculate percentile for this metric
                    all_scores = [score for _, score in metric_scores]
                    if len(all_scores) > 1:
                        from scipy.stats import percentileofscore
                        metric_percentile = percentileofscore(all_scores, miner_score, kind='rank') / 100
                    else:
                        metric_percentile = 1.0
                    
                    bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - {metric_name.upper()}: score={miner_score:.6f}, percentile={metric_percentile:.3f}, weight={weight}, contribution={weight * metric_percentile:.6f}")
                else:
                    bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - {metric_name.upper()}: NO SCORE FOUND")
            
            # Show penalty information
            penalty = trial_scores_dict["penalties"].get(inspection_hotkey, 1.0)
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: Penalty multiplier: {penalty}")
            
            # Show score distribution context
            all_combined_scores = list(combined_scores.values())
            all_combined_scores.sort(reverse=True)
            miner_combined_score = combined_scores.get(inspection_hotkey, 0)
            miner_rank = all_combined_scores.index(miner_combined_score) + 1 if miner_combined_score in all_combined_scores else len(all_combined_scores)
            
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: Final Results:")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Combined score: {miner_combined_score:.6f}")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Rank: {miner_rank}/{len(combined_scores)}")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Percentile: {inspection_percentile:.3f}")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Required threshold: {ValiConfig.CHALLENGE_PERIOD_PERCENTILE_THRESHOLD}")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Result: {'PASSED' if passed else 'FAILED'}")
            
            # Show top and bottom performers for context
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]: Score Distribution Context:")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Best score: {max(all_combined_scores):.6f}")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Worst score: {min(all_combined_scores):.6f}")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - Median score: {all_combined_scores[len(all_combined_scores)//2]:.6f}")
            bt.logging.info(f"DEBUG [{DEBUG_HOTKEY}]:   - 75th percentile score: {all_combined_scores[int(len(all_combined_scores)*0.25)]:.6f}")

        return passed

    @staticmethod
    def screen_minimum_interaction(
            ledger_element: PerfLedger
    ) -> (bool, float):
        """
        Returns False if the miner doesn't have the minimum number of trading days.
        """
        if ledger_element is None:
            bt.logging.warning("Ledger element is None. Returning False.")
            return False

        miner_returns = LedgerUtils.daily_return_log(ledger_element)
        return len(miner_returns) >= ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS.value()

    @staticmethod
    def screen_minimum_ledger(
            ledger: dict[str, PerfLedger],
            inspection_hotkey: str
    ) -> tuple[bool, dict[str, PerfLedger]]:
        """
        Ensures there is enough ledger data globally and for the specific miner to evaluate challenge period.
        """

        if ledger is None or len(ledger) == 0:
            bt.logging.info(f"No ledgers for any miner to evaluate for challenge period. ledger: {ledger}")
            return False, {}

        single_ledger = ledger.get(inspection_hotkey, None)
        has_minimum_ledger = single_ledger is not None and len(single_ledger.cps) > 0
        if not has_minimum_ledger:
            bt.logging.info(f"Hotkey: {inspection_hotkey} doesn't have the minimum ledger for challenge period. ledger: {single_ledger}")

        inspection_ledger = {inspection_hotkey: single_ledger} if has_minimum_ledger else {}

        return has_minimum_ledger, inspection_ledger


    @staticmethod
    def screen_minimum_positions(
            positions: dict[str, list[Position]],
            inspection_hotkey: str
    ) -> tuple[bool, dict[str, list[Position]]]:
        """
        Ensures there are enough positions globally and for the specific miner to evaluate challenge period.
        """

        if positions is None or len(positions) == 0:
            bt.logging.info(f"No positions for any miner to evaluate for challenge period. positions: {positions}")
            return False, {}

        positions_list = positions.get(inspection_hotkey, None)
        has_minimum_positions = positions_list is not None and len(positions_list) > 0

        inspection_positions = {inspection_hotkey: positions_list} if has_minimum_positions else {}

        return has_minimum_positions, inspection_positions


    def get_challengeperiod_testing(self, from_disk=False):
        if from_disk:
            return ValiUtils.get_vali_json_file_dict(
                ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
            ).get('testing', {})
        else:
            ans = self.challengeperiod_testing
            if self.using_ipc:
                return copy.deepcopy(ans)
            return ans

    def sync_challenege_period_data(self, challenge_period_testing, challenge_period_success):
        temp = [(self.challengeperiod_testing, challenge_period_testing),
                (self.challengeperiod_success, challenge_period_success)]
        for ref_dict, dat_to_copy in temp:
            if not dat_to_copy:
                bt.logging.error(f'challenge_period_data {(challenge_period_testing, challenge_period_success)} appears invalid')
            ref_dict.clear()
            ref_dict.update(dat_to_copy)
        self._write_challengeperiod_from_memory_to_disk()

    def get_challengeperiod_success(self, from_disk=False):
        if from_disk:
            return ValiUtils.get_vali_json_file_dict(
                ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=self.running_unit_tests)
            ).get('success', {})
        else:
            ans = self.challengeperiod_success
            if self.using_ipc:
                return copy.deepcopy(ans)
            return ans

    def _remove_eliminated_from_memory(self, eliminations: list[dict] = None) -> bool:
        if eliminations is None:
            eliminations_hotkeys = self.elimination_manager.get_eliminated_hotkeys()
        else:
            eliminations_hotkeys = set([x['hotkey'] for x in eliminations])

        any_changes = False
        for k in list(self.challengeperiod_testing.keys()):
            if k in eliminations_hotkeys:
                any_changes = True
                del self.challengeperiod_testing[k]

        for k in list(self.challengeperiod_success.keys()):
            if k in eliminations_hotkeys:
                any_changes = True
                del self.challengeperiod_success[k]
        return any_changes

    def remove_eliminated(self, eliminations=None):
        if eliminations is None:
            eliminations = []

        any_changes = self._remove_eliminated_from_memory(eliminations=eliminations)
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def clear_challengeperiod_from_disk(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_challengeperiod_file_location(
            running_unit_tests=self.running_unit_tests),
            {"testing": {}, "success": {}}
        )

    def _clear_challengeperiod_in_memory_and_disk(self):
        for k in list(self.challengeperiod_testing.keys()):
            del self.challengeperiod_testing[k]
        for k in list(self.challengeperiod_success.keys()):
            del self.challengeperiod_success[k]

        self.clear_challengeperiod_from_disk()

    def _promote_challengeperiod_in_memory(self, hotkeys: list[str], current_time: int):
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting hotkeys {hotkeys} to challengeperiod success.")

        new_success = {hotkey: current_time for hotkey in hotkeys}
        for k, v in new_success.items():
            if k == self.DEBUG_HOTKEY:
                bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Adding to challengeperiod_success with time {v}")
            self.challengeperiod_success[k] = v

        for hotkey in hotkeys:
            if hotkey in self.challengeperiod_testing:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Removing from challengeperiod_testing")
                del self.challengeperiod_testing[hotkey]
            else:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.error(f"DEBUG [{self.DEBUG_HOTKEY}]: ERROR - Was not in challengeperiod_testing but promotion attempted!")
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but promotion to success was attempted.")

    def _demote_challengeperiod_in_memory(self, eliminations_with_reasons: dict[str, tuple[str, float]]):
        hotkeys = list(eliminations_with_reasons.keys())
        if hotkeys:
            bt.logging.info(f"Removing hotkeys {hotkeys} from challenge period.")

        for hotkey in hotkeys:
            if hotkey in self.challengeperiod_testing:
                if hotkey == self.DEBUG_HOTKEY:
                    reason, drawdown = eliminations_with_reasons[hotkey]
                    bt.logging.info(f"DEBUG [{self.DEBUG_HOTKEY}]: Being eliminated from testing - reason={reason}, drawdown={drawdown}")
                del self.challengeperiod_testing[hotkey]
            else:
                if hotkey == self.DEBUG_HOTKEY:
                    bt.logging.error(f"DEBUG [{self.DEBUG_HOTKEY}]: ERROR - Was not in challengeperiod_testing but demotion attempted!")
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but demotion to failure was attempted.")

    def _write_challengeperiod_from_memory_to_disk(self):
        if self.is_backtesting:
            return
        challengeperiod_data = {
            "testing": self.get_challengeperiod_testing(),
            "success": self.get_challengeperiod_success()
        }
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_challengeperiod_file_location(
                running_unit_tests=self.running_unit_tests
            ),
            challengeperiod_data
        )
