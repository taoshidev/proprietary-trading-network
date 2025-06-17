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
from vali_objects.utils.miner_bucket_enum import MinerBucket

class ChallengePeriodManager(CacheController):
    def __init__(
            self,
            metagraph,
            perf_ledger_manager : PerfLedgerManager=None,
            position_manager: PositionManager=None,
            ipc_manager=None,
            *,
            running_unit_tests=False,
            is_backtesting=False):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.perf_ledger_manager = perf_ledger_manager if perf_ledger_manager else \
            PerfLedgerManager(metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        self.elimination_manager = self.position_manager.elimination_manager
        self.eliminations_with_reasons: dict[str, tuple[str, float]] = {}

        self.CHALLENGE_FILE = ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=running_unit_tests)

        self.active_miners = {}
        initial_active_miners = {}
        if not self.is_backtesting:
            disk_data = ValiUtils.get_vali_json_file_dict(self.CHALLENGE_FILE)
            initial_active_miners = self.parse_checkpoint_dict(disk_data)

        if ipc_manager:
            self.active_miners = ipc_manager.dict(initial_active_miners)
        else:
            self.active_miners = initial_active_miners

        if not self.is_backtesting and len(self.active_miners) == 0:
            self._write_challengeperiod_from_memory_to_disk()

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
            hk_to_first_order_time=challenge_hk_to_first_order_time,
            default_time=current_time_ms
        )

        miners_to_promote = self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE) \
                          + self.get_hotkeys_by_bucket(MinerBucket.PROBATION)

        #Finally promote all testing miners to success
        self._promote_challengeperiod_in_memory(miners_to_promote, current_time_ms)

    def _add_challengeperiod_testing_in_memory_and_disk(
            self,
            new_hotkeys: list[str],
            eliminations: list[dict],
            hk_to_first_order_time: dict[str, int],
            default_time: int
    ):
        if not eliminations:
            eliminations = self.elimination_manager.get_eliminations_from_memory()

        elimination_hotkeys = set(x['hotkey'] for x in eliminations)
        maincomp_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        probation_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PROBATION)

        any_changes = False
        for hotkey in new_hotkeys:
            if hotkey in elimination_hotkeys:
                continue

            if hotkey in maincomp_hotkeys or hotkey in probation_hotkeys:
                continue

            first_order_time = hk_to_first_order_time.get(hotkey)
            if first_order_time is None:
                if hotkey not in self.active_miners:
                    self.active_miners[hotkey] = (MinerBucket.CHALLENGE, default_time)
                    bt.logging.info(f"Adding {hotkey} to challenge period with start time {default_time}")
                    any_changes = True
                continue

            # Has a first order time but not yet stored in memory
            # Has a first order time but start time is set as default
            if hotkey not in self.active_miners or self.active_miners[hotkey][1] != first_order_time:
                self.active_miners[hotkey] = (MinerBucket.CHALLENGE, first_order_time)
                bt.logging.info(f"Adding {hotkey} to challenge period with first order time {first_order_time}")
                any_changes = True

        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def _refresh_challengeperiod_start_time(self, hk_to_first_order_time_ms: dict[str, int]):
        """
        retroactively update the challengeperiod_testing start time based on time of first order.
        used when a miner is un-eliminated, and positions are preserved.
        """
        bt.logging.info("Refreshing challengeperiod start times")

        any_changes = False
        for hotkey in self.get_testing_miners().keys():
            start_time_ms = self.active_miners[hotkey][1]
            if hotkey not in hk_to_first_order_time_ms:
                bt.logging.warning(f"Hotkey {hotkey} in challenge period has no first order time. Skipping for now.")
                continue
            first_order_time_ms = hk_to_first_order_time_ms[hotkey]

            if start_time_ms != first_order_time_ms:
                bt.logging.info(f"Challengeperiod start time for {hotkey} updated from: {datetime.utcfromtimestamp(start_time_ms/1000)} "
                                f"to: {datetime.utcfromtimestamp(first_order_time_ms/1000)}, {(start_time_ms-first_order_time_ms)/1000}s delta")
                self.active_miners[hotkey] = (MinerBucket.CHALLENGE, first_order_time_ms)
                any_changes = True

        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

        bt.logging.info("All challengeperiod start times up to date")

    def refresh(self, current_time: int):
        if not self.refresh_allowed(ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS):
            time.sleep(1)
            return
        bt.logging.info(f"Refreshing challenge period. invalidation data {self.perf_ledger_manager.perf_ledger_hks_to_invalidate}")
        # The refresh should just read the current eliminations
        eliminations = self.elimination_manager.get_eliminations_from_memory()

        # Collect challenge period and update with new eliminations criteria
        self.remove_eliminated(eliminations=eliminations)

        hk_to_positions, hk_to_first_order_time = self.position_manager.filtered_positions_for_scoring(hotkeys=self.metagraph.hotkeys)

        # challenge period adds to testing if not in eliminated, already in the challenge period, or in the new eliminations list from disk
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph.hotkeys,
            eliminations=eliminations,
            hk_to_first_order_time=hk_to_first_order_time,
            default_time=current_time
        )

        challengeperiod_success_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        challengeperiod_testing_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE)
        challengeperiod_probation_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PROBATION)
        all_miners = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys + challengeperiod_probation_hotkeys

        if not self.refreshed_challengeperiod_start_time:
            self.refreshed_challengeperiod_start_time = True
            self._refresh_challengeperiod_start_time(hk_to_first_order_time)

        ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=all_miners)
        ledger = {hotkey: ledger.get(hotkey, None) for hotkey in all_miners}

        inspection_miners = self.get_testing_miners() | self.get_probation_miners()
        challengeperiod_success, challengeperiod_demoted, challengeperiod_eliminations = self.inspect(
            positions=hk_to_positions,
            ledger=ledger,
            success_hotkeys=challengeperiod_success_hotkeys,
            inspection_hotkeys=inspection_miners,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time
        )
        self.eliminations_with_reasons = challengeperiod_eliminations

        any_changes = bool(challengeperiod_success) or bool(challengeperiod_eliminations) or bool(challengeperiod_demoted)

        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(challengeperiod_success, current_time)
        self._demote_challengeperiod_in_memory(challengeperiod_demoted, current_time)
        self._eliminate_challengeperiod_in_memory(eliminations_with_reasons=challengeperiod_eliminations)

        # Now remove any miners who are no longer in the metagraph
        any_changes |= self._prune_deregistered_metagraph()

        # Now sync challenge period with the disk
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

        self.set_last_update_time()

        bt.logging.info(
            "Challenge Period snapshot after refresh "
            f"(MAINCOMP, {len(self.get_success_miners())}) "
            f"(PROBATION, {len(self.get_probation_miners())}) "
            f"(CHALLENGE, {len(self.get_testing_miners())}) "
        )

    def _prune_deregistered_metagraph(self, hotkeys=None) -> bool:
        """
        Prune the challenge period of all miners who are no longer in the metagraph
        """
        if not hotkeys:
            hotkeys = self.metagraph.hotkeys

        any_changes = False
        for hotkey in list(self.active_miners.keys()):
            if hotkey not in hotkeys:
                del self.active_miners[hotkey]
                any_changes = True

        return any_changes

    @staticmethod
    def is_recently_re_registered(ledger, hotkey, hk_to_first_order_time):
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
        return ans

    def inspect(
        self,
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedger],
        success_hotkeys: list[str],
        inspection_hotkeys: dict[str, int],
        current_time: int,
        success_scores_dict: dict[str, dict] | None = None,
        inspection_scores_dict: dict[str, dict] | None = None,
        hk_to_first_order_time: dict[str, int] | None = None,
    ) -> tuple[list[str], list[str], dict[str, tuple[str, float]]]:
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
        if len(inspection_hotkeys) == 0:
            return [], [], {}  # no hotkeys to inspect

        if not current_time:
            current_time = TimeUtil.now_in_millis()

        eliminate_miners = {}
        miners_recently_reregistered = set()
        miners_not_enough_positions = []

        valid_candidate_hotkeys = []
        for hotkey, bucket_start_time in inspection_hotkeys.items():

            if ChallengePeriodManager.is_recently_re_registered(ledger.get(hotkey), hotkey, hk_to_first_order_time):
                miners_recently_reregistered.add(hotkey)
                continue

            if bucket_start_time is None:
                bt.logging.warning(f'Hotkey {hotkey} has no inspection time. Unexpected.')
                continue

            before_challenge_end = ChallengePeriodManager.meets_time_criteria(current_time, bucket_start_time, self.get_miner_bucket(hotkey))
            if not before_challenge_end:
                bt.logging.info(f'Hotkey {hotkey} has failed the challenge period due to time. cp_failed')
                eliminate_miners[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value, -1)
                continue

            # Get hotkey to positions dict that only includes the inspection miner
            has_minimum_positions, inspection_positions = ChallengePeriodManager.screen_minimum_positions(positions, hotkey)
            if not has_minimum_positions:
                miners_not_enough_positions.append(hotkey)
                continue

            # Get hotkey to ledger dict that only includes the inspection miner
            has_minimum_ledger, inspection_ledger = ChallengePeriodManager.screen_minimum_ledger(ledger, hotkey)
            if not has_minimum_ledger:
                continue

            # This step we want to check their drawdown. If they fail, we can move on.
            ledger_element = inspection_ledger[hotkey]
            exceeds_max_drawdown, recorded_drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element)
            if exceeds_max_drawdown:
                bt.logging.info(f'Hotkey {hotkey} has failed the challenge period due to drawdown {recorded_drawdown_percentage}. cp_failed')
                eliminate_miners[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value, recorded_drawdown_percentage)
                continue

            if not self.screen_minimum_interaction(ledger_element):
                continue

            valid_candidate_hotkeys.append(hotkey)

        candidates_positions = {hotkey: positions[hotkey] for hotkey in valid_candidate_hotkeys}
        candidates_ledgers = {hotkey: ledger[hotkey] for hotkey in valid_candidate_hotkeys}

        # If success_scoring_dict is already calculated, no need to calculate scores. Useful for testing
        if not success_scores_dict:
            success_positions = {hotkey: miner_positions for hotkey, miner_positions in positions.items() if hotkey in success_hotkeys}
            success_ledger = {hotkey: ledger_data for hotkey, ledger_data in ledger.items() if hotkey in success_hotkeys}

            # Get the penalized scores of all successful miners
            success_scores_dict = Scoring.score_miners(ledger_dict=success_ledger,
                                                       positions=success_positions,
                                                       evaluation_time_ms=current_time,
                                                       weighting=True)
        if not inspection_scores_dict:
            inspection_scores_dict = Scoring.score_miners(ledger_dict=candidates_ledgers,
                                                          positions=candidates_positions,
                                                          evaluation_time_ms=current_time,
                                                          weighting=True)

        hotkeys_to_promote, hotkeys_to_demote = ChallengePeriodManager.evaluate_promotions(success_hotkeys,
                                                                                           success_scores_dict,
                                                                                           valid_candidate_hotkeys,
                                                                                           inspection_scores_dict)

        bt.logging.info(f"Challenge Period {len(inspection_hotkeys)} inspected miners")
        bt.logging.info(f"Hotkeys to promote: {hotkeys_to_promote}")
        bt.logging.info(f"Hotkeys to demote: {hotkeys_to_demote}")
        bt.logging.info(f"Hotkeys to eliminate: {list(eliminate_miners.keys())}")
        bt.logging.info(f"Miners with no positions (skipped): {len(miners_not_enough_positions)}")
        bt.logging.info(f"Miners recently re-registered (skipped): {list(miners_recently_reregistered)}")

        return hotkeys_to_promote, hotkeys_to_demote, eliminate_miners

    @staticmethod
    def evaluate_promotions(
            success_hotkeys,
            success_scores_dict,
            candidate_hotkeys,
            inspection_scores_dict,
            threshold_rank=ValiConfig.PROMOTION_THRESHOLD_RANK
            ) -> tuple[list[str], list[str]]:
        combined_scores_dict = copy.deepcopy(success_scores_dict)
        for metric_name, config in combined_scores_dict["metrics"].items():
            candidate_metric_score = inspection_scores_dict["metrics"][metric_name]["scores"]
            miner_scores = config["scores"] + candidate_metric_score
            combined_scores_dict["metrics"][metric_name]["scores"] = miner_scores
        combined_scores_dict["penalties"].update(inspection_scores_dict["penalties"])

        all_scores = Scoring.combine_scores(combined_scores_dict)
        sorted_scores = sorted(all_scores.values(), reverse=True)

        promote_hotkeys = []
        demote_hotkeys = []

        if len(sorted_scores) < threshold_rank:
            for hotkey in all_scores.keys():
                if hotkey in candidate_hotkeys:
                    promote_hotkeys.append(hotkey)
        else:
            threshold_idx = threshold_rank - 1
            threshold_score = sorted_scores[threshold_idx]

            for hotkey, score in all_scores.items():
                if hotkey in candidate_hotkeys and score >= threshold_score:
                    promote_hotkeys.append(hotkey)
                elif hotkey in success_hotkeys and score < threshold_score:
                    demote_hotkeys.append(hotkey)

        return promote_hotkeys, demote_hotkeys

    @staticmethod
    def screen_minimum_interaction(
            ledger_element: PerfLedger
    ) -> bool:
        """
        Returns False if the miner doesn't have the minimum number of trading days.
        """
        if ledger_element is None:
            bt.logging.warning("Ledger element is None. Returning False.")
            return False

        miner_returns = LedgerUtils.daily_return_log(ledger_element)
        return len(miner_returns) >= ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS.value()

    @staticmethod
    def meets_time_criteria(current_time, bucket_start_time, bucket):
        if bucket == MinerBucket.MAINCOMP:
            return False

        elapsed_time_ms = current_time - bucket_start_time
        if bucket == MinerBucket.CHALLENGE:
            upper_bound = ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
            return elapsed_time_ms <= upper_bound

        if bucket == MinerBucket.PROBATION:
            return elapsed_time_ms <= ValiConfig.PROBATION_MAXIMUM_MS

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


    def sync_challenge_period_data(self, active_miners_sync):
        if not active_miners_sync:
            bt.logging.error(f'challenge_period_data {active_miners_sync} appears invalid')

        synced_miners = self.parse_checkpoint_dict(active_miners_sync)

        self.active_miners.clear()
        self.active_miners.update(synced_miners)
        self._write_challengeperiod_from_memory_to_disk()

    def get_hotkeys_by_bucket(self, bucket: MinerBucket) -> list[str]:
        return [hotkey for hotkey, (b, _) in self.active_miners.items() if b == bucket]

    def _remove_eliminated_from_memory(self, eliminations: list[dict] = None) -> bool:
        if eliminations is None:
            eliminations_hotkeys = self.elimination_manager.get_eliminated_hotkeys()
        else:
            eliminations_hotkeys = set([x['hotkey'] for x in eliminations])

        any_changes = False
        for hotkey in eliminations_hotkeys:
            if hotkey in self.active_miners:
                del self.active_miners[hotkey]
                any_changes = True

        return any_changes

    def remove_eliminated(self, eliminations=None):
        if eliminations is None:
            eliminations = []

        any_changes = self._remove_eliminated_from_memory(eliminations=eliminations)
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def _clear_challengeperiod_in_memory_and_disk(self):
        self.active_miners.clear()
        self._write_challengeperiod_from_memory_to_disk()

    def _promote_challengeperiod_in_memory(self, hotkeys: list[str], current_time: int):
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting {len(hotkeys)} miners to main competition.")

        for hotkey in hotkeys:
            bt.logging.info(f"Promoting {hotkey} from {self.get_miner_bucket(hotkey).value} to MAINCOMP")
            self.active_miners[hotkey] = (MinerBucket.MAINCOMP, current_time)

    def _eliminate_challengeperiod_in_memory(self, eliminations_with_reasons: dict[str, tuple[str, float]]):
        hotkeys = eliminations_with_reasons.keys()
        if hotkeys:
            bt.logging.info(f"Removing {len(hotkeys)} hotkeys from challenge period.")

        for hotkey in hotkeys:
            if hotkey in self.active_miners:
                bt.logging.info(f"Eliminating {hotkey}")
                del self.active_miners[hotkey]
            else:
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but demotion to failure was attempted.")

    def _demote_challengeperiod_in_memory(self, hotkeys: list[str], current_time):
        if hotkeys:
            bt.logging.info(f"Demoting {len(hotkeys)} miners to probation")

        for hotkey in hotkeys:
            bt.logging.info(f"Demoting {hotkey} to PROBATION")
            self.active_miners[hotkey] = (MinerBucket.PROBATION, current_time)

    def _write_challengeperiod_from_memory_to_disk(self):
        if self.is_backtesting:
            return
        challengeperiod_data = self.to_checkpoint_dict()
        ValiBkpUtils.write_file(self.CHALLENGE_FILE, challengeperiod_data)

    def get_miner_bucket(self, hotkey): return self.active_miners[hotkey][0]
    def get_testing_miners(self):   return copy.deepcopy(self._bucket_view(MinerBucket.CHALLENGE))
    def get_success_miners(self):   return copy.deepcopy(self._bucket_view(MinerBucket.MAINCOMP))
    def get_probation_miners(self): return copy.deepcopy(self._bucket_view(MinerBucket.PROBATION))

    def _bucket_view(self, bucket: MinerBucket):
        return {hk: ts for hk, (b, ts) in self.active_miners.items() if b == bucket}

    def to_checkpoint_dict(self):
        snapshot = list(self.active_miners.items())
        json_dict = {
            hotkey: {
                "bucket": bucket.value,
                "bucket_start_time": start_time
            }
            for hotkey, (bucket, start_time) in snapshot
        }
        return json_dict

    @staticmethod
    def parse_checkpoint_dict(json_dict):
        formatted_dict = {}

        if "testing" in json_dict.keys() and "success" in json_dict.keys():
            testing = json_dict.get("testing", {})
            success = json_dict.get("success", {})
            for hotkey, start_time in testing.items():
                formatted_dict[hotkey] = (MinerBucket.CHALLENGE, start_time)
            for hotkey, start_time in success.items():
                formatted_dict[hotkey] = (MinerBucket.MAINCOMP, start_time)

        else:
            for hotkey, info in json_dict.items():
                bucket = MinerBucket(info["bucket"])
                bucket_start_time = info["bucket_start_time"]

                formatted_dict[hotkey] = (bucket, bucket_start_time)

        return formatted_dict

