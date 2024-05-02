# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import copy
import time
from typing import List, Union
import numpy as np

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.position import Position
from vali_objects.scoring.scoring import Scoring
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfCheckpoint, PerfLedger

class SubtensorWeightSetter(CacheController):
    def __init__(self, config, wallet, metagraph, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.perf_manager = PerfLedgerManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.wallet = wallet
        self.subnet_version = 200

    def set_weights(self):
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return

        bt.logging.info("running set weights")
        ## First run the challenge period miner filtering
        current_time = TimeUtil.now_in_millis()

        ## we want to do this first because we will add to the eliminations list
        self._refresh_eliminations_in_memory()
        challengeperiod_resultdict = self.challenge_period_screening(
            current_time = current_time
        )

        challengeperiod_miners = challengeperiod_resultdict["challengeperiod_miners"]
        challengeperiod_elimination_hotkeys = challengeperiod_resultdict["challengeperiod_eliminations"]

        # augmented ledger should have the gain, loss, n_updates, and time_duration
        augmented_ledger = self.augmented_ledger(
            omitted_miners = challengeperiod_miners + challengeperiod_elimination_hotkeys,
            eliminations = self.eliminations
        )

        bt.logging.trace(f"return per uid [{augmented_ledger}]")
        bt.logging.info(f"number of returns for uid: {len(augmented_ledger)}")
        if len(augmented_ledger) == 0:
            bt.logging.info("no returns to set weights with. Do nothing for now.")
        else:
            bt.logging.info("calculating new subtensor weights...")
            checkpoint_results = Scoring.compute_results_checkpoint(augmented_ledger)
            bt.logging.info(f"sorted results for weight setting: [{sorted(checkpoint_results, key=lambda x: x[1], reverse=True)}]")

            checkpoint_netuid_weights = []
            for miner, score in checkpoint_results:
                if miner in self.metagraph.hotkeys:
                    checkpoint_netuid_weights.append((
                        self.metagraph.hotkeys.index(miner),
                        score
                    ))
                else:
                    bt.logging.error(f"Miner {miner} not found in the metagraph.")

            challengeperiod_weights = []
            for miner in challengeperiod_miners:
                if miner in self.metagraph.hotkeys:
                    challengeperiod_weights.append((
                        self.metagraph.hotkeys.index(miner),
                        ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_WEIGHT
                    ))
                else:
                    bt.logging.error(f"Challengeperiod miner {miner} not found in the metagraph.")

            transformed_list = checkpoint_netuid_weights + challengeperiod_weights
            bt.logging.info(f"transformed list: {transformed_list}")

            self._set_subtensor_weights(transformed_list)
        self.set_last_update_time()

    def challenge_period_screening(
        self,
        hotkeys: List[str] = None,
        current_time: int = None,
        eliminations: List[str] = None,
        log: bool = False,
    ):
        """
        Runs a screening process to elminate miners who didn't pass the challenge period.
        """
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        if eliminations is not None:
            self.eliminations = eliminations

        # Get all possible positions, even beyond the lookback range
        full_hotkey_positions = self.position_manager.get_all_miner_positions_by_hotkey(
            hotkeys,
            sort_positions=True,
            eliminations=self.eliminations,
            acceptable_position_end_ms=None,
        )

        challengeperiod_miners = []
        challengeperiod_eliminations = []

        for hotkey, positions in full_hotkey_positions.items():
            challenge_check_logic = self._challengeperiod_check(
                positions, 
                current_time,
                log=log
            )
            if challenge_check_logic is None:
                challengeperiod_miners.append(hotkey)
                continue
            
            if challenge_check_logic == False:
                challengeperiod_eliminations.append(hotkey)

        if challengeperiod_eliminations:
            bt.logging.info(
                f"Miners {challengeperiod_eliminations} failed the challenge period - weight will be set to 0."
            )

        # update for new eliminations due to challenge period
        return {
            "challengeperiod_miners": challengeperiod_miners,
            "challengeperiod_eliminations": challengeperiod_eliminations,
        }

    def augmented_ledger(
        self,
        local: bool = False,
        hotkeys: List[str] = None,
        eliminations: List[str] = None,
        omitted_miners: List[str] = None,
        evaluation_time_ms: int = None,
    ) -> PerfLedger:
        """
        Calculate all returns for the .
        """
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        if eliminations is None:
            eliminations = self.eliminations

        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        # Note, eliminated miners will not appear in the dict below
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()
        ledger = self.perf_manager.load_perf_ledgers_from_disk()

        augmented_ledger = {}
        for hotkey, miner_ledger in ledger.items():
            if hotkey not in hotkeys:
                continue

            if omitted_miners is not None and hotkey in omitted_miners:
                continue

            if hotkey in eliminated_hotkeys:
                continue

            miner_checkpoints = copy.deepcopy(miner_ledger.cps)
            miner_checkpoints_filtered = self._filter_checkpoint_elements(miner_checkpoints)
            checkpoint_meets_criteria = self._filter_checkpoint_list(miner_checkpoints_filtered)
            if not checkpoint_meets_criteria:
                continue

            augmented_ledger[hotkey] = miner_ledger
            augmented_ledger[hotkey].cps = self.position_manager.augment_perf_checkpoint(
                miner_checkpoints_filtered,
                evaluation_time_ms
            )

        return augmented_ledger
        
    def _challengeperiod_returns_logic(
            self, 
            challengeperiod_subset_positions: list[Position],
            log:bool = False
        ) -> bool:
        challengeperiod_returns = [ np.log(x.return_at_close) for x in challengeperiod_subset_positions ]
        minimum_return = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_RETURN
        minimum_nreturns = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_NRETURNS
        minimum_total_position_duration = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_TOTAL_POSITION_DURATION
        minimum_mrad = ValiConfig.SET_WEIGHT_MINER_CHALLENGE_PERIOD_MRAD

        challengeperiod_nreturns = len(challengeperiod_subset_positions)
        challengeperiod_return = Scoring.total_return(challengeperiod_returns)
        challengeperiod_total_position_duration = PositionUtils.compute_total_position_duration(challengeperiod_subset_positions)
        challengeperiod_mrad = Scoring.mad_variation(challengeperiod_returns)

        return_logic = challengeperiod_return >= minimum_return
        nreturns_logic = challengeperiod_nreturns >= minimum_nreturns
        total_position_duration_logic = challengeperiod_total_position_duration >= minimum_total_position_duration
        coefficient_of_variation_logic = challengeperiod_mrad <= minimum_mrad

        challengeperiod_passing = (
            coefficient_of_variation_logic and return_logic and nreturns_logic and total_position_duration_logic
        )

        if log:
            bt.logging.info(f"Miner {challengeperiod_subset_positions[0].miner_hotkey} challenge period - {challengeperiod_passing}.")
            bt.logging.info(f"Return Logic:\t\t\t{return_logic} - {round(challengeperiod_return, 3)} >= {minimum_return}")
            bt.logging.info(f"N Returns Logic:\t\t\t{nreturns_logic} - {challengeperiod_nreturns} >= {minimum_nreturns}")
            bt.logging.info(f"Total Position Duration Logic:\t{total_position_duration_logic} - {round(challengeperiod_total_position_duration / 1e9, 3)} >= {round(minimum_total_position_duration / 1e9, 3)}")
            bt.logging.info(f"MRAD Logic:\t\t\t{coefficient_of_variation_logic} - {round(challengeperiod_mrad, 3)} <= {minimum_mrad}\n")

        return challengeperiod_passing
    
    def _challengeperiod_check(
            self, 
            positions: list[Position], 
            current_time: int,
            log: bool = False
        ) -> Union[bool, None]:
        """
        Check if the miner is within the grace period:
        - positions: list[Position] - the list of positions
        - current_time: int - the current time of evaluation in milliseconds
        """
        if len(positions) == 0:
            return False

        # check for the first closed position
        first_position_time = current_time
        challengeperiod_time_ms = ValiConfig.SET_WEIGHT_CHALLENGE_PERIOD_MS
        challengeperiod_minimum_positions = ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS
                
        for i in range(len(positions)):
            # capture the moment the first position was opened
            first_position_time = min(first_position_time, positions[i].open_ms)

        time_criteria = (current_time - first_position_time) < challengeperiod_time_ms
        if log:
            bt.logging.info(f"Miner {positions[0].miner_hotkey} first position opened at {TimeUtil.millis_to_timestamp(first_position_time)}.")
            bt.logging.info(f"Miner {positions[0].miner_hotkey} challenge period ends at {TimeUtil.millis_to_timestamp(first_position_time + challengeperiod_time_ms)}.")
            bt.logging.info(f"Miner {positions[0].miner_hotkey} time criteria: {time_criteria} - {round((current_time - first_position_time) / 1e9, 3)} < {round(challengeperiod_time_ms / 1e9, 3)}.\n")

        challengeperiod_positions = []
        challengeperiod_start = first_position_time
        challengeperiod_end = first_position_time + challengeperiod_time_ms

        for position in positions:
            if position.is_closed_position and (challengeperiod_start <= position.close_ms < challengeperiod_end) and position.return_at_close > 0.0:
                challengeperiod_positions.append(position)

        # check if the miner has enough positions to potentially pass the challenge period, if not just put them in challengeperiod now
        if len(challengeperiod_positions) < challengeperiod_minimum_positions and time_criteria:
            return None

        for i in range(challengeperiod_minimum_positions, len(challengeperiod_positions) + 1):
            challengeperiod_subset_positions = challengeperiod_positions[:i]
            challengeperiod_passing = self._challengeperiod_returns_logic(
                challengeperiod_subset_positions=challengeperiod_subset_positions,
                log=log
            )

            if challengeperiod_passing:
                # if at any point the miner is passing the competition, they are good to go
                return True
        
        # If they don't have a passing position but the challenge period is not over, return None - they are still in the challenge period
        if time_criteria:
            return None
        
        # if they have not passed the challenge period, return False
        return False
    
    def _filter_checkpoint_elements(self, checkpoints: list[PerfCheckpoint]) -> list[PerfCheckpoint]:
        """
        Filter checkpoint elements if they don't meet minimum evaluation criteria.
        """
        checkpoint_filtered = []

        for checkpoint in checkpoints:
            if checkpoint.open_ms >= ValiConfig.SET_WEIGHT_MINIMUM_SINGLE_CHECKPOINT_DURATION_MS:
                checkpoint_filtered.append(checkpoint)

        return checkpoint_filtered

    
    def _filter_checkpoint_list(self, checkpoints: list[PerfCheckpoint]):
        """
        Filter out miners based on a minimum total duration of interaction with the system.
        """
        if len(checkpoints) == 0:
            return False
        
        total_checkpoint_duration = 0

        for checkpoint in checkpoints:
            total_checkpoint_duration += checkpoint.open_ms

        if total_checkpoint_duration < ValiConfig.SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS:
            return False

        return True

    def _filter_miner(self, positions: list[Position], current_time: int):
        """
        Filter out miners who don't have enough positions to be considered for setting weights - True means that we want to filter the miner out
        """
        if len(positions) == 0:
            return True

        # find the time when the first position was opened
        first_closed_position_ms = positions[0].close_ms
        for i in range(1, len(positions)):
            first_closed_position_ms = min(
                first_closed_position_ms, positions[i].close_ms
            )

        closed_positions = len(
            [position for position in positions if position.is_closed_position]
        )

        if closed_positions < ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
            return True

        # check that the position
        return False

    def _filter_positions(self, positions: list[Position]):
        """
        Filter out positions that are not within the lookback range.
        """
        filtered_positions = []
        for position in positions:
            if not position.is_closed_position:
                continue

            if position.close_ms - position.open_ms < ValiConfig.SET_WEIGHT_MINIMUM_POSITION_DURATION_MS:
                continue

            filtered_positions.append(position)
        return filtered_positions

    def _set_subtensor_weights(self, filtered_results: list[tuple[str, float]]):
        filtered_netuids = [x[0] for x in filtered_results]
        scaled_transformed_list = [x[1] for x in filtered_results]

        success, err_msg = self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=filtered_netuids,
            weights=scaled_transformed_list,
            version_key=self.subnet_version,
        )

        if success:
            bt.logging.success("Successfully set weights.")
        else:
            bt.logging.error(f"Failed to set weights. Error message: {err_msg}")
