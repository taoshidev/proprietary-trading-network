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
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
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

    def set_weights(self, current_time: int = None):
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return

        bt.logging.info("running set weights")
        ## First run the challenge period miner filtering
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        ## we want to do this first because we will add to the eliminations list
        self._refresh_eliminations_in_memory()
        self._refresh_challengeperiod_in_memory()

        # augmented ledger should have the gain, loss, n_updates, and time_duration
        testing_hotkeys = list(self.challengeperiod_testing.keys())
        success_hotkeys = list(self.challengeperiod_success.keys())

        # only collect ledger elements for the miners that passed the challenge period
        filtered_ledger = self.filtered_ledger(hotkeys=success_hotkeys)

        if len(filtered_ledger) == 0:
            bt.logging.info("No returns to set weights with. Do nothing for now.")
        else:
            bt.logging.info("Calculating new subtensor weights...")
            checkpoint_results = Scoring.compute_results_checkpoint(
                filtered_ledger,
                evaluation_time_ms=current_time
            )
            bt.logging.info(f"Sorted results for weight setting: [{sorted(checkpoint_results, key=lambda x: x[1], reverse=True)}]")

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
            for miner in testing_hotkeys:
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

    def filtered_ledger(
        self,
        hotkeys: List[str] = None
    ) -> PerfLedger:
        """
        Filter the ledger for a set of hotkeys.
        """
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        # Note, eliminated miners will not appear in the dict below
        ledger = self.perf_manager.load_perf_ledgers_from_disk()

        augmented_ledger = {}
        for hotkey, miner_ledger in ledger.items():
            if hotkey not in hotkeys:
                continue

            miner_checkpoints = copy.deepcopy(miner_ledger.cps)
            miner_checkpoints_filtered = self._filter_checkpoint_elements(miner_checkpoints)
            checkpoint_meets_criteria = self._filter_checkpoint_list(miner_checkpoints_filtered)
            if not checkpoint_meets_criteria:
                continue

            augmented_ledger[hotkey] = miner_ledger

        return augmented_ledger

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

        return augmented_ledger
    
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
