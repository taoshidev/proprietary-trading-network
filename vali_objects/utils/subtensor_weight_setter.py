# developer: jbonilla
import copy
from typing import List

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.position import Position
from vali_objects.scoring.scoring import Scoring
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfCheckpoint, PerfLedger, PerfLedgerData


class SubtensorWeightSetter(CacheController):
    def __init__(self, config, wallet, metagraph, running_unit_tests=False, perf_manager=None, running_backtesting=True):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        if perf_manager is None:
            perf_manager = PerfLedgerManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.perf_manager = perf_manager
        self.wallet = wallet
        self.subnet_version = 200
        self.running_backtesting = running_backtesting

    def set_weights(self, current_time: int = None,
                    filtered_positions: dict[str, list[Position]] = None,
                    filtered_ledger: dict[str, PerfLedgerData] = None,
                    challenge_period_success_hotkeys: List[str] = None,
                    challenge_period_testing_hotkeys: List[str] = None):

        if not self.running_backtesting and not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return

        hotkey_to_weight = {}
        bt.logging.info("running set weights")
        # First run the challenge period miner filtering
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        # Collect metagraph hotkeys to ensure we are only setting weights for miners in the metagraph
        metagraph_hotkeys = self.metagraph.hotkeys

        # we want to do this first because we will add to the eliminations list
        if not self.running_backtesting:
            # When running in backtesting, eliminations are always passed in from the metagraph
            self._refresh_eliminations_in_memory()

            self._refresh_challengeperiod_in_memory()
            # augmented ledger should have the gain, loss, n_updates, and time_duration
            challenge_period_testing_hotkeys = list(self.challengeperiod_testing.keys())
            challenge_period_success_hotkeys = list(self.challengeperiod_success.keys())
            filtered_ledger = self.filtered_ledger(hotkeys=challenge_period_success_hotkeys)
            filtered_positions = self.filtered_positions(hotkeys=challenge_period_success_hotkeys)
        else:
            filtered_ledger = {k:v for k,v in filtered_ledger.items() if k in challenge_period_success_hotkeys}
            filtered_positions = {k:v for k,v in filtered_positions.items() if k in challenge_period_success_hotkeys}

        # synced_ledger, synced_positions = self.sync_ledger_positions(
        #     filtered_ledger,
        #     filtered_positions
        # )

        if not self.running_backtesting and len(filtered_ledger) == 0:
            bt.logging.info("No returns to set weights with. Do nothing for now.")
            return []
        else:
            bt.logging.info("Calculating new subtensor weights...")
            checkpoint_results = Scoring.compute_results_checkpoint(
                filtered_ledger,
                filtered_positions,
                evaluation_time_ms=current_time
            )
            bt.logging.info(f"Sorted results for weight setting: [{sorted(checkpoint_results, key=lambda x: x[1], reverse=True)}]")

            checkpoint_netuid_weights = []
            for miner, score in checkpoint_results:
                if miner in metagraph_hotkeys:
                    hotkey_to_weight[miner] = score
                    checkpoint_netuid_weights.append((
                        metagraph_hotkeys.index(miner),
                        score
                    ))
                else:
                    bt.logging.error(f"Miner {miner} not found in the metagraph {self.metagraph.hotkeys}.")

            challengeperiod_weights = []
            for miner in challenge_period_testing_hotkeys:
                if miner in metagraph_hotkeys:
                    hotkey_to_weight[miner] = ValiConfig.CHALLENGE_PERIOD_WEIGHT
                    challengeperiod_weights.append((
                        metagraph_hotkeys.index(miner),
                        ValiConfig.CHALLENGE_PERIOD_WEIGHT
                    ))
                else:
                    bt.logging.error(f"Challengeperiod miner {miner} not found in the metagraph.")

            self.set_last_update_time()
            if self.running_backtesting:
                bt.logging.info(f"hotkey to weight: {hotkey_to_weight}")
                return hotkey_to_weight
            else:
                transformed_list = checkpoint_netuid_weights + challengeperiod_weights
                bt.logging.info(f"transformed list: {transformed_list}")
                self._set_subtensor_weights(transformed_list)
                return transformed_list


    @staticmethod
    def sync_ledger_positions(
            ledger,
            positions
    ) -> tuple[dict[str, PerfLedger], dict[str, list[Position]]]:
        """
        Sync the ledger and positions to ensure that the ledger and positions are in the same state.
        """
        ledger_keys = set(ledger.keys())
        position_keys = set(positions.keys())

        common_keys = ledger_keys.intersection(position_keys)
        # uncommon_keys = ledger_keys.union(position_keys) - common_keys

        synced_ledger = {}
        synced_positions = {}

        for hotkey in common_keys:
            synced_ledger[hotkey] = ledger[hotkey]
            synced_positions[hotkey] = positions[hotkey]

        # if len(uncommon_keys) > 0:
        #     for hotkey in uncommon_keys:
        #         if hotkey in ledger_keys:
        #             bt.logging.warning(f"Hotkey found in ledger but not positions: {hotkey}")
        #         elif hotkey in position_keys:
        #             bt.logging.warning(f"Hotkey found in positions but not ledger: {hotkey}")

        return synced_ledger, synced_positions

    def filtered_ledger(
            self,
            hotkeys: List[str] = None
    ) -> dict[str, PerfLedgerData]:
        """
        Filter the ledger for a set of hotkeys.
        """
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        # Note, eliminated miners will not appear in the dict below
        ledger = self.perf_manager.load_perf_ledgers_from_disk()

        filtering_ledger = {}
        for hotkey, miner_ledger in ledger.items():
            if hotkey not in hotkeys:
                continue

            ledger_copy = copy.deepcopy(miner_ledger)
            if not self._filter_checkpoint_list(ledger_copy.cps):
                continue

            filtering_ledger[hotkey] = ledger_copy

        return filtering_ledger

    def _filter_checkpoint_list(self, checkpoints: list[PerfCheckpoint]):
        """
        Filter out miners based on a minimum total duration of interaction with the system.
        """
        if len(checkpoints) == 0:
            return False

        return True

    def filtered_positions(
            self,
            hotkeys: List[str] = None
    ) -> dict[str, list[Position]]:
        """
        Filter the positions for a set of hotkeys.
        """
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys

        positions = self.position_manager.get_all_miner_positions_by_hotkey(
            hotkeys,
            sort_positions=True
        )

        filtering_positions = {}
        for hotkey, miner_positions in positions.items():
            if hotkey not in hotkeys:
                continue

            filtered_positions = self._filter_positions(miner_positions)
            filtering_positions[hotkey] = filtered_positions

        return filtering_positions

    def _filter_positions(self, positions: list[Position]):
        """
        Filter out positions that are not within the lookback range.
        """
        filtered_positions = []
        for position in positions:
            if not position.is_closed_position:
                continue

            if position.close_ms - position.open_ms < ValiConfig.MINIMUM_POSITION_DURATION_MS:
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
