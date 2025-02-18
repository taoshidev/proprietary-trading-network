# developer: jbonilla
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.position import Position
from vali_objects.scoring.scoring import Scoring
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger


class SubtensorWeightSetter(CacheController):
    def __init__(self, config, metagraph, position_manager: PositionManager,
                 running_unit_tests=False):
        super().__init__(metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.subnet_version = 200

    def set_weights(self, wallet, netuid, subtensor, current_time: int = None):
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return
        bt.logging.info("running set weights")
        # First run the challenge period miner filtering
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        # Collect metagraph hotkeys to ensure we are only setting weights for miners in the metagraph
        metagraph_hotkeys = list(self.metagraph.hotkeys)
        hotkey_to_idx = {hotkey: idx for idx, hotkey in enumerate(metagraph_hotkeys)}
        idx_to_hotkey = {idx: hotkey for idx, hotkey in enumerate(metagraph_hotkeys)}
        hotkey_registration_blocks = list(self.metagraph.block_at_registration)
        target_dtao_block = 4941752
        block_reg_failures = set()

        # augmented ledger should have the gain, loss, n_updates, and time_duration
        testing_hotkeys = list(self.position_manager.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.position_manager.challengeperiod_manager.challengeperiod_success.keys())

        # only collect ledger elements for the miners that passed the challenge period
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=success_hotkeys)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(hotkeys=success_hotkeys)

        # synced_ledger, synced_positions = self.sync_ledger_positions(
        #     filtered_ledger,
        #     filtered_positions
        # )

        if len(filtered_ledger) == 0:
            bt.logging.info("No returns to set weights with. Do nothing for now.")
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
                if miner in hotkey_to_idx:
                    checkpoint_netuid_weights.append((
                        hotkey_to_idx[miner],
                        score
                    ))
                else:
                    bt.logging.error(f"Miner {miner} not found in the metagraph.")

            challengeperiod_weights = []
            for miner in testing_hotkeys:
                if miner in hotkey_to_idx:
                    challengeperiod_weights.append((
                        hotkey_to_idx[miner],
                        ValiConfig.CHALLENGE_PERIOD_WEIGHT
                    ))
                else:
                    bt.logging.error(f"Challengeperiod miner {miner} not found in the metagraph.")

            transformed_list = checkpoint_netuid_weights + challengeperiod_weights
            for tl_idx, (metagraph_idx, score) in enumerate(transformed_list):
                if hotkey_registration_blocks[metagraph_idx] > target_dtao_block:
                    try:
                        block_reg_failures.add(idx_to_hotkey[metagraph_idx])
                        transformed_list[tl_idx] = (metagraph_idx, 0.0)
                    except Exception as e:
                        warning_str = (f"metagraph_idx {metagraph_idx} ({idx_to_hotkey.get(metagraph_idx)}),"
                                       f" hotkey_registration_blocks {hotkey_registration_blocks} ({len(hotkey_registration_blocks)}),"
                                       f" block_reg_failures {block_reg_failures}, "
                                       f"idx_to_hotkey {idx_to_hotkey} ({len(idx_to_hotkey)}), ")
                        bt.logging.warning(warning_str)
                        raise e
            bt.logging.info(f"transformed list: {transformed_list}")
            if block_reg_failures:
                bt.logging.info(f"Miners with registration blocks after target DTAO block: {block_reg_failures}")

            self._set_subtensor_weights(wallet, subtensor, transformed_list, netuid)
        self.set_last_update_time()

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

    def _set_subtensor_weights(self, wallet, subtensor, filtered_results: list[tuple[str, float]], netuid):
        filtered_netuids = [x[0] for x in filtered_results]
        scaled_transformed_list = [x[1] for x in filtered_results]

        success, err_msg = subtensor.set_weights(
            netuid=netuid,
            wallet=wallet,
            uids=filtered_netuids,
            weights=scaled_transformed_list,
            version_key=self.subnet_version,
        )

        if success:
            bt.logging.success("Successfully set weights.")
        else:
            bt.logging.error(f"Failed to set weights. Error message: {err_msg}")
