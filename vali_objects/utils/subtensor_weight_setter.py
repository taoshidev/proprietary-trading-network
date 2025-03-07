# developer: jbonilla
from functools import partial

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.scoring.scoring import Scoring

class SubtensorWeightSetter(CacheController):
    def __init__(self, metagraph, position_manager: PositionManager,
                 running_unit_tests=False, is_backtesting=False):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.subnet_version = 200
        # Store weights for use in backtesting
        self.checkpoint_results = []
        self.transformed_list = []

    def compute_weights_default(self, current_time: int) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        # Collect metagraph hotkeys to ensure we are only setting weights for miners in the metagraph
        metagraph_hotkeys = list(self.metagraph.hotkeys)
        hotkey_to_idx = {hotkey: idx for idx, hotkey in enumerate(metagraph_hotkeys)}
        idx_to_hotkey = {idx: hotkey for idx, hotkey in enumerate(metagraph_hotkeys)}
        hotkey_registration_blocks = list(self.metagraph.block_at_registration)
        target_dtao_block_zero_incentive_start = 4916273
        target_dtao_block_zero_incentive_end = 4951874

        block_reg_failures = set()

        # augmented ledger should have the gain, loss, n_updates, and time_duration
        testing_hotkeys = list(self.position_manager.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.position_manager.challengeperiod_manager.challengeperiod_success.keys())

        if self.is_backtesting:
            hotkeys_to_compute_weights_for = testing_hotkeys + success_hotkeys
        else:
            hotkeys_to_compute_weights_for = success_hotkeys
        # only collect ledger elements for the miners that passed the challenge period
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=hotkeys_to_compute_weights_for)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(hotkeys=hotkeys_to_compute_weights_for)

        if len(filtered_ledger) == 0:
            bt.logging.info("No returns to set weights with. Do nothing for now.")
            return [], []
        else:
            bt.logging.info("Calculating new subtensor weights...")
            checkpoint_results = sorted(Scoring.compute_results_checkpoint(
                filtered_ledger,
                filtered_positions,
                evaluation_time_ms=current_time,
                weighting=True
            ), key=lambda x: x[1], reverse=True)

            bt.logging.info(f"Sorted results for weight setting: [{checkpoint_results}]")

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
            self.handle_block_reg_failures(transformed_list, target_dtao_block_zero_incentive_start, hotkey_registration_blocks, idx_to_hotkey, target_dtao_block_zero_incentive_end, block_reg_failures)
            bt.logging.info(f"transformed list: {transformed_list}")
            if block_reg_failures:
                bt.logging.info(f"Miners with registration blocks outside of permissible dTAO blocks: {block_reg_failures}")

            return checkpoint_results, transformed_list
    def _store_weights(self, checkpoint_results: list[tuple[str, float]], transformed_list: list[tuple[str, float]]):
        self.checkpoint_results = checkpoint_results
        self.transformed_list = transformed_list

    def handle_block_reg_failures(self, transformed_list, target_dtao_block_zero_incentive_start, hotkey_registration_blocks,
                                  idx_to_hotkey, target_dtao_block_zero_incentive_end, block_reg_failures):
        if self.is_backtesting:
            return
        for tl_idx, (metagraph_idx, score) in enumerate(transformed_list):
            if target_dtao_block_zero_incentive_start < hotkey_registration_blocks[metagraph_idx] <= target_dtao_block_zero_incentive_end:
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

    def set_weights(self, wallet, netuid, subtensor, current_time: int = None, scoring_function: callable = None, scoring_func_args: dict = None):
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return

        bt.logging.info("running set weights")
        if scoring_func_args is None:
            scoring_func_args = {'current_time': current_time}

        if scoring_function is None:
            scoring_function = self.compute_weights_default  # Uses instance method
        elif not hasattr(scoring_function, '__self__'):
            scoring_function = partial(scoring_function, self)  # Only bind if external

        checkpoint_results, transformed_list = scoring_function(**scoring_func_args)
        self.checkpoint_results = checkpoint_results
        self.transformed_list = transformed_list
        if not self.is_backtesting:
            self._set_subtensor_weights(wallet, subtensor, netuid)
        self.set_last_update_time()


    def _set_subtensor_weights(self, wallet, subtensor, netuid):
        filtered_netuids = [x[0] for x in self.transformed_list]
        scaled_transformed_list = [x[1] for x in self.transformed_list]

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
            bt.logging.warning(f"Failed to set weights. Error message: {err_msg}")
