# developer: jbonilla
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.scoring.scoring import Scoring

class SubtensorWeightSetter(CacheController):
    def __init__(self, metagraph, position_manager: PositionManager,
                 running_unit_tests=False):
        super().__init__(metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.subnet_version = 200
        # Store weights for use in backtesting
        self.checkpoint_results = []
        self.transformed_list = []

    def compute_weights_default(self, current_time: int, metagraph_hotkeys: list[int], ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        testing_hotkeys = list(self.position_manager.challengeperiod_manager.challengeperiod_testing.keys())
        success_hotkeys = list(self.position_manager.challengeperiod_manager.challengeperiod_success.keys())

        # only collect ledger elements for the miners that passed the challenge period
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=success_hotkeys)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(hotkeys=success_hotkeys)


        if len(filtered_ledger) == 0:
            bt.logging.info("No returns to set weights with. Do nothing for now.")
            return [], []
        else:
            bt.logging.info("Calculating new subtensor weights...")
            checkpoint_results = sorted(Scoring.compute_results_checkpoint(
                filtered_ledger,
                filtered_positions,
                evaluation_time_ms=current_time
            ), key=lambda x: x[1], reverse=True)

            bt.logging.info(f"Sorted results for weight setting: [{checkpoint_results}]")

            checkpoint_netuid_weights = []
            for miner, score in checkpoint_results:
                if miner in metagraph_hotkeys:
                    checkpoint_netuid_weights.append((
                        metagraph_hotkeys.index(miner),
                        score
                    ))
                else:
                    bt.logging.error(f"Miner {miner} not found in the metagraph.")

            challengeperiod_weights = []
            for miner in testing_hotkeys:
                if miner in metagraph_hotkeys:
                    challengeperiod_weights.append((
                        metagraph_hotkeys.index(miner),
                        ValiConfig.CHALLENGE_PERIOD_WEIGHT
                    ))
                else:
                    bt.logging.error(f"Challengeperiod miner {miner} not found in the metagraph.")

            transformed_list = checkpoint_netuid_weights + challengeperiod_weights
            bt.logging.info(f"transformed list: {transformed_list}")
            return checkpoint_results, transformed_list
    def _store_weights(self, checkpoint_results: list[tuple[str, float]], transformed_list: list[tuple[str, float]]):
        self.checkpoint_results = checkpoint_results
        self.transformed_list = transformed_list

    def set_weights(self, wallet, netuid, subtensor, current_time: int = None, scoring_function: callable = None, scoring_func_args: dict = None):
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return
        bt.logging.info("running set weights")
        if scoring_function is None:
            scoring_function = self.compute_weights_default
            scoring_func_args = {'current_time': current_time,
                                 'metagraph': self.metagraph.hotkeys,
                                 }
        else:
            assert scoring_func_args is not None, "scoring_func_args must be provided if scoring_function is not None."

        checkpoint_results, transformed_list = scoring_function(**scoring_func_args)
        self._store_weights(checkpoint_results, transformed_list)
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
            bt.logging.error(f"Failed to set weights. Error message: {err_msg}")
