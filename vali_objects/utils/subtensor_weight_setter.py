# developer: jbonilla
from functools import partial

import bittensor as bt
import numpy as np

from time_util.time_util import TimeUtil
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.metrics import Metrics
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.scoring.scoring import Scoring

class SubtensorWeightSetter(CacheController):
    def __init__(self, metagraph, position_manager: PositionManager,
                 running_unit_tests=False, is_backtesting=False, is_mainnet=True, live_price_fetcher=None):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.subnet_version = 200
        # Store weights for use in backtesting
        self.checkpoint_results = []
        self.transformed_list = []
        self.is_mainnet = is_mainnet

        secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        if live_price_fetcher is None:
            self.live_price_fetcher = LivePriceFetcher(secrets=secrets)
        else:
            self.live_price_fetcher = live_price_fetcher

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
        _, checkpoint_results = self._compute_miner_weights(hotkeys_to_compute_weights_for, hotkey_to_idx, current_time, scoring_challenge=False)

        # compute burn amount
        tao_price, _ = self.live_price_fetcher.get_latest_price(TradePair.TAOUSD, current_time)
        burn_amt = self.compute_burn_amount(tao_price, checkpoint_results)
        checkpoint_results = Scoring.burn_scores(checkpoint_results, burn_amt, self.is_mainnet)
        checkpoint_netuid_weights = self._compute_miner_netuid_weights(hotkey_to_idx, checkpoint_results)

        if checkpoint_netuid_weights is None or len(checkpoint_netuid_weights) == 0:
            bt.logging.info("No returns to set weights with. Do nothing for now.")
            return [], []

        if self.is_backtesting:
            challengeperiod_weights = []
        else:
            challengeperiod_weights, _ = self._compute_miner_weights(testing_hotkeys, hotkey_to_idx, current_time, scoring_challenge=True)

        transformed_list = checkpoint_netuid_weights + challengeperiod_weights
        self.handle_block_reg_failures(transformed_list, target_dtao_block_zero_incentive_start, hotkey_registration_blocks, idx_to_hotkey, target_dtao_block_zero_incentive_end, block_reg_failures)
        bt.logging.info(f"transformed list: {transformed_list}")
        if block_reg_failures:
            bt.logging.info(f"Miners with registration blocks outside of permissible dTAO blocks: {block_reg_failures}")

        return checkpoint_results, transformed_list

    def _compute_miner_weights(self, hotkeys_to_compute_weights_for, hotkey_to_idx, current_time, scoring_challenge: bool=False):

        miner_group = "challenge period" if scoring_challenge else "main competition"

        # only collect ledger elements for the miners that passed the challenge period
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=hotkeys_to_compute_weights_for)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(
            hotkeys=hotkeys_to_compute_weights_for)

        if len(filtered_ledger) == 0:
            return [], []
        else:
            bt.logging.info(f"Calculating new subtensor weights for {miner_group}...")
            checkpoint_results = sorted(Scoring.compute_results_checkpoint(
                filtered_ledger,
                filtered_positions,
                evaluation_time_ms=current_time,
                weighting=True
            ), key=lambda x: x[1], reverse=True)

            bt.logging.info(f"Sorted results for weight setting for {miner_group}: [{checkpoint_results}]")

            # Extra processing if scoring challenge
            processed_checkpoint_results = checkpoint_results

            if scoring_challenge:
                processed_checkpoint_results = Scoring.score_testing_miners(filtered_ledger, checkpoint_results)

            checkpoint_netuid_weights = self._compute_miner_netuid_weights(hotkey_to_idx, processed_checkpoint_results)
        return checkpoint_netuid_weights, checkpoint_results

    def _compute_miner_netuid_weights(self, hotkey_to_idx, checkpoint_results):
        checkpoint_netuid_weights = []
        for miner, score in checkpoint_results:
            if miner in hotkey_to_idx:
                checkpoint_netuid_weights.append((
                    hotkey_to_idx[miner],
                    score
                ))
            else:
                bt.logging.error(f"Miner {miner} not found in the metagraph.")
        return checkpoint_netuid_weights

    def compute_burn_amount(self, tao_price, checkpoint_results):
        """
        Burn excess emissions based on the realtime price of alpha.

        $ a miner makes per tempo based on returns = (miner_returns_per_day / # num_tempos_per_day) * capital
        $ emitted to a miner per tempo = miner_weight * (emissions_per_tempo * 41%) * alpha_token_price * $ tao_price

        The $ emitted per tempo to a miner is capped at 30x the $ returns
        """
        top_miner_hk, top_miner_weight = checkpoint_results[0]
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=[top_miner_hk])
        ledger_returns = LedgerUtils.ledger_returns_log(filtered_ledger)
        top_miner_returns_annualized = Metrics.base_return_log_percentage(ledger_returns[top_miner_hk]) / 100

        alpha_token_price = self.metagraph.pool.tao_in / self.metagraph.pool.alpha_in
        alpha_emissions_per_tempo = self.metagraph.emissions.alpha_in_emission * self.metagraph.tempo * 0.41

        daily_blocks = (60 * 60 * 24) / 12  # each block is 12 seconds
        num_tempos_per_day = daily_blocks / self.metagraph.tempo
        dollar_emissions_per_tempo = alpha_emissions_per_tempo * alpha_token_price * tao_price

        # determine the $ value of emissions to the top miner
        dollar_returns_per_tempo = top_miner_returns_annualized / (num_tempos_per_day * ValiConfig.DAYS_IN_YEAR) * ValiConfig.CAPITAL
        burnt_miner_weight = (dollar_returns_per_tempo * 30) / dollar_emissions_per_tempo
        burn_amt = np.clip(1 - (burnt_miner_weight / top_miner_weight), 0, 1)
        bt.logging.info(f"Top miner: {top_miner_hk} Returns: {top_miner_returns_annualized}")
        bt.logging.info(f"alpha_token_price: {alpha_token_price}\n"
                        f"alpha_emissions_per_tempo: {alpha_emissions_per_tempo}\n"
                        f"tao_price: {tao_price}\n"
                        f"tempos per day: {num_tempos_per_day}\n"
                        f"dollar_emissions_per_tempo: {dollar_emissions_per_tempo}\n"
                        f"dollar_returns_per_tempo: {dollar_returns_per_tempo}\n"
                        f"burnt_miner_weight: {burnt_miner_weight}\n"
                        f"original_miner_weight: {top_miner_weight}")
        return burn_amt

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
                    if metagraph_idx == ValiConfig.SN_OWNER_UID:
                        bt.logging.info(f"SN Owner UID registered at {hotkey_registration_blocks[metagraph_idx]}")
                        continue
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
