# developer: jbonilla
import time
import traceback
from setproctitle import setproctitle

import bittensor as bt

from miner_objects.slack_notifier import SlackNotifier
from time_util.time_util import TimeUtil
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.scoring.scoring import Scoring
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
from shared_objects.error_utils import ErrorUtils



class SubtensorWeightSetter(CacheController):
    def __init__(self, metagraph, position_manager: PositionManager,
                 running_unit_tests=False, is_backtesting=False, use_slack_notifier=False,
                 shutdown_dict=None, weight_request_queue=None, config=None, hotkey=None, contract_manager=None,
                 debt_ledger_manager=None, metagraph_updater=None, emissions_ledger_manager=None, is_mainnet=True):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.subnet_version = 200
        # Store weights for use in backtesting
        self.checkpoint_results = []
        self.transformed_list = []
        self.use_slack_notifier = use_slack_notifier
        self._slack_notifier = None
        self.config = config
        self.hotkey = hotkey
        self.contract_manager = contract_manager

        # Debt-based scoring dependencies
        self.debt_ledger_manager = debt_ledger_manager
        self.metagraph_updater = metagraph_updater  # For IPC-safe subtensor calls
        self.emissions_ledger_manager = emissions_ledger_manager
        self.is_mainnet = is_mainnet

        # IPC setup
        self.shutdown_dict = shutdown_dict if shutdown_dict is not None else {}
        self.weight_request_queue = weight_request_queue

    @property
    def slack_notifier(self):
        if self.use_slack_notifier and self._slack_notifier is None:
            self._slack_notifier = SlackNotifier(hotkey=self.hotkey,
                                                webhook_url=getattr(self.config, 'slack_webhook_url', None),
                                                error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
                                                is_miner=False)  # This is a validator
        return self._slack_notifier

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

        # Get all miners from all buckets
        challenge_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
        probation_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.PROBATION))
        plagiarism_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM))
        success_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP))

        # DebtBasedScoring handles all miners together - it applies:
        # - Debt-based weights for MAINCOMP/PROBATION (earning periods)
        # - Minimum dust weights for CHALLENGE/PLAGIARISM/UNKNOWN
        # - Burn address gets excess weight when sum < 1.0
        if self.is_backtesting:
            all_hotkeys = challenge_hotkeys + probation_hotkeys + plagiarism_hotkeys + success_hotkeys
        else:
            all_hotkeys = challenge_hotkeys + probation_hotkeys + plagiarism_hotkeys + success_hotkeys

        bt.logging.info(
            f"Computing weights for {len(all_hotkeys)} miners: "
            f"{len(success_hotkeys)} MAINCOMP, {len(probation_hotkeys)} PROBATION, "
            f"{len(challenge_hotkeys)} CHALLENGE, {len(plagiarism_hotkeys)} PLAGIARISM"
        )

        # Compute weights for all miners using debt-based scoring
        # subcategory_min_days parameter no longer needed for debt-based scoring
        checkpoint_netuid_weights, checkpoint_results = self._compute_miner_weights(
            all_hotkeys, hotkey_to_idx, current_time, subcategory_min_days={}, scoring_challenge=False
        )

        if checkpoint_netuid_weights is None or len(checkpoint_netuid_weights) == 0:
            bt.logging.info("No weights computed. Do nothing for now.")
            return [], []

        transformed_list = checkpoint_netuid_weights
        self.handle_block_reg_failures(transformed_list, target_dtao_block_zero_incentive_start, hotkey_registration_blocks, idx_to_hotkey, target_dtao_block_zero_incentive_end, block_reg_failures)
        bt.logging.info(f"transformed list: {transformed_list}")
        if block_reg_failures:
            bt.logging.info(f"Miners with registration blocks outside of permissible dTAO blocks: {block_reg_failures}")

        return checkpoint_results, transformed_list

    def _compute_miner_weights(self, hotkeys_to_compute_weights_for, hotkey_to_idx, current_time, subcategory_min_days, scoring_challenge: bool = False):
        miner_group = "challenge period" if scoring_challenge else "main competition"

        if len(hotkeys_to_compute_weights_for) == 0:
            return [], []

        bt.logging.info(f"Calculating new subtensor weights for {miner_group} using debt-based scoring...")

        # Get debt ledgers for the specified miners
        if self.debt_ledger_manager:
            # Filter debt ledgers to only include specified hotkeys
            filtered_debt_ledgers = {
                hotkey: ledger
                for hotkey, ledger in self.debt_ledger_manager.debt_ledgers.items()
                if hotkey in hotkeys_to_compute_weights_for
            }
        else:
            bt.logging.warning("debt_ledger_manager not available for scoring")
            return [], []

        if len(filtered_debt_ledgers) == 0:
            bt.logging.warning(f"No debt ledgers found for {miner_group}")
            return [], []

        # Use debt-based scoring
        checkpoint_results = DebtBasedScoring.compute_results(
            ledger_dict=filtered_debt_ledgers,
            metagraph_updater=self.metagraph_updater,
            emissions_ledger_manager=self.emissions_ledger_manager,
            current_time_ms=current_time,
            verbose=True,
            is_testnet=not self.is_mainnet
        )

        bt.logging.info(f"Debt-based scoring results for {miner_group}: [{checkpoint_results}]")

        checkpoint_netuid_weights = []
        for miner, score in checkpoint_results:
            if miner in hotkey_to_idx:
                checkpoint_netuid_weights.append((
                    hotkey_to_idx[miner],
                    score
                ))
            else:
                bt.logging.error(f"Miner {miner} not found in the metagraph.")

        return checkpoint_netuid_weights, checkpoint_results

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

    
    def run_update_loop(self):
        """
        Weight setter loop that sends fire-and-forget requests to MetagraphUpdater.
        """
        setproctitle(f"vali_{self.__class__.__name__}")
        bt.logging.enable_info()
        bt.logging.info("Starting weight setter update loop (fire-and-forget IPC mode)")

        while not self.shutdown_dict:
            try:
                if self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
                    bt.logging.info("Computing weights for IPC request")
                    current_time = TimeUtil.now_in_millis()

                    # Compute weights (existing logic)
                    checkpoint_results, transformed_list = self.compute_weights_default(current_time)
                    self.checkpoint_results = checkpoint_results
                    self.transformed_list = transformed_list

                    if transformed_list and self.weight_request_queue:
                        # Send weight setting request (fire-and-forget)
                        self._send_weight_request(transformed_list)
                        self.set_last_update_time()
                    else:
                        # No weights computed - likely debt_ledger_manager not ready yet
                        # Sleep for 5 minutes to avoid busy looping
                        if not self.debt_ledger_manager:
                            bt.logging.warning(
                                "debt_ledger_manager not available. "
                                "Waiting 5 minutes before retry..."
                            )
                            time.sleep(300)  # 5 minutes
                        else:
                            bt.logging.debug("No weights to set or IPC not available")

            except Exception as e:
                bt.logging.error(f"Error in weight setter update loop: {e}")
                bt.logging.error(traceback.format_exc())

                # Send error notification
                if self.slack_notifier:
                    # Get compact stack trace using shared utility
                    compact_trace = ErrorUtils.get_compact_stacktrace(e)
                    self.slack_notifier.send_message(
                        f"❌ Weight setter process error!\n"
                        f"Error: {str(e)}\n"
                        f"This occurred in the weight setter update loop\n"
                        f"Trace: {compact_trace}",
                        level="error"
                    )
                time.sleep(30)

            time.sleep(1)

        bt.logging.info("Weight setter update loop shutting down")
    
    def _send_weight_request(self, transformed_list):
        """Send weight setting request to MetagraphUpdater (fire-and-forget)"""
        try:
            uids = [x[0] for x in transformed_list]
            weights = [x[1] for x in transformed_list]
            
            # Send request (no response expected)
            # MetagraphUpdater will use its own config for netuid and wallet
            request = {
                'uids': uids,
                'weights': weights,
                'version_key': self.subnet_version,
                'timestamp': TimeUtil.now_in_millis()
            }
            
            self.weight_request_queue.put_nowait(request)
            bt.logging.info(f"Weight request sent: {len(uids)} UIDs via IPC")
            
        except Exception as e:
            bt.logging.error(f"Error sending weight request: {e}")
            bt.logging.error(traceback.format_exc())

            # Send error notification
            if self.slack_notifier:
                # Get compact stack trace using shared utility
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"❌ Weight request IPC error!\n"
                    f"Error: {str(e)}\n"
                    f"This occurred while sending weight request via IPC\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )

    def set_weights(self, current_time):
        # Compute weights (existing logic)
        checkpoint_results, transformed_list = self.compute_weights_default(current_time)
        self.checkpoint_results = checkpoint_results
        self.transformed_list = transformed_list

