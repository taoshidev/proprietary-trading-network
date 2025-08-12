# developer: jbonilla
from functools import partial
import time
import traceback
from setproctitle import setproctitle

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.scoring.scoring import Scoring
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger

from shared_objects.subtensor_lock import get_subtensor_lock


class WeightFailureTracker:
    """Track weight setting failures and manage alerting logic"""
    
    def __init__(self):
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.last_alert_time = 0
        self.failure_patterns = {}  # Track unknown error patterns
        self.had_critical_failure = False
        
    def classify_failure(self, err_msg):
        """Classify failure based on production patterns"""
        error_lower = err_msg.lower()
        
        # BENIGN - Don't alert (expected behavior)
        if any(phrase in error_lower for phrase in [
            "no attempt made. perhaps it is too soon to commit weights",
            "too soon to commit weights",
            "too soon to commit"
        ]):
            return "benign"
        
        # CRITICAL - Alert immediately (known problematic patterns)
        elif any(phrase in error_lower for phrase in [
            "maximum recursion depth exceeded",
            "invalid transaction",
            "subtensor returned: invalid transaction"
        ]):
            return "critical"
        
        # UNKNOWN - Alert after pattern emerges
        else:
            return "unknown"
    
    def should_alert(self, failure_type, consecutive_count):
        """Determine if we should send an alert"""
        # Get current time once for consistency
        current_time = time.time()
        time_since_success = current_time - self.last_success_time
        time_since_last_alert = current_time - self.last_alert_time
        
        # Alert if we haven't had a successful weight setting in 2 hours
        # This is an absolute timeout that bypasses all other checks
        if time_since_success > 7200:  # 2 hours
            return True
        
        # Rate limiting check - but exempt critical errors and 1+ hour timeouts
        if failure_type != "critical" and time_since_success <= 3600:
            if time_since_last_alert < 600:
                return False
        
        # Always alert for known critical errors (no rate limiting)
        if failure_type == "critical":
            return True
        
        # Alert if we haven't had a successful weight setting in 1 hour
        # This check happens before benign check to catch prolonged benign failures
        if time_since_success > 3600:
            return True
        
        # Never alert for benign "too soon" errors (unless prolonged, caught above)
        if failure_type == "benign":
            return False
        
        # For unknown errors, alert after 2 consecutive failures
        if failure_type == "unknown" and consecutive_count >= 2:
            return True
        
        return False
    
    def track_failure(self, err_msg, failure_type):
        """Track a failure"""
        self.consecutive_failures += 1
        
        # Track if this was a critical failure
        if failure_type == "critical":
            self.had_critical_failure = True
        
        # Track unknown error patterns
        if failure_type == "unknown":
            pattern_key = err_msg[:50] if len(err_msg) > 50 else err_msg
            self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
    
    def track_success(self):
        """Track a successful weight setting"""
        # Check if we should send recovery alert
        should_send_recovery = self.consecutive_failures > 0 and self.had_critical_failure
        
        # Reset tracking
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.had_critical_failure = False
        
        return should_send_recovery

class SubtensorWeightSetter(CacheController):
    def __init__(self, metagraph, position_manager: PositionManager,
                 running_unit_tests=False, is_backtesting=False, slack_notifier=None,
                 config=None, netuid=None, shutdown_dict=None, weight_request_queue=None):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.subnet_version = 200
        # Store weights for use in backtesting
        self.checkpoint_results = []
        self.transformed_list = []
        self.weight_failure_tracker = WeightFailureTracker()
        self.slack_notifier = slack_notifier
        
        # Config and IPC setup
        self.config = config
        self.netuid = netuid
        self.shutdown_dict = shutdown_dict if shutdown_dict is not None else {}
        self.weight_request_queue = weight_request_queue

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
        challenge_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
        probation_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.PROBATION))
        testing_hotkeys = challenge_hotkeys +  probation_hotkeys
        success_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP))

        if self.is_backtesting:
            hotkeys_to_compute_weights_for = testing_hotkeys + success_hotkeys
        else:
            hotkeys_to_compute_weights_for = success_hotkeys
        checkpoint_netuid_weights, checkpoint_results = self._compute_miner_weights(hotkeys_to_compute_weights_for, hotkey_to_idx, current_time, scoring_challenge=False)

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

    def _compute_miner_weights(self, hotkeys_to_compute_weights_for, hotkey_to_idx, current_time, scoring_challenge: bool = False):

        miner_group = "challenge period" if scoring_challenge else "main competition"

        # only collect ledger elements for the miners that passed the challenge period
        filtered_ledger: dict[str, dict[str, PerfLedger]] = self.perf_ledger_manager.filtered_ledger_for_scoring(
            hotkeys=hotkeys_to_compute_weights_for)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(
            hotkeys=hotkeys_to_compute_weights_for)

        if len(filtered_ledger) == 0:
            return [], []
        else:
            bt.logging.info(f"Calculating new subtensor weights for {miner_group}...")
            checkpoint_results = Scoring.compute_results_checkpoint(
                filtered_ledger,
                filtered_positions,
                evaluation_time_ms=current_time,
                verbose=True,
                weighting=True
            )
            checkpoint_results = sorted(checkpoint_results, key=lambda x: x[1], reverse=True)

            bt.logging.info(f"Sorted results for weight setting for {miner_group}: [{checkpoint_results}]")

            # Extra processing if scoring challenge
            processed_checkpoint_results = checkpoint_results

            if scoring_challenge:
                processed_checkpoint_results = Scoring.score_testing_miners(filtered_ledger, checkpoint_results)

            checkpoint_netuid_weights = []
            for miner, score in processed_checkpoint_results:
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

        # Synchronize with metagraph updates to prevent WebSocket concurrency errors
        with get_subtensor_lock():
            success, err_msg = subtensor.set_weights(
                netuid=netuid,
                wallet=wallet,
                uids=filtered_netuids,
                weights=scaled_transformed_list,
                version_key=self.subnet_version,
            )

        if success:
            bt.logging.success("Successfully set weights.")
            
            # Check if we should send recovery alert
            if self.weight_failure_tracker.track_success() and self.slack_notifier:
                self._send_recovery_alert(wallet)
        else:
            # Classify the failure
            failure_type = self.weight_failure_tracker.classify_failure(err_msg)
            
            # Log appropriately
            if failure_type == "benign":
                bt.logging.warning(f"Failed to set weights. Error message: {err_msg} (benign)")
            else:
                bt.logging.warning(f"Failed to set weights. Error message: {err_msg}")
            
            # Track the failure
            self.weight_failure_tracker.track_failure(err_msg, failure_type)
            
            # Check if we should alert
            if self.weight_failure_tracker.should_alert(failure_type, self.weight_failure_tracker.consecutive_failures):
                self._send_weight_failure_alert(err_msg, failure_type, wallet)
                self.weight_failure_tracker.last_alert_time = time.time()
    
    def _send_weight_failure_alert(self, err_msg, failure_type, wallet):
        """Send contextual Slack alert for weight setting failure"""
        if not self.slack_notifier:
            return
        
        # Get context information
        hotkey = "unknown"
        if wallet:
            if hasattr(wallet, 'hotkey'):
                if hasattr(wallet.hotkey, 'ss58_address'):
                    hotkey = wallet.hotkey.ss58_address
                else:
                    bt.logging.warning("Wallet hotkey missing ss58_address attribute")
            else:
                bt.logging.warning("Wallet missing hotkey attribute")
        else:
            bt.logging.warning("Wallet parameter is None in weight failure alert")
        
        netuid = "unknown"
        network = "unknown"
        if self.config:
            if hasattr(self.config, 'netuid'):
                netuid = self.config.netuid
            else:
                bt.logging.warning("Config missing netuid attribute")
                
            if hasattr(self.config, 'subtensor'):
                if hasattr(self.config.subtensor, 'network'):
                    network = self.config.subtensor.network
                else:
                    bt.logging.warning("Config subtensor missing network attribute")
            else:
                bt.logging.warning("Config missing subtensor attribute")
        else:
            bt.logging.warning("Config is None - cannot determine network/netuid for alert")
            
        consecutive = self.weight_failure_tracker.consecutive_failures
        
        # Build alert message based on failure type
        if "maximum recursion depth exceeded" in err_msg.lower():
            message = f"🚨 CRITICAL: Weight setting recursion error\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"Error: {err_msg}\n" \
                     f"This indicates a serious code issue that needs immediate attention."
        
        elif "invalid transaction" in err_msg.lower():
            message = f"🚨 CRITICAL: Subtensor rejected weight transaction\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"Error: {err_msg}\n" \
                     f"This may indicate wallet/balance issues or network problems."
        
        elif failure_type == "unknown":
            message = f"❓ NEW PATTERN: Unknown weight setting failure\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"Consecutive failures: {consecutive}\n" \
                     f"Error: {err_msg}\n" \
                     f"This is a new error pattern that needs investigation."
        
        else:
            # Prolonged failure alert
            time_since_success = time.time() - self.weight_failure_tracker.last_success_time
            hours_since_success = time_since_success / 3600
            
            if hours_since_success >= 2:
                urgency = "🚨 URGENT"
                time_msg = f"No successful weight setting in {hours_since_success:.1f} hours"
            else:
                urgency = "⚠️ WARNING"
                time_msg = f"No successful weight setting in {hours_since_success:.1f} hours"
            
            message = f"{urgency}: Weight setting issues detected\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"{time_msg}\n" \
                     f"Last error: {err_msg}"
        
        self.slack_notifier.send_message(message, level="error")
    
    def _send_recovery_alert(self, wallet):
        """Send recovery alert after critical failures"""
        if not self.slack_notifier:
            return
        
        hotkey = "unknown"
        if wallet:
            if hasattr(wallet, 'hotkey'):
                if hasattr(wallet.hotkey, 'ss58_address'):
                    hotkey = wallet.hotkey.ss58_address
                else:
                    bt.logging.warning("Wallet hotkey missing ss58_address attribute in recovery alert")
            else:
                bt.logging.warning("Wallet missing hotkey attribute in recovery alert")
        else:
            bt.logging.warning("Wallet parameter is None in recovery alert")
            
        network = "unknown"
        if self.config:
            if hasattr(self.config, 'subtensor'):
                if hasattr(self.config.subtensor, 'network'):
                    network = self.config.subtensor.network
                else:
                    bt.logging.warning("Config subtensor missing network attribute in recovery alert")
            else:
                bt.logging.warning("Config missing subtensor attribute in recovery alert")
        else:
            bt.logging.warning("Config is None - cannot determine network for recovery alert")
        
        message = f"✅ Weight setting recovered after failures\n" \
                 f"Network: {network}\n" \
                 f"Hotkey: {hotkey}"
        
        self.slack_notifier.send_message(message, level="info")
    
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
                        bt.logging.debug("No weights to set or IPC not available")
                        
            except Exception as e:
                bt.logging.error(f"Error in weight setter update loop: {e}")
                bt.logging.error(traceback.format_exc())
                
                # Send error notification
                if self.slack_notifier:
                    self.slack_notifier.send_message(
                        f"❌ Weight setter process error!\n"
                        f"Error: {str(e)}\n"
                        f"This occurred in the weight setter update loop",
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
            
            # Create serializable wallet config
            wallet_config = {
                'name': self.config.wallet.name,
                'hotkey': self.config.wallet.hotkey,
                'path': self.config.wallet.path
            }
            
            # Send request (no response expected)
            request = {
                'wallet_config': wallet_config,
                'netuid': self.netuid,
                'uids': uids,
                'weights': weights,
                'version_key': self.subnet_version,
                'timestamp': TimeUtil.now_in_millis()
            }
            
            self.weight_request_queue.put_nowait(request)
            bt.logging.info(f"Weight request sent: {len(uids)} UIDs via IPC")
            
        except Exception as e:
            bt.logging.error(f"Error sending weight request: {e}")
