# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import os
import sys
import threading
import signal
import uuid
from typing import Tuple
from enum import Enum

import template
import argparse
import traceback
import time
import bittensor as bt
import json
import gzip
import base64

from runnable.generate_request_core import generate_request_core
from runnable.generate_request_minerstatistics import generate_miner_statistics_data
from vali_objects.utils.auto_sync import PositionSyncer
from vali_objects.utils.p2p_syncer import P2PSyncer
from shared_objects.rate_limiter import RateLimiter
from vali_objects.utils.timestamp_manager import TimestampManager
from vali_objects.uuid_tracker import UUIDTracker
from time_util.time_util import TimeUtil
from vali_config import TradePair
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.metagraph_updater import MetagraphUpdater
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_dataclasses.order import Order
from vali_objects.position import Position
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig

# Global flag used to indicate shutdown
shutdown_dict = {}

# Enum class that represents the method associated with Synapse
class SynapseMethod(Enum):
    POSITION_INSPECTOR = "GetPositions"
    DASHBOARD = "GetDashData"
    SIGNAL = "SendSignal"
    CHECKPOINT = "SendCheckpoint"

def signal_handler(signum, frame):
    global shutdown_dict

    if shutdown_dict:
        return  # Ignore if already in shutdown

    if signum in (signal.SIGINT, signal.SIGTERM):
        signal_message = "Handling SIGINT" if signum == signal.SIGINT else "Handling SIGTERM"
        print(f"{signal_message} - Initiating graceful shutdown")

        shutdown_dict[True] = True
        # Set a 2-second alarm
        signal.alarm(2)

def alarm_handler(signum, frame):
    print("Graceful shutdown failed, force killing the process")
    sys.exit(1)  # Exit immediately

# Set up signal handling
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGALRM, alarm_handler)

class Validator:
    def __init__(self):
        # Try to read the file meta/meta.json and print it out
        try:
            with open("meta/meta.json", "r") as f:
                bt.logging.info(f"Found meta.json file {f.read()}")
        except Exception as e:
            bt.logging.error(f"Error reading meta/meta.json: {e}")

        ValiBkpUtils.clear_tmp_dir()
        self.uuid_tracker = UUIDTracker()
        # Lock to stop new signals from being processed while a validator is restoring
        self.signal_sync_lock = threading.Lock()
        self.signal_sync_condition = threading.Condition(self.signal_sync_lock)
        self.n_orders_being_processed = [0]  # Allow this to be updated across threads by placing it in a list (mutable)

        self.config = self.get_config()
        # Use the getattr function to safely get the autosync attribute with a default of False if not found.
        self.auto_sync = getattr(self.config, 'autosync', False) and 'mothership' not in ValiUtils.get_secrets()
        self.is_mainnet = self.config.netuid == 8
        # Ensure the directory for logging exists, else create one.
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

        self.secrets = ValiUtils.get_secrets()
        if self.secrets is None:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )

        self.live_price_fetcher = LivePriceFetcher(secrets=self.secrets)
        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} with autosync set to: {self.auto_sync} "
            f"on network: {self.config.subtensor.chain_endpoint} with config:"
        )

        # This logs the active configuration to the specified logging directory for review.
        bt.logging.info(self.config)

        # Initialize Bittensor miner objects
        # These classes are vital to interact and function within the Bittensor network.
        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
        subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {subtensor}")


        # metagraph provides the network's current state, holding state about other participants in a subnet.
        # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
        self.metagraph = subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        #force_validator_to_restore_from_checkpoint(self.wallet.hotkey.ss58_address, self.metagraph, self.config, self.secrets)

        self.position_manager = PositionManager(metagraph=self.metagraph, config=self.config,
                                                perform_price_adjustment=False,
                                                live_price_fetcher=self.live_price_fetcher,
                                                perform_fee_structure_update=False,
                                                perform_order_corrections=True,
                                                apply_corrections_template=False,
                                                perform_compaction=True)


        self.metagraph_updater = MetagraphUpdater(self.config, self.metagraph, self.wallet.hotkey.ss58_address,
                                                  False, position_manager=self.position_manager,
                                                  shutdown_dict=shutdown_dict)

        # Start the metagraph updater loop in its own thread
        self.metagraph_updater_thread = threading.Thread(target=self.metagraph_updater.run_update_loop, daemon=True)
        self.metagraph_updater_thread.start()



        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} is not registered to chain "
                f"connection: {subtensor} \nRun btcli register and try again. "
            )
            exit()

        self.position_syncer = PositionSyncer(shutdown_dict=shutdown_dict, signal_sync_lock=self.signal_sync_lock,
                                              signal_sync_condition=self.signal_sync_condition,
                                              n_orders_being_processed=self.n_orders_being_processed)
        self.p2p_syncer = P2PSyncer(wallet=self.wallet, metagraph=self.metagraph, is_testnet=not self.is_mainnet,
                                    shutdown_dict=shutdown_dict, signal_sync_lock=self.signal_sync_lock,
                                    signal_sync_condition=self.signal_sync_condition,
                                    n_orders_being_processed=self.n_orders_being_processed
                                    )
        self.checkpoint_lock = threading.Lock()
        self.encoded_checkpoint = ""
        self.last_checkpoint_time = 0
        self.timestamp_manager = TimestampManager(config=self.config, metagraph=self.metagraph, hotkey=self.wallet.hotkey.ss58_address)

        self.perf_ledger_manager = PerfLedgerManager(self.metagraph, shutdown_dict=shutdown_dict, position_syncer=self.position_syncer)
        # Start the perf ledger updater loop in its own thread
        self.perf_ledger_updater_thread = threading.Thread(target=self.perf_ledger_manager.run_update_loop, daemon=True)
        self.perf_ledger_updater_thread.start()

        # Build and link vali functions to the axon.
        # The axon handles request processing, allowing validators to send this process requests.
        bt.logging.info(f"setting port [{self.config.axon.port}]")
        bt.logging.info(f"setting external port [{self.config.axon.external_port}]")
        self.axon = bt.axon(
            wallet=self.wallet, port=self.config.axon.port, external_port=self.config.axon.external_port
        )
        bt.logging.info(f"Axon {self.axon}")

        # Attach determines which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to axon.")

        self.order_rate_limiter = RateLimiter()
        self.position_inspector_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 4)
        self.dash_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60)
        self.checkpoint_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 60 * 6)

        def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def rs_priority_fn(synapse: template.protocol.SendSignal) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def gp_blacklist_fn(synapse: template.protocol.GetPositions) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def gp_priority_fn(synapse: template.protocol.GetPositions) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def gd_blacklist_fn(synapse: template.protocol.GetDashData) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def gd_priority_fn(synapse: template.protocol.GetDashData) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def rc_blacklist_fn(synapse: template.protocol.ValidatorCheckpoint) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def rc_priority_fn(synapse: template.protocol.ValidatorCheckpoint) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        self.axon.attach(
            forward_fn=self.receive_signal,
            blacklist_fn=rs_blacklist_fn,
            priority_fn=rs_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.get_positions,
            blacklist_fn=gp_blacklist_fn,
            priority_fn=gp_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.get_data,
            blacklist_fn=gd_blacklist_fn,
            priority_fn=gd_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.receive_checkpoint,
            blacklist_fn=rc_blacklist_fn,
            priority_fn=rc_priority_fn,
        )

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving attached axons on network:"
            f" {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=subtensor)

        # Starts the miner's axon, making it active on the network.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

        # see if cache files exist and if not set them to empty
        self.position_manager.init_cache_files()

        # Each hotkey gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

        # Eliminations are read in validator, elimination_manager, mdd_checker, weight setter.
        # Eliminations are written in elimination_manager, mdd_checker
        # Since the mainloop is run synchronously, we just need to lock eliminations when writing to them and when
        # reading outside of the mainloop (validator).
        self.eliminations_lock = threading.Lock()
        # self.plagiarism_detector = PlagiarismDetector(self.config, self.metagraph)
        self.mdd_checker = MDDChecker(self.config, self.metagraph, self.position_manager, self.eliminations_lock,
                                      live_price_fetcher=self.live_price_fetcher, shutdown_dict=shutdown_dict)
        self.weight_setter = SubtensorWeightSetter(self.config, self.wallet, self.metagraph)
        self.challengeperiod_manager = ChallengePeriodManager(self.config, self.metagraph)

        self.elimination_manager = EliminationManager(self.metagraph, self.position_manager, self.eliminations_lock)

        # Validators on mainnet net to be syned for the first time or after interruption need to resync their
        # positions. Assert there are existing orders that occurred > 24hrs in the past. Assert that the newest order
        # was placed within 24 hours.
        if self.is_mainnet:
            n_positions_on_disk = self.position_manager.get_number_of_miners_with_any_positions()
            oldest_disk_ms, youngest_disk_ms = (
                self.position_manager.get_extreme_position_order_processed_on_disk_ms())
            if (n_positions_on_disk > 0):
                bt.logging.info(f"Found {n_positions_on_disk} positions on disk."
                                f" Found oldest_disk_ms: {TimeUtil.millis_to_datetime(oldest_disk_ms)},"
                                f" oldest_disk_ms: {TimeUtil.millis_to_datetime(youngest_disk_ms)}")
            one_day_ago = TimeUtil.timestamp_to_millis(TimeUtil.generate_start_timestamp(days=1))
            if (n_positions_on_disk == 0 or youngest_disk_ms < one_day_ago):
                msg = ("Validator data needs to be synced with mainnet validators. "
                       "Restoring validator with 24 hour lagged file. More info here: "
                       "https://github.com/taoshidev/proprietary-trading-network/"
                       "blob/main/docs/regenerating_validator_state.md")
                bt.logging.warning(msg)
                self.position_syncer.sync_positions(
                    False, candidate_data=self.position_syncer.read_validator_checkpoint_from_gcloud_zip())




    @staticmethod
    def blacklist_fn(synapse, metagraph) -> Tuple[bool, str]:
        miner_hotkey = synapse.dendrite.hotkey
        # Ignore requests from unrecognized entities.
        if miner_hotkey not in metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    @staticmethod
    def priority_fn(synapse, metagraph) -> float:
        # simply just prioritize based on uid as it's not significant
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.uids[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def get_config(self):
        # Step 2: Set up the configuration parser
        # This function initializes the necessary command-line arguments.
        # Using command-line arguments allows users to customize various miner settings.
        parser = argparse.ArgumentParser()
        # Set autosync to store true if flagged, otherwise defaults to False.
        parser.add_argument("--autosync", action='store_true',
                            help="Automatically sync order data with a validator trusted by Taoshi.")
        # (developer): Adds your custom arguments to the parser.
        # Adds override arguments for network and netuid.
        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
        # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.wallet.add_args(parser)
        # Adds axon specific arguments i.e. --axon.port ...
        bt.axon.add_args(parser)
        # Activating the parser to read any command-line inputs.
        # To print help message, run python3 template/miner.py --help
        config = bt.config(parser)
        bt.logging.enable_default()
        if config.logging.debug:
            bt.logging.enable_debug()
        if config.logging.trace:
            bt.logging.enable_trace()


        # Step 3: Set up logging directory
        # Logging captures events for diagnosis or understanding miner's behavior.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "miner",
            )
        )
        return config

    def check_shutdown(self):
        global shutdown_dict
        if not shutdown_dict:
            return
        # Handle shutdown gracefully
        bt.logging.warning("Performing graceful exit...")
        bt.logging.warning("Stopping axon...")
        self.axon.stop()
        bt.logging.warning("Stopping metagrpah update...")
        self.metagraph_updater_thread.join()
        bt.logging.warning("Stopping live price fetcher...")
        self.live_price_fetcher.stop_all_threads()
        bt.logging.warning("Stopping perf ledger...")
        self.perf_ledger_updater_thread.join()
        signal.alarm(0)
        print("Graceful shutdown completed")
        sys.exit(0)

    def main(self):
        global shutdown_dict
        # Keep the vali alive. This loop maintains the vali's operations until intentionally stopped.
        bt.logging.info("Starting main loop")
        while not shutdown_dict:
            try:
                current_time = TimeUtil.now_in_millis()
                self.position_syncer.sync_positions_with_cooldown(self.auto_sync)
                self.mdd_checker.mdd_check()
                self.challengeperiod_manager.refresh(current_time=current_time)
                self.elimination_manager.process_eliminations()
                self.weight_setter.set_weights(current_time=current_time)
                self.position_manager.position_locks.cleanup_locks(self.metagraph.hotkeys)
                self.p2p_syncer.sync_positions_with_cooldown()

            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception:
                bt.logging.error(traceback.format_exc())
                time.sleep(10)

        self.check_shutdown()

    def parse_trade_pair_from_signal(self, signal) -> TradePair | None:
        if not signal or not isinstance(signal, dict):
            return None
        if 'trade_pair' not in signal:
            return None
        temp = signal["trade_pair"]
        if 'trade_pair_id' not in temp:
            return None
        string_trade_pair = signal["trade_pair"]["trade_pair_id"]
        trade_pair = TradePair.from_trade_pair_id(string_trade_pair)
        return trade_pair

    def convert_signal_to_order(self, signal, hotkey, now_ms, miner_order_uuid) -> Order:
        """
        Example input signal
          {'trade_pair': {'trade_pair_id': 'BTCUSD', 'trade_pair': 'BTC/USD', 'fees': 0.003, 'min_leverage': 0.0001, 'max_leverage': 20},
          'order_type': 'LONG',
          'leverage': 0.5}
        """
        trade_pair = self.parse_trade_pair_from_signal(signal)
        if trade_pair is None:
            bt.logging.error(f"[{trade_pair}] not in TradePair enum.")
            raise SignalException(
                f"miner [{hotkey}] incorrectly sent trade pair. Raw signal: {signal}"
            )

        signal_order_type = OrderType.from_string(signal["order_type"])
        signal_leverage = signal["leverage"]

        bt.logging.info("Attempting to get live price for trade pair: " + trade_pair.trade_pair_id)
        live_closing_price, price_sources = self.live_price_fetcher.get_latest_price(trade_pair=trade_pair,
                                                                                     time_ms=now_ms)

        order = Order(
            trade_pair=trade_pair,
            order_type=signal_order_type,
            leverage=signal_leverage,
            price=live_closing_price,
            processed_ms=now_ms,
            order_uuid=miner_order_uuid if miner_order_uuid else str(uuid.uuid4()),
            price_sources=price_sources
        )
        bt.logging.success(f"Converted signal to order: {order}")
        return order

    def _enforce_num_open_order_limit(self, trade_pair_to_open_position: dict, signal_to_order):
        # Check if there are too many orders across all open positions.
        # If so, check if the current order is a FLAT order (reduces number of open orders). If not, raise an exception
        n_open_positions = sum([len(position.orders) for position in trade_pair_to_open_position.values()])
        if n_open_positions >= ValiConfig.MAX_OPEN_ORDERS_PER_HOTKEY:
            if signal_to_order.order_type != OrderType.FLAT:
                raise SignalException(
                    f"miner [{signal_to_order}] sent too many open orders [{len(trade_pair_to_open_position)}] and "
                    f"order [{signal_to_order}] is not a FLAT order."
                )

    def _get_or_create_open_position(self, signal_to_order: Order, miner_hotkey: str, trade_pair_to_open_position: dict, miner_order_uuid: str):
        trade_pair = signal_to_order.trade_pair

        # if a position already exists, add the order to it
        if trade_pair in trade_pair_to_open_position:
            # If the position is closed, raise an exception. This can happen if the miner is eliminated in the main
            # loop thread.
            if trade_pair_to_open_position[trade_pair].is_closed_position:
                raise SignalException(
                    f"miner [{miner_hotkey}] sent signal for "
                    f"closed position [{trade_pair}]")
            bt.logging.debug("adding to existing position")
            open_position = trade_pair_to_open_position[trade_pair]
        else:
            bt.logging.debug("processing new position")
            # if the order is FLAT ignore (noop)
            if signal_to_order.order_type == OrderType.FLAT:
                open_position = None
            else:
                # if a position doesn't exist, then make a new one
                open_position = Position(
                    miner_hotkey=miner_hotkey,
                    position_uuid=miner_order_uuid if miner_order_uuid else str(uuid.uuid4()),
                    open_ms=TimeUtil.now_in_millis(),
                    trade_pair=trade_pair
                )
        return open_position

    def enforce_no_duplicate_order(self, synapse: template.protocol.SendSignal):
        order_uuid = synapse.miner_order_uuid
        if order_uuid:
            if self.uuid_tracker.exists(order_uuid):
                msg = (f"Order with uuid [{order_uuid}] has already been processed. "
                       f"Please try again with a new order.")
                bt.logging.error(msg)
                synapse.successfully_processed = False
                synapse.error_message = msg

    def should_fail_early(self, synapse: template.protocol.SendSignal | template.protocol.GetPositions | template.protocol.GetDashData | template.protocol.ValidatorCheckpoint, method:SynapseMethod,
                          signal:dict=None) -> bool:
        global shutdown_dict
        if shutdown_dict:
            synapse.successfully_processed = False
            synapse.error_message = "Validator is restarting due to update. Please try again later."
            bt.logging.trace(synapse.error_message)
            return True

        sender_hotkey = synapse.dendrite.hotkey
        # Don't allow miners to send too many signals in a short period of time
        if method == SynapseMethod.POSITION_INSPECTOR:
            allowed, wait_time = self.position_inspector_rate_limiter.is_allowed(sender_hotkey)
        elif method == SynapseMethod.DASHBOARD:
            allowed, wait_time = self.dash_rate_limiter.is_allowed(sender_hotkey)
        elif method == SynapseMethod.SIGNAL:
            allowed, wait_time = self.order_rate_limiter.is_allowed(sender_hotkey)
        elif method == SynapseMethod.CHECKPOINT:
            allowed, wait_time = self.checkpoint_rate_limiter.is_allowed(sender_hotkey)
        else:
            msg = "Received synapse does not match one of expected methods for: receive_signal, get_positions, get_data, or receive_checkpoint"
            bt.logging.trace(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if not allowed:
            msg = (f"Rate limited. Please wait {wait_time} seconds before sending another signal. "
                   f"{method.value}")
            bt.logging.trace(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if method == SynapseMethod.CHECKPOINT or method == SynapseMethod.DASHBOARD:
            return False

        # don't process eliminated miners
        with self.eliminations_lock:
            eliminations = self.mdd_checker.get_eliminations_from_disk()
        eliminated_hotkey_to_info ={x['hotkey']: x for x in eliminations} if eliminations else dict()
        if synapse.dendrite.hotkey in eliminated_hotkey_to_info:
            info = eliminated_hotkey_to_info[synapse.dendrite.hotkey]
            msg = f"This miner hotkey {synapse.dendrite.hotkey} has been eliminated for reason {info} and cannot participate in this subnet. Try again after re-registering."
            bt.logging.debug(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if signal:
            tp = self.parse_trade_pair_from_signal(signal)
            if tp and not self.live_price_fetcher.polygon_data_service.is_market_open(tp):
                msg = (f"Market for trade pair [{tp.trade_pair_id}] is likely closed or this validator is"
                       f" having issues fetching live price. Please try again later.")
                bt.logging.error(msg)
                synapse.successfully_processed = False
                synapse.error_message = msg
                return True

            if tp and tp in self.live_price_fetcher.polygon_data_service.UNSUPPORTED_TRADE_PAIRS:
                msg = (f"Trade pair [{tp.trade_pair_id}] has been temporarily halted. "
                       f"Please try again with a different trade pair.")
                bt.logging.error(msg)
                synapse.successfully_processed = False
                synapse.error_message = msg
                return True

            self.enforce_no_duplicate_order(synapse)
            if synapse.error_message:
                return True


        return False

    def enforce_order_cooldown(self, order, open_position):
        # Don't allow orders to be placed within ORDER_COOLDOWN_MS of the previous order. The only exception is if
        # there is a new websocket price for the trade pair. The intention here is to prevent exploiting a lag in a
        # data provider.
        if len(open_position.orders) == 0:
            return
        last_order = open_position.orders[-1]
        last_order_time_ms = last_order.processed_ms
        time_since_last_order_ms = order.processed_ms - last_order_time_ms

        if time_since_last_order_ms >= ValiConfig.ORDER_COOLDOWN_MS:
            return

        #lag_time_ms = self.live_price_fetcher.time_since_last_ws_ping_s(order.trade_pair) * 1000

        previous_order_time = TimeUtil.millis_to_formatted_date_str(last_order.processed_ms)
        current_time = TimeUtil.millis_to_formatted_date_str(order.processed_ms)
        time_to_wait_in_s = (ValiConfig.ORDER_COOLDOWN_MS - (order.processed_ms - last_order.processed_ms)) / 1000
        raise SignalException(
            f"Order for trade pair [{order.trade_pair.trade_pair_id}] was placed too soon after the previous order. "
            f"Last order was placed at [{previous_order_time}] and current order was placed at [{current_time}]."
            f"Please wait {time_to_wait_in_s} seconds before placing another order."
        )

    def parse_miner_uuid(self, synapse: template.protocol.SendSignal):
        temp = synapse.miner_order_uuid
        assert isinstance(temp, str), f"excepted string miner uuid but got {temp}"
        return temp[:50]

    # This is the core validator function to receive a signal
    def receive_signal(self, synapse: template.protocol.SendSignal,
                       ) -> template.protocol.SendSignal:
        # pull miner hotkey to reference in various activities
        now_ms = TimeUtil.now_in_millis()
        miner_hotkey = synapse.dendrite.hotkey
        synapse.validator_hotkey = self.wallet.hotkey.ss58_address
        signal = synapse.signal
        bt.logging.info(f"received signal [{signal}] from miner_hotkey [{miner_hotkey}].")
        if self.should_fail_early(synapse, SynapseMethod.SIGNAL, signal=signal):
            return synapse

        with self.signal_sync_lock:
            self.n_orders_being_processed[0] += 1

        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""
        try:
            miner_order_uuid = self.parse_miner_uuid(synapse)
            signal_to_order = self.convert_signal_to_order(signal, miner_hotkey, now_ms, miner_order_uuid)
            # Multiple threads can run receive_signal at once. Don't allow two threads to trample each other.
            with self.position_manager.position_locks.get_lock(miner_hotkey, signal_to_order.trade_pair.trade_pair_id):
                self.enforce_no_duplicate_order(synapse)
                if synapse.error_message:
                    return synapse
                # gather open positions and see which trade pairs have an open position
                trade_pair_to_open_position = {position.trade_pair: position for position in
                                               self.position_manager.get_all_miner_positions(miner_hotkey,
                                                                                             only_open_positions=True)}
                self._enforce_num_open_order_limit(trade_pair_to_open_position, signal_to_order)
                open_position = self._get_or_create_open_position(signal_to_order, miner_hotkey, trade_pair_to_open_position, miner_order_uuid)
                if open_position:
                    self.enforce_order_cooldown(signal_to_order, open_position)
                    open_position.add_order(signal_to_order)
                    self.position_manager.save_miner_position_to_disk(open_position)
                    bt.logging.info(
                        f"Position {open_position.trade_pair.trade_pair_id} for miner [{miner_hotkey}] updated.")
                    # Log the open position for the miner
                    open_position.log_position_status()
                    if miner_order_uuid:
                        self.uuid_tracker.add(miner_order_uuid)
                else:
                    # Happens if a FLAT is sent when no order exists
                    pass
                # Update the last received order time
                self.timestamp_manager.update_timestamp(signal_to_order)

            # self.plagiarism_detector.check_plagiarism(open_position, signal_to_order)

        except SignalException as e:
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())
        except Exception as e:
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False

        synapse.error_message = error_message
        bt.logging.success(f"Sending ack back to miner [{miner_hotkey}]. Synapse Message: {synapse.error_message}")
        with self.signal_sync_lock:
            self.n_orders_being_processed[0] -= 1
            if self.n_orders_being_processed[0] == 0:
                self.signal_sync_condition.notify_all()
        return synapse

    def get_positions(self, synapse: template.protocol.GetPositions,
                      ) -> template.protocol.GetPositions:
        if self.should_fail_early(synapse, SynapseMethod.POSITION_INSPECTOR):
            return synapse

        miner_hotkey = synapse.dendrite.hotkey
        error_message = ""
        try:
            hotkey = synapse.dendrite.hotkey
            # Return the last n positions
            positions = self.position_manager.get_all_miner_positions(hotkey, sort_positions=True)[-30:]
            synapse.positions = [position.to_dict() for position in positions]
            bt.logging.info(f"Sending {len(positions)} positions back to miner: " + hotkey)
        except Exception as e:
            error_message = f"Error in GetPositions for [{miner_hotkey}] with error [{e}]. Perhaps the position was being written to disk at the same time."
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False
        synapse.error_message = error_message
        return synapse

    def get_data(self, synapse: template.protocol.GetDashData,
                      ) -> template.protocol.GetDashData:
        if self.should_fail_early(synapse, SynapseMethod.DASHBOARD):
            return synapse

        miner_hotkey = synapse.dendrite.hotkey
        error_message = ""
        try:
            stats = generate_miner_statistics_data(time_now=TimeUtil.now_in_millis(), checkpoints=True, selected_miner_hotkeys=[miner_hotkey])
            positions = generate_request_core(time_now=TimeUtil.now_in_millis(), selected_miner_hotkeys=[miner_hotkey])
            dash_data = {"statistics": stats, **positions}

            if not stats["data"]:
                error_message = f"Validator {self.wallet.hotkey.ss58_address} has no stats for miner {miner_hotkey}"
            elif not positions:
                error_message = f"Validator {self.wallet.hotkey.ss58_address} has no positions for miner {miner_hotkey}"

            synapse.data = dash_data
            bt.logging.info("Sending data back to miner: " + miner_hotkey)
        except Exception as e:
            error_message = f"Error in GetData for [{miner_hotkey}] with error [{e}]."
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False
        synapse.error_message = error_message
        return synapse

    def receive_checkpoint(self, synapse: template.protocol.ValidatorCheckpoint) -> template.protocol.ValidatorCheckpoint:
        """
        receive checkpoint request, and ensure that only requests received from valid validators are processed.
        """
        sender_hotkey = synapse.dendrite.hotkey

        # validator responds to poke from validator and attaches their checkpoint
        if sender_hotkey in [axon.hotkey for axon in self.p2p_syncer.get_validators()]:
            synapse.validator_receive_hotkey = self.wallet.hotkey.ss58_address

            bt.logging.info(f"Received checkpoint request poke from validator hotkey [{sender_hotkey}].")
            if self.should_fail_early(synapse, SynapseMethod.CHECKPOINT):
                return synapse

            error_message = ""
            try:
                with self.checkpoint_lock:
                    # reset checkpoint after 10 minutes
                    if TimeUtil.now_in_millis() - self.last_checkpoint_time > 1000 * 60 * 10:
                        self.encoded_checkpoint = ""
                    # save checkpoint so we only generate it once for all requests
                    if not self.encoded_checkpoint:
                        # get our current checkpoint
                        self.last_checkpoint_time = TimeUtil.now_in_millis()
                        checkpoint_dict = generate_request_core(time_now=self.last_checkpoint_time)

                        # compress json and encode as base64 to keep as a string
                        checkpoint_str = json.dumps(checkpoint_dict, cls=CustomEncoder)
                        compressed = gzip.compress(checkpoint_str.encode("utf-8"))
                        self.encoded_checkpoint = base64.b64encode(compressed).decode("utf-8")

                    # only send a checkpoint if we are an up-to-date validator
                    timestamp = self.timestamp_manager.get_last_order_timestamp()
                    if TimeUtil.now_in_millis() - timestamp < 1000 * 60 * 60 * 10:  # validators with no orders processed in 10 hrs are considered stale
                        synapse.checkpoint = self.encoded_checkpoint
                    else:
                        error_message = f"Validator is stale, no orders received in 10 hrs, last order timestamp {timestamp}, {round((TimeUtil.now_in_millis() - timestamp)/(1000 * 60 * 60))} hrs ago"
            except Exception as e:
                error_message = f"Error processing checkpoint request poke from [{sender_hotkey}] with error [{e}]"
                bt.logging.error(traceback.format_exc())

            if error_message == "":
                synapse.successfully_processed = True
            else:
                bt.logging.error(error_message)
                synapse.successfully_processed = False
            synapse.error_message = error_message
            bt.logging.success(f"Sending checkpoint back to validator [{sender_hotkey}]")
        else:
            bt.logging.info(f"Received a checkpoint poke from non validator [{sender_hotkey}]")
            synapse.error_message = "Rejecting checkpoint poke from non validator"
            synapse.successfully_processed = False
        return synapse

# This is the main function, which runs the miner.
if __name__ == "__main__":
    validator = Validator()
    validator.main()
