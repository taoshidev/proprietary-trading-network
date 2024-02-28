# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc
import multiprocessing

# Step 1: Import necessary libraries and modules
import os
import shutil
import threading
import time
import uuid
from typing import Tuple

import numpy as np
from scipy.stats import yeojohnson

import template
import argparse
import traceback
import bittensor as bt

from data_generator.twelvedata import TwelveData
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.scaling.scaling import Scaling
from vali_objects.utils.challenge_utils import ChallengeUtils
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.position import Position
from vali_objects.vali_dataclasses.signal import Signal
from vali_objects.enums.order_type_enum import OrderTypeEnum
from vali_objects.utils.vali_utils import ValiUtils


def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom miner arguments to the parser.
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
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


# Main takes the config and starts the miner.
def main(config):
    # def set_weights(netuid, wallet):
    #     while True:
    #         try:
    #             time_now = TimeUtil.generate_start_timestamp(0)
    #             if time_now.minute in ValiConfig.SET_WEIGHT_INTERVALS:
    #                 hotkeys = metagraph.hotkeys
    #                 eliminations = ValiUtils.get_vali_json_file(
    #                     ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
    #                 )
    #
    #                 return_per_netuid = {}
    #
    #                 netuid_returns = []
    #                 netuids = []
    #
    #                 hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
    #                     hotkeys,
    #                     sort_positions=True,
    #                     eliminations=eliminations,
    #                     acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
    #                         TimeUtil.generate_start_timestamp(
    #                             ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
    #                         )
    #                     ),
    #                 )
    #
    #                 for hotkey, positions in hotkey_positions.items():
    #                     # have to have a minimum number of positions during the period
    #                     # this removes anyone who got lucky on a couple trades
    #                     if len(positions) > ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
    #                         per_position_return = (
    #                             PositionUtils.get_return_per_closed_position(positions)
    #                         )
    #                         last_positional_return = 1
    #                         if len(per_position_return) > 0:
    #                             last_positional_return = per_position_return[
    #                                 len(per_position_return) - 1
    #                             ]
    #                         netuid_returns.append(last_positional_return)
    #                         netuid = metagraph.hotkeys.index(hotkey)
    #                         return_per_netuid[netuid] = last_positional_return
    #                         netuids.append(netuid)
    #
    #                 bt.logging.info(f"return per uid [{return_per_netuid}]")
    #
    #                 mean = np.mean(netuid_returns)
    #                 std_dev = np.std(netuid_returns)
    #
    #                 lower_bound = mean - 3 * std_dev
    #                 bt.logging.debug(f"returns lower bound: [{lower_bound}]")
    #
    #                 if lower_bound < 0:
    #                     lower_bound = 0
    #
    #                 filtered_results = [
    #                     (k, v) for k, v in return_per_netuid.items() if lower_bound < v
    #                 ]
    #                 filtered_scores = np.array([x[1] for x in filtered_results])
    #                 filtered_netuids = np.array([x[0] for x in filtered_results])
    #
    #                 # Normalize the list using Z-score normalization
    #                 transformed_results = yeojohnson(filtered_scores, lmbda=500)
    #                 scaled_transformed_list = Scaling.min_max_scalar_list(
    #                     transformed_results
    #                 )
    #
    #                 bt.logging.info(f"filtered results list [{filtered_results}]")
    #                 bt.logging.info(
    #                     f"scaled transformed list [{scaled_transformed_list}]"
    #                 )
    #
    #                 result = subtensor.set_weights(
    #                     netuid=netuid,
    #                     wallet=wallet,
    #                     uids=filtered_netuids,
    #                     weights=scaled_transformed_list,
    #                 )
    #
    #                 if result:
    #                     bt.logging.success("Successfully set weights.")
    #                 else:
    #                     bt.logging.error("Failed to set weights.")
    #                 time.sleep(60)
    #         except Exception:
    #             bt.logging.error(traceback.format_exc())
    #             time.sleep(15)
    #
    # def mdd_check():
    #     def _is_beyond_mdd(dd, miner_hotkey):
    #         if (
    #             dd > ValiConfig.MAX_DAILY_DRAWDOWN
    #             and time_now.hour == 0
    #             and time_now.minute < 5
    #         ) or (dd < ValiConfig.MAX_TOTAL_DRAWDOWN):
    #             miner_dir = ValiBkpUtils.get_miner_dir(miner_hotkey)
    #             bt.logging.debug(
    #                 f"miner_hotkey [{miner_hotkey}] with miner dd [{current_dd}]"
    #             )
    #             bt.logging.info(
    #                 f"miner eliminated with hotkey [{hotkey}] with "
    #                 f"max dd of [{current_dd}]. "
    #                 f"Removing miner dir [{miner_dir}]"
    #             )
    #             try:
    #                 shutil.rmtree(miner_dir)
    #             except FileNotFoundError:
    #                 bt.logging.info(f"miner dir not found [{miner_dir}]")
    #             return True
    #         return False
    #
    #     while True:
    #         time_now = TimeUtil.generate_start_timestamp(0)
    #
    #         if time_now.second < 15:
    #             bt.logging.debug("checking mdd.")
    #
    #             _eliminations = ValiUtils.get_vali_json_file(
    #                 ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
    #             )
    #             _miner_copying = ValiUtils.get_vali_json_file(
    #                 ValiBkpUtils.get_miner_copying_dir()
    #             )
    #
    #             try:
    #                 all_trade_pairs = [trade_pair for trade_pair in TradePair]
    #                 twelvedata = TwelveData(api_key=secrets["twelvedata_apikey"])
    #                 signal_closing_prices = twelvedata.get_closes(
    #                     trade_pairs=all_trade_pairs
    #                 )
    #                 hotkeys = metagraph.hotkeys
    #
    #                 # remove miners who've already been deregd
    #                 # only keep miners who are still registered
    #                 updated_eliminations = [
    #                     elimination
    #                     for elimination in _eliminations
    #                     if elimination in hotkeys
    #                 ]
    #
    #                 # update miner copying with miners who've been deregd
    #                 # only keep miners who are still registered
    #                 updated_miner_copying = {
    #                     mch: mc for mch, mc in _miner_copying.items() if mch in hotkeys
    #                 }
    #
    #                 ValiBkpUtils.write_vali_file(
    #                     ValiBkpUtils.get_miner_copying_dir(), updated_miner_copying
    #                 )
    #
    #                 hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
    #                     hotkeys, sort_positions=True, eliminations=updated_eliminations
    #                 )
    #                 for hotkey, positions in hotkey_positions.items():
    #                     current_dd = 1
    #                     if len(positions) > 0:
    #                         per_position_return = (
    #                             PositionUtils.get_return_per_closed_position(positions)
    #                         )
    #
    #                         if len(per_position_return) > 0:
    #                             # get the max return in order to calculate the current dd
    #                             max_portfolio_return = max(per_position_return)
    #                             max_index = per_position_return.index(
    #                                 max_portfolio_return
    #                             )
    #
    #                             bt.logging.info(
    #                                 f"max port return for [{hotkey}] "
    #                                 f"is [{max_portfolio_return}]"
    #                             )
    #
    #                             # check to see if any closed positions beyond the max index
    #                             # already passed max dd as a safety measure
    #
    #                             for i, position_return in enumerate(
    #                                 per_position_return
    #                             ):
    #                                 # only check for the positions after the max positional return
    #                                 if i > max_index:
    #                                     # gets the drawdown as a decimal for comparative purposes
    #                                     closed_position_return = (
    #                                         position_return / max_portfolio_return
    #                                     )
    #                                     # beyond max daily dd on start of new day
    #                                     # or beyond total max drawdown at any point in time
    #                                     if _is_beyond_mdd(
    #                                         closed_position_return, hotkey
    #                                     ):
    #                                         updated_eliminations.append(hotkey)
    #
    #                             # get current dd using the last closed position against the max
    #                             last_position_ind = len(per_position_return) - 1
    #                             if max_index != last_position_ind:
    #                                 current_dd = (
    #                                     per_position_return[last_position_ind]
    #                                     / max_portfolio_return
    #                                 )
    #                         else:
    #                             bt.logging.debug(
    #                                 f"no existing closed positions for [{hotkey}]"
    #                             )
    #
    #                         if hotkey not in updated_eliminations:
    #                             bt.logging.debug(
    #                                 f"reviewing open positions for [{hotkey}]"
    #                             )
    #
    #                             # review open positions
    #                             open_positions = [
    #                                 position
    #                                 for position in positions
    #                                 if not position.is_closed_position
    #                             ]
    #                             open_position_trade_pairs = {
    #                                 position.position_uuid: position.trade_pair
    #                                 for position in open_positions
    #                             }
    #
    #                             bt.logging.debug(
    #                                 f"number of open positions [{len(open_positions)}]"
    #                             )
    #
    #                             for open_position in open_positions:
    #                                 # get trade pair closing price using position uuid map
    #                                 position_closing_price = signal_closing_prices[
    #                                     open_position_trade_pairs[
    #                                         open_position.position_uuid
    #                                     ]
    #                                 ]
    #                                 # get return, set current return, and update the current dd
    #                                 current_return = (
    #                                     open_position.calculate_unrealized_pnl(
    #                                         position_closing_price
    #                                     )
    #                                 )
    #                                 open_position.current_return = current_return
    #                                 current_dd *= current_return
    #
    #                             if _is_beyond_mdd(current_dd, hotkey):
    #                                 updated_eliminations.append(hotkey)
    #                 vali_elims = {ValiUtils.ELIMINATIONS: eliminations}
    #                 ValiBkpUtils.write_vali_file(
    #                     ValiBkpUtils.get_eliminations_dir(), vali_elims
    #                 )
    #                 time.sleep(15)
    #             except Exception:
    #                 bt.logging.error(traceback.format_exc())
    #                 time.sleep(15)

    secrets = ValiUtils.get_secrets()

    if secrets is None:
        raise Exception(
            "unable to get secrets data from "
            "validation/secrets.json. Please ensure it exists"
        )

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} "
        f"on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} is not registered to chain "
            f"connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
        bt.logging.debug("got to blacklisting rs")

        # Ignore requests from unrecognized entities.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )

        # don't process eliminated miners
        if synapse.dendrite.hotkey in eliminations:
            # Ignore requests from eliminated hotkeys
            bt.logging.trace(f"Eliminated hotkey {synapse.dendrite.hotkey}")
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    def rs_priority_fn(synapse: template.protocol.SendSignal) -> float:
        bt.logging.debug("got to prioritization rs")
        # simply just prioritize based on uid as it's not significant
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.uids[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    # This is the core validator function to receive a signal
    def receive_signal(
        synapse: template.protocol.SendSignal,
    ) -> template.protocol.SendSignal:
        bt.logging.info(f"received signal request")

        # pull miner hotkey to reference in various activities
        miner_hotkey = synapse.dendrite.hotkey

        eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )

        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""

        order_type_lookup = {
            order_type.value: order_type for order_type in OrderTypeEnum
        }
        trade_pair_lookup = {pair.value: pair for pair in TradePair}

        # convert every signal to orders
        # set the closing price for every order
        signals = [
            Signal(
                trade_pair_lookup[signal["trade_pair"]],
                order_type_lookup[signal["order_type"]],
                signal["leverage"],
            )
            for signal in synapse.signals
        ]

        bt.logging.info(f"received [{len(signals)}] " f"signal request(s)")

        # gather open positions and see which trade pairs have an open position
        open_positions = PositionUtils.get_all_miner_positions(
            miner_hotkey, only_open_positions=True
        )
        open_position_trade_pairs = {
            position.trade_pair: position for position in open_positions
        }

        # ensure all signals are for an existing trade pair
        # can determine using the fees object
        try:
            for signal in signals:
                if signal.trade_pair not in ValiConfig.TRADE_PAIR_FEES:
                    raise SignalException(
                        f"miner [{synapse.dendrite.hotkey}] incorrectly "
                        f"sent trade pair [{signal.trade_pair}]"
                    )
                else:
                    trade_pair = signal.trade_pair
                    # trade pair exists, convert signal to order
                    twelvedata = TwelveData(api_key=secrets["twelvedata_apikey"])
                    signal_closing_price = twelvedata.get_close(trade_pair=trade_pair)[
                        trade_pair
                    ]
                    signal_to_order = Order(
                        trade_pair=trade_pair,
                        order_type=signal.order_type,
                        leverage=signal.leverage,
                        price=signal_closing_price,
                        processed_ms=TimeUtil.now_in_millis(),
                        order_uuid=str(uuid.uuid4()),
                    )
                    # if a position already exists, add the order to it and
                    # close if the order generates a FLAT
                    if trade_pair in open_position_trade_pairs:
                        bt.logging.debug("adding to existing position")
                        open_position = open_position_trade_pairs[trade_pair]
                        open_position.orders.append(signal_to_order)
                        open_position.update_position()
                        ValiUtils.save_miner_position(
                            miner_hotkey, open_position.position_uuid, open_position
                        )
                    else:
                        bt.logging.debug("processing new position")
                        # if the order is FLAT ignore and log
                        if signal_to_order.order_type != OrderTypeEnum.FLAT:
                            # if a position doesn't exist, then make a new one
                            open_position = Position(
                                miner_hotkey=miner_hotkey,
                                position_uuid=str(uuid.uuid4()),
                                open_ms=TimeUtil.now_in_millis(),
                                open_price=signal_closing_price,
                                trade_pair=trade_pair,
                                orders=[signal_to_order],
                            )
                            ValiUtils.save_miner_position(
                                miner_hotkey, open_position.position_uuid, open_position
                            )
                        else:
                            raise SignalException(
                                f"miner [{synapse.dendrite.hotkey}] sent a "
                                f"FLAT order with no existing position."
                            )

                        # check to see if order is similar to existing order
                        is_similar_order = (
                            PositionUtils.is_order_similar_to_positional_orders(
                                open_position.open_ms,
                                signal_to_order,
                                hotkey=miner_hotkey,
                                hotkeys=metagraph.hotkeys,
                            )
                        )
                        miner_copying_json = ValiUtils.get_vali_json_file(
                            ValiBkpUtils.get_miner_copying_dir()
                        )
                        current_hotkey_mc = miner_copying_json[miner_hotkey]
                        if is_similar_order:
                            current_hotkey_mc += ValiConfig.MINER_COPYING_WEIGHT
                            if current_hotkey_mc > 1:
                                eliminations.append(miner_hotkey)
                                # updating both elims and miner copying
                                miner_copying_json[miner_hotkey] = current_hotkey_mc
                                ValiBkpUtils.write_vali_file(
                                    ValiBkpUtils.get_miner_copying_dir(),
                                    miner_copying_json,
                                )
                                ValiBkpUtils.write_vali_file(
                                    ValiBkpUtils.get_eliminations_dir(), eliminations
                                )
                                raise SignalException(
                                    f"miner eliminated for signal copying [{miner_hotkey}]."
                                )
                        else:
                            if current_hotkey_mc > 0:
                                current_hotkey_mc -= ValiConfig.MINER_COPYING_WEIGHT
                                # updating miner copying file
                                miner_copying_json[miner_hotkey] = current_hotkey_mc
                                ValiBkpUtils.write_vali_file(
                                    ValiBkpUtils.get_miner_copying_dir(),
                                    miner_copying_json,
                                )
                        bt.logging.info(
                            f"updated miner copying - [{miner_copying_json[miner_hotkey]}]"
                        )
                    open_position.log_position_status()
        except SignalException as e:
            error_message = str(e)
            bt.logging.warning(error_message)
        except Exception as e:
            error_message = e
            bt.logging.error(traceback.format_exc())

        synapse.received = 1
        synapse.error_message = error_message

        return synapse

    # Step 5: Build and link vali functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    bt.logging.info(f"setting port [{config.axon.port}]")
    bt.logging.info(f"setting external port [{config.axon.external_port}]")
    axon = bt.axon(
        wallet=wallet, port=config.axon.port, external_port=config.axon.external_port
    )
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn=receive_signal,
        blacklist_fn=rs_blacklist_fn,
        priority_fn=rs_priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving attached axons on network:"
        f" {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # see if files exist and if not set them to empty
    _eliminations = ValiUtils.get_vali_json_file(
        ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
    )

    ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_dir())
    _miner_copying = ValiUtils.get_vali_json_file(ValiBkpUtils.get_miner_copying_dir())

    if len(_eliminations) == 0:
        ValiBkpUtils.write_vali_file(
            ValiBkpUtils.get_eliminations_dir(), {ValiUtils.ELIMINATIONS: []}
        )

    if len(_miner_copying) == 0:
        miner_copying_file = {hotkey: 0 for hotkey in metagraph.hotkeys}
        ValiBkpUtils.write_vali_file(
            ValiBkpUtils.get_miner_copying_dir(), miner_copying_file
        )

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    run_mdd_check = threading.Thread(target=ChallengeUtils.mdd_check, args=(config,))
    run_mdd_check.start()

    run_set_weights = threading.Thread(
        target=ChallengeUtils.set_weights, args=(config, wallet)
    )
    run_set_weights.start()

    # Step 6: Keep the vali alive
    # This loop maintains the vali's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    while True:
        try:
            # Update our metagraph every 10 minutes
            bt.logging.info("Updating metagraph.")
            metagraph.sync(subtensor=subtensor)
            metagraph = subtensor.metagraph(config.netuid)
            bt.logging.info(f"Metagraph updated: {metagraph}")
            time.sleep(120)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception:
            bt.logging.error(traceback.format_exc())

    run_mdd_check.join()
    run_set_weights.join()


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config())
