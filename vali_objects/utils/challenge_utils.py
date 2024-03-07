import shutil
import traceback
import time

import numpy as np
from scipy.stats import yeojohnson

from data_generator.twelvedata_service import TwelveDataService
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from vali_objects.scaling.scaling import Scaling
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt


class ChallengeUtils:
    @staticmethod
    def set_weights(config, wallet):
        bt.logging.info("running set weights")

        subtensor = bt.subtensor(config=config)
        bt.logging.info(f"subtensor: {subtensor}")
        metagraph = subtensor.metagraph(config.netuid)

        while True:
            try:
                time_now = TimeUtil.generate_start_timestamp(0)
                if time_now.minute in ValiConfig.SET_WEIGHT_INTERVALS:
                    # updating metagraph
                    bt.logging.info("updating metagraph in set weights...")
                    metagraph.sync(subtensor=subtensor)
                    metagraph = subtensor.metagraph(config.netuid)
                    bt.logging.info("metagraph updated.")
                    hotkeys = metagraph.hotkeys

                    eliminations = ValiUtils.get_vali_json_file(
                        ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
                    )

                    return_per_netuid = {}

                    netuid_returns = []
                    netuids = []

                    hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
                        hotkeys,
                        sort_positions=True,
                        eliminations=eliminations,
                        acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                            TimeUtil.generate_start_timestamp(
                                ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
                            )
                        ),
                    )

                    for hotkey, positions in hotkey_positions.items():
                        # have to have a minimum number of positions during the period
                        # this removes anyone who got lucky on a couple trades
                        if len(positions) > ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
                            per_position_return = (
                                PositionUtils.get_return_per_closed_position(positions)
                            )
                            last_positional_return = 1
                            if len(per_position_return) > 0:
                                last_positional_return = per_position_return[
                                    len(per_position_return) - 1
                                ]
                            netuid_returns.append(last_positional_return)
                            netuid = hotkeys.index(hotkey)
                            return_per_netuid[netuid] = last_positional_return
                            netuids.append(netuid)

                    bt.logging.info(f"return per uid [{return_per_netuid}]")

                    mean = np.mean(netuid_returns)
                    std_dev = np.std(netuid_returns)

                    lower_bound = mean - 3 * std_dev
                    bt.logging.debug(f"returns lower bound: [{lower_bound}]")

                    if lower_bound < 0:
                        lower_bound = 0

                    filtered_results = [
                        (k, v) for k, v in return_per_netuid.items() if lower_bound < v
                    ]
                    filtered_scores = np.array([x[1] for x in filtered_results])
                    filtered_netuids = np.array([x[0] for x in filtered_results])

                    # Normalize the list using Z-score normalization
                    transformed_results = yeojohnson(filtered_scores, lmbda=500)
                    scaled_transformed_list = Scaling.min_max_scalar_list(
                        transformed_results
                    )

                    bt.logging.info(f"filtered results list [{filtered_results}]")
                    bt.logging.info(
                        f"scaled transformed list [{scaled_transformed_list}]"
                    )

                    result = subtensor.set_weights(
                        netuid=netuid,
                        wallet=wallet,
                        uids=filtered_netuids,
                        weights=scaled_transformed_list,
                    )

                    if result:
                        bt.logging.success("Successfully set weights.")
                    else:
                        bt.logging.error("Failed to set weights.")
                    time.sleep(60)
            except Exception:
                bt.logging.error(traceback.format_exc())
                time.sleep(15)

    @staticmethod
    def mdd_check(config):
        # TODO: Check with Arrash about adding a dict to track hot keys that need to be eliminated so that this function does not block when waiting the 1 hr for file deletion.
        bt.logging.info("running mdd checker")

        def _is_beyond_mdd(dd, miner_hotkey):
            if (
                dd < ValiConfig.MAX_DAILY_DRAWDOWN
                and time_now.hour == 0
                and time_now.minute < 5
            ) or (dd < ValiConfig.MAX_TOTAL_DRAWDOWN):
                miner_dir = ValiBkpUtils.get_miner_dir(miner_hotkey)
                bt.logging.debug(
                    f"miner_hotkey [{miner_hotkey}] with miner dd [{current_dd}]"
                )
                bt.logging.info(
                    f"miner eliminated with hotkey [{hotkey}] with "
                    f"max dd of [{current_dd}]. "
                    f"Removing miner dir [{miner_dir}]"
                )
                try:
                    shutil.rmtree(miner_dir)
                except FileNotFoundError:
                    bt.logging.info(f"miner dir not found [{miner_dir}]")
                return True
            return False

        subtensor = bt.subtensor(config=config)
        bt.logging.info(f"Subtensor: {subtensor}")
        metagraph = subtensor.metagraph(config.netuid)
        hotkeys = metagraph.hotkeys

        secrets = ValiUtils.get_secrets()

        while True:
            time_now = TimeUtil.generate_start_timestamp(0)

            if time_now.second < 15:
                if time_now.minute % 5 == 0:
                    # updating metagraph
                    bt.logging.info("updating metagraph in set weights...")
                    metagraph.sync(subtensor=subtensor)
                    metagraph = subtensor.metagraph(config.netuid)
                    bt.logging.info("metagraph updated.")
                    hotkeys = metagraph.hotkeys

                bt.logging.debug("checking mdd.")

                eliminations = ValiUtils.get_vali_json_file(
                    ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
                )
                miner_copying = ValiUtils.get_vali_json_file(
                    ValiBkpUtils.get_miner_copying_dir()
                )

                try:
                    all_trade_pairs = [trade_pair for trade_pair in TradePair]
                    twelvedata = TwelveDataService(api_key=secrets["twelvedata_apikey"])
                    signal_closing_prices = twelvedata.get_closes(
                        trade_pairs=all_trade_pairs
                    )
                    hotkeys = hotkeys

                    # remove miners who've already been deregd
                    # only keep miners who are still registered
                    updated_eliminations = [
                        elimination
                        for elimination in eliminations
                        if elimination in hotkeys
                    ]

                    # update miner copying with miners who've been deregd
                    # only keep miners who are still registered
                    updated_miner_copying = {
                        mch: mc for mch, mc in miner_copying.items() if mch in hotkeys
                    }

                    ValiBkpUtils.write_file(
                        ValiBkpUtils.get_miner_copying_dir(), updated_miner_copying
                    )

                    hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
                        hotkeys, sort_positions=True, eliminations=updated_eliminations
                    )
                    for hotkey, positions in hotkey_positions.items():
                        current_dd = 1
                        if len(positions) > 0:
                            per_position_return = (
                                PositionUtils.get_return_per_closed_position(positions)
                            )

                            bt.logging.debug(f"per position return [{per_position_return}]")

                            if len(per_position_return) > 0:
                                # get the max return in order to calculate the current dd
                                max_portfolio_return = max(per_position_return)
                                max_index = per_position_return.index(
                                    max_portfolio_return
                                )

                                if max_portfolio_return < current_dd:
                                    max_portfolio_return = current_dd

                                bt.logging.info(
                                    f"max port return for [{hotkey}] "
                                    f"is [{max_portfolio_return}]"
                                )

                                # check to see if any closed positions beyond the max index
                                # already passed max dd as a safety measure

                                for i, position_return in enumerate(
                                    per_position_return
                                ):
                                    # only check for the positions after the max positional return
                                    if i > max_index:
                                        # gets the drawdown as a decimal for comparative purposes
                                        closed_position_return = (
                                            position_return / max_portfolio_return
                                        )
                                        # beyond max daily dd on start of new day
                                        # or beyond total max drawdown at any point in time
                                        if _is_beyond_mdd(
                                            closed_position_return, hotkey
                                        ):
                                            updated_eliminations.append(hotkey)

                                # get current dd using the last closed position against the max
                                last_position_ind = len(per_position_return) - 1
                                if max_index != last_position_ind:
                                    current_dd = (
                                        per_position_return[last_position_ind]
                                        / max_portfolio_return
                                    )
                            else:
                                bt.logging.debug(
                                    f"no existing closed positions for [{hotkey}]"
                                )

                            if hotkey not in updated_eliminations:
                                bt.logging.debug(
                                    f"reviewing open positions for [{hotkey}]"
                                )

                                # review open positions
                                open_positions = [
                                    position
                                    for position in positions
                                    if not position.is_closed_position
                                ]
                                open_position_trade_pairs = {
                                    position.position_uuid: position.trade_pair
                                    for position in open_positions
                                }

                                bt.logging.debug(
                                    f"number of open positions [{len(open_positions)}]"
                                )

                                for open_position in open_positions:
                                    # get trade pair closing price using position uuid map
                                    position_closing_price = signal_closing_prices[
                                        open_position_trade_pairs[
                                            open_position.position_uuid
                                        ]
                                    ]
                                    # get return, set current return, and update the current dd
                                    current_return = (
                                        open_position.calculate_unrealized_pnl(
                                            position_closing_price
                                        )
                                    )

                                    open_position.current_return = current_return
                                    current_dd *= current_return

                                    bt.logging.debug(
                                        f"updating - current return [{current_return}]"
                                    )
                                    bt.logging.debug(
                                        f"updating - current dd [{current_dd}]"
                                    )
                                    bt.logging.debug(
                                        f"updating - net leverage [{open_position._net_leverage}]"
                                    )

                                if _is_beyond_mdd(current_dd, hotkey):
                                    updated_eliminations.append(hotkey)
                    vali_elims = {ValiUtils.ELIMINATIONS: updated_eliminations}
                    ValiBkpUtils.write_file(
                        ValiBkpUtils.get_eliminations_dir(), vali_elims
                    )
                    time.sleep(15)
                except Exception:
                    bt.logging.error(traceback.format_exc())
                    time.sleep(15)
