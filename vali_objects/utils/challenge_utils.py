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

class ChallengeBase:
    def __init__(self, config):
        self.config = config
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.last_metagraph_update_time_s = 0
        self.last_update_time_s = 0

    def _update_metagraph(self):
        if time.time() - self.last_metagraph_update_time_s < 60 * 5:
            return
        
        bt.logging.info("updating metagraph in set weights...")
        self.metagraph.sync(subtensor=self.subtensor)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info("metagraph updated.")
        self.last_metagraph_update_time_s = time.time()

class SubtensorWeightSetter(ChallengeBase):
    def __init__(self, config, wallet):
        super().__init__(config)
        self.wallet = wallet

    def set_weights(self):
        bt.logging.info("running set weights")
        bt.logging.info(f"subtensor: {self.subtensor}")

        while True:
            try:
                self._update_metagraph()
                self._handle_weights()
                
            except Exception:
                bt.logging.error(traceback.format_exc())

    def _handle_weights(self):
        if time.time() - self.last_update_time_s < ValiConfig.SET_WEIGHT_REFRESH_TIME_S:
            time.sleep(1)
            return
        
        cached_eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )

        return_per_netuid = self._calculate_return_per_netuid(self.metagraph.hotkeys, cached_eliminations)
        bt.logging.info(f"return per uid [{return_per_netuid}]")
        if len(return_per_netuid) == 0:
            bt.logging.info("no returns to set weights with. Do nothing for now.")
        else:
            filtered_results, filtered_netuids = self._filter_results(return_per_netuid)
            scaled_transformed_list = self._transform_and_scale_results(filtered_results)
            self._set_subtensor_weights(filtered_netuids, scaled_transformed_list)
        self.last_update_time_s = time.time()

    def _calculate_return_per_netuid(self, hotkeys, eliminations):
        return_per_netuid = {}
        netuid_returns = []
        netuids = []

        hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
            hotkeys,
            sort_positions=True,
            eliminations=eliminations,
            acceptable_position_end_ms=TimeUtil.timestamp_to_millis(
                TimeUtil.generate_start_timestamp(ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS)
            ),
        )

        # have to have a minimum number of positions during the period
        # this removes anyone who got lucky on a couple trades
        for hotkey, positions in hotkey_positions.items():
            if len(positions) <= ValiConfig.SET_WEIGHT_MINIMUM_POSITIONS:
                continue
            per_position_return = PositionUtils.get_return_per_closed_position(positions)
            last_positional_return = 1
            if len(per_position_return) > 0:
                last_positional_return = per_position_return[len(per_position_return) - 1]
            netuid_returns.append(last_positional_return)
            netuid = hotkeys.index(hotkey)
            return_per_netuid[netuid] = last_positional_return
            netuids.append(netuid)

        return return_per_netuid

    def _filter_results(self, return_per_netuid):
        mean = np.mean(list(return_per_netuid.values()))
        std_dev = np.std(list(return_per_netuid.values()))

        lower_bound = mean - 3 * std_dev
        bt.logging.debug(f"returns lower bound: [{lower_bound}]")

        if lower_bound < 0:
            lower_bound = 0

        filtered_results = [(k, v) for k, v in return_per_netuid.items() if lower_bound < v]
        filtered_netuids = np.array([x[0] for x in filtered_results])
        bt.logging.info(f"filtered results list [{filtered_results}]")
        bt.logging.info(f"filtered netuids list [{filtered_netuids}]")
        return filtered_results, filtered_netuids

    def _transform_and_scale_results(self, filtered_results):
        filtered_scores = np.array([x[1] for x in filtered_results])
        bt.logging.success("calculated filtered_scores {filtered_scores}".format(filtered_scores=filtered_scores))
        # Normalize the list using Z-score normalization
        transformed_results = yeojohnson(filtered_scores, lmbda=500)
        bt.logging.success("calculated transformed_results {transformed_results}".format(transformed_results=transformed_results))
        if len(transformed_results) == 0:
            return []
        scaled_transformed_list = Scaling.min_max_scalar_list(transformed_results)
        return scaled_transformed_list

    def _set_subtensor_weights(self, filtered_netuids, scaled_transformed_list):
        result = self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=filtered_netuids,
            weights=scaled_transformed_list,
        )

        if result:
            bt.logging.success("Successfully set weights.")
        else:
            bt.logging.error("Failed to set weights.")


class MDDChecker(ChallengeBase):
    def __init__(self, config):
        super().__init__(config)        
        self.hotkeys_to_eliminate = {}


    def mdd_check(self):
        bt.logging.info("running mdd checker")
        bt.logging.info(f"Subtensor: {self.subtensor}")

        while True:
            try:
                self._update_metagraph()
                self._handle_eliminations()
            except Exception:
                bt.logging.error(traceback.format_exc())


    def _handle_eliminations(self):
        if time.time() - self.last_update_time_s < ValiConfig.MDD_CHECK_REFRESH_TIME_S:
            time.sleep(1)
            return
        
        bt.logging.debug("checking mdd.")

        updated_eliminations = self._update_elimination_and_copying_data()

        try:
            secrets = ValiUtils.get_secrets()
            all_trade_pairs = [trade_pair for trade_pair in TradePair]
            twelvedata = TwelveDataService(api_key=secrets["twelvedata_apikey"])
            signal_closing_prices = twelvedata.get_closes(trade_pairs=all_trade_pairs)
            hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
                self.metagraph.hotkeys, sort_positions=True, eliminations=updated_eliminations
            )

            for hotkey, positions in hotkey_positions.items():
                current_dd = self._process_positions(hotkey, positions, signal_closing_prices, updated_eliminations)

                if self._is_beyond_mdd(current_dd, hotkey):
                    updated_eliminations.append(hotkey)

            self._write_updated_eliminations(updated_eliminations)

        except Exception:
            bt.logging.error(traceback.format_exc())

        self.last_update_time_s = time.time()

    def _load_elimination_and_copying_data(self):
        eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )
        miner_copying = ValiUtils.get_vali_json_file(ValiBkpUtils.get_miner_copying_dir())
        return eliminations, miner_copying

    def _update_elimination_and_copying_data(self):
        cached_eliminations = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_eliminations_dir(), ValiUtils.ELIMINATIONS
        )
        cached_miner_copying = ValiUtils.get_vali_json_file(ValiBkpUtils.get_miner_copying_dir())

        # remove miners who've already been deregd. only keep miners who are still registered
        updated_eliminations = [elimination for elimination in cached_eliminations if elimination in self.metagraph.hotkeys]
        updated_miner_copying = {mch: mc for mch, mc in cached_miner_copying.items() if mch in self.metagraph.hotkeys}
        ValiBkpUtils.write_file(ValiBkpUtils.get_miner_copying_dir(), updated_miner_copying)
        return updated_eliminations

    def _process_positions(self, hotkey, positions, signal_closing_prices, updated_eliminations):
        current_dd = 1
        if len(positions) > 0:
            per_position_return = PositionUtils.get_return_per_closed_position(positions)
            bt.logging.debug(f"per position return [{per_position_return}]")

            if len(per_position_return) > 0:
                max_portfolio_return = max(per_position_return)
                max_index = per_position_return.index(max_portfolio_return)

                if max_portfolio_return < current_dd:
                    max_portfolio_return = current_dd

                bt.logging.info(f"max port return for [{hotkey}] is [{max_portfolio_return}]")

                for i, position_return in enumerate(per_position_return):
                    if i > max_index:
                        closed_position_return = position_return / max_portfolio_return
                        if self._is_beyond_mdd(closed_position_return, hotkey):
                            updated_eliminations.append(hotkey)

                last_position_ind = len(per_position_return) - 1
                if max_index != last_position_ind:
                    current_dd = per_position_return[last_position_ind] / max_portfolio_return
            else:
                bt.logging.debug(f"no existing closed positions for [{hotkey}]")

            if hotkey not in updated_eliminations:
                bt.logging.debug(f"reviewing open positions for [{hotkey}]")
                open_positions = [position for position in positions if not position.is_closed_position]
                open_position_trade_pairs = {
                    position.position_uuid: position.trade_pair for position in open_positions
                }

                bt.logging.debug(f"number of open positions [{len(open_positions)}]")

                for open_position in open_positions:
                    position_closing_price = signal_closing_prices[
                        open_position_trade_pairs[open_position.position_uuid]
                    ]
                    current_return = open_position.calculate_unrealized_pnl(position_closing_price)
                    open_position.current_return = current_return
                    current_dd *= current_return

                    bt.logging.debug(f"updating - current return [{current_return}]")
                    bt.logging.debug(f"updating - current dd [{current_dd}]")
                    bt.logging.debug(f"updating - net leverage [{open_position._net_leverage}]")

        return current_dd

    def _is_beyond_mdd(self, dd, miner_hotkey):
        time_now = TimeUtil.generate_start_timestamp(0)
        if (dd < ValiConfig.MAX_DAILY_DRAWDOWN and time_now.hour == 0 and time_now.minute < 5) or (
            dd < ValiConfig.MAX_TOTAL_DRAWDOWN
        ):
            miner_dir = ValiBkpUtils.get_miner_dir(miner_hotkey)
            bt.logging.debug(f"miner_hotkey [{miner_hotkey}] with miner dd [{dd}]")
            bt.logging.info(
                f"miner eliminated with hotkey [{miner_hotkey}] with max dd of [{dd}]. "
                f"Removing miner dir [{miner_dir}]"
            )
            try:
                shutil.rmtree(miner_dir)
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found [{miner_dir}]")
            return True
        return False

    def _write_updated_eliminations(self, updated_eliminations):
        vali_elims = {ValiUtils.ELIMINATIONS: updated_eliminations}
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(), vali_elims)

