# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import traceback
import time

from data_generator.twelvedata_service import TwelveDataService
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

class MDDChecker(CacheController):
    MAX_DAILY_DRAWDOWN = 'MAX_DAILY_DRAWDOWN'
    MAX_TOTAL_DRAWDOWN = 'MAX_TOTAL_DRAWDOWN'
    def __init__(self, config, metagraph, position_manager, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        secrets = ValiUtils.get_secrets()
        self.position_manager = position_manager
        assert self.running_unit_tests == self.position_manager.running_unit_tests
        self.all_trade_pairs = [trade_pair for trade_pair in TradePair]
        self.twelvedata = TwelveDataService(api_key=secrets["twelvedata_apikey"])

    def get_required_closing_prices(self, hotkey_positions):
        required_trade_pairs = set()
        for sorted_positions in hotkey_positions.values():
            for position in sorted_positions:
                # Only need live price for open positions
                if position.is_closed_position:
                    continue
                required_trade_pairs.add(position.trade_pair)

        trade_pairs_list = list(required_trade_pairs)
        if len(trade_pairs_list) == 0:
            return {}
        return self.twelvedata.get_closes(trade_pairs=trade_pairs_list)
    
    def mdd_check(self):
        if not self.refresh_allowed(ValiConfig.MDD_CHECK_REFRESH_TIME_MS):
            time.sleep(1)
            return

        bt.logging.info("running mdd checker")
        self._refresh_eliminations_in_memory_and_disk()

        hotkey_to_positions = self.position_manager.get_all_miner_positions_by_hotkey(
            self.metagraph.hotkeys, sort_positions=True,
            eliminations=self.eliminations
        )
        signal_closing_prices = self.get_required_closing_prices(hotkey_to_positions)
        for hotkey, sorted_positions in hotkey_to_positions.items():
            self._search_for_miner_dd_failures(hotkey, sorted_positions, signal_closing_prices)

        self._write_eliminations_from_memory_to_disk()

        self.set_last_update_time()

    def _replay_all_closed_positions(self, hotkey, sorted_positions, current_dd):
        elimination_occurred = False
        sorted_per_position_return = self.position_manager.get_return_per_closed_position(sorted_positions)
        if len(sorted_per_position_return) == 0:
            bt.logging.info(f"no existing closed positions for [{hotkey}]")
            return elimination_occurred, current_dd

        # Already sorted
        for position_return in sorted_per_position_return:
            mdd_failure = self._is_beyond_mdd(position_return)
            if mdd_failure:
                self.position_manager.close_open_positions_for_miner(hotkey)
                self.append_elimination_row(hotkey, position_return, mdd_failure)
                elimination_occurred = True
                return elimination_occurred, current_dd

        # Replay of closed positions complete. If
        return elimination_occurred, sorted_per_position_return[-1]


    def _search_for_miner_dd_failures(self, hotkey, sorted_positions, signal_closing_prices):
        seen_trade_pairs = set()
        current_dd = 1
        # Log sorted positions length
        if len(sorted_positions) == 0:
            return
        # Already eliminated
        if self._hotkey_in_eliminations(hotkey):
            return

        elimination_occurred, current_dd = self._replay_all_closed_positions(hotkey, sorted_positions, current_dd)
        if elimination_occurred:
            return

        open_positions = []
        closed_positions = []
        for position in sorted_positions:
            if position.is_closed_position:
                closed_positions.append(position)
            else:
                open_positions.append(position)
        bt.logging.info(f"reviewing open positions for [{hotkey}]. Current_dd [{current_dd}]. n positions open [{len(open_positions)} / {len(sorted_positions)}]")

        open_position_trade_pairs = {
            position.position_uuid: position.trade_pair for position in open_positions
        }

        # Enforce only one open position per trade pair
        for open_position in open_positions:
            if open_position.trade_pair.trade_pair_id in seen_trade_pairs:
                raise ValueError(f"Miner [{hotkey}] has multiple open positions for trade pair [{open_position.trade_pair}]. Please restore cache.")
            else:
                seen_trade_pairs.add(open_position.trade_pair.trade_pair_id)
            realtime_price = signal_closing_prices[
                open_position_trade_pairs[open_position.position_uuid]
            ]
            open_position.set_returns(realtime_price, open_position.get_net_leverage())

            #bt.logging.success(f"current return with fees for [{open_position.position_uuid}] is [{open_position.return_at_close}]")
            current_dd *= open_position.return_at_close

        for position in closed_positions:
            seen_trade_pairs.add(position.trade_pair.trade_pair_id)
        # Log the dd for this miner and the positions trade_pairs they are in as well as total number of positions
        bt.logging.info(f"MDD checker -- current dd for [{hotkey}] is [{current_dd}]. Seen trade pairs: {seen_trade_pairs}. n_positions: [{len(sorted_positions)}]")
        mdd_failure = self._is_beyond_mdd(current_dd)
        if mdd_failure:
            self.position_manager.close_open_positions_for_miner(hotkey)
            self.append_elimination_row(hotkey, current_dd, mdd_failure)

    def _is_beyond_mdd(self, dd):
        time_now = TimeUtil.generate_start_timestamp(0)
        if (dd < ValiConfig.MAX_DAILY_DRAWDOWN and time_now.hour == 0 and time_now.minute < 5):
            return MDDChecker.MAX_DAILY_DRAWDOWN
        elif (dd < ValiConfig.MAX_TOTAL_DRAWDOWN):
            return MDDChecker.MAX_TOTAL_DRAWDOWN
        else:
            return None







                

