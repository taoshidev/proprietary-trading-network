# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import traceback
import time
from typing import List

from data_generator.twelvedata_service import TwelveDataService
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager

from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

class MDDChecker(CacheController):

    def __init__(self, config, metagraph, position_manager, eliminations_lock, running_unit_tests=False):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.n_miners_skipped_already_eliminated = 0
        self.n_eliminations_this_round = 0
        self.n_miners_mdd_checked = 0
        self.portfolio_max_dd_closed_positions = 0
        self.portfolio_max_dd_all_positions = 0
        secrets = ValiUtils.get_secrets()
        self.position_manager = position_manager
        assert self.running_unit_tests == self.position_manager.running_unit_tests
        self.all_trade_pairs = [trade_pair for trade_pair in TradePair]
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets)
        self.eliminations_lock = eliminations_lock
        self.reset_debug_counters()

    def reset_debug_counters(self, reset_global_counters=True):
        self.portfolio_max_dd_closed_positions = 0
        self.portfolio_max_dd_all_positions = 0
        if reset_global_counters:
            self.n_miners_skipped_already_eliminated = 0
            self.n_eliminations_this_round = 0
            self.n_miners_mdd_checked = 0
            self.max_portfolio_value_seen = 0
            self.min_portfolio_value_seen = float("inf")

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
        return self.live_price_fetcher.get_closes(trade_pairs=trade_pairs_list)
    
    def mdd_check(self):
        if not self.refresh_allowed(ValiConfig.MDD_CHECK_REFRESH_TIME_MS):
            time.sleep(1)
            return

        bt.logging.info("running mdd checker")
        self.reset_debug_counters()
        self._refresh_eliminations_in_memory()

        hotkey_to_positions = self.position_manager.get_all_miner_positions_by_hotkey(
            self.metagraph.hotkeys, sort_positions=True,
            eliminations=self.eliminations
        )
        signal_closing_prices = self.get_required_closing_prices(hotkey_to_positions)

        any_eliminations = False
        for hotkey, sorted_positions in hotkey_to_positions.items():
            self.reset_debug_counters(reset_global_counters=False)
            if self._search_for_miner_dd_failures(hotkey, sorted_positions, signal_closing_prices):
                any_eliminations = True
                self.n_eliminations_this_round += 1

        if any_eliminations:
            with self.eliminations_lock:
                self._write_eliminations_from_memory_to_disk()

        bt.logging.info(f"mdd checker completed for {self.n_miners_mdd_checked} miners. n_eliminations_this_round: "
                        f"{self.n_eliminations_this_round}. Max portfolio value seen: {self.max_portfolio_value_seen}. "
                        f"Min portfolio value seen: {self.min_portfolio_value_seen}. ")
        self.set_last_update_time(skip_message=True)

    def _replay_all_closed_positions(self, hotkey: str, sorted_closed_positions: List[Position]) -> (bool, float):
        max_cuml_return_so_far = 1.0
        cuml_return = 1.0

        if len(sorted_closed_positions) == 0:
            #bt.logging.info(f"no existing closed positions for [{hotkey}]")
            return False, cuml_return, max_cuml_return_so_far

        # Already sorted
        for position in sorted_closed_positions:
            position_return = position.return_at_close
            cuml_return *= position_return
            max_cuml_return_so_far = max(cuml_return, max_cuml_return_so_far)
            self._update_portfolio_debug_counters(cuml_return)
            drawdown = self.calculate_drawdown(cuml_return, max_cuml_return_so_far)
            self.portfolio_max_dd_closed_positions = max(drawdown, self.portfolio_max_dd_closed_positions)
            mdd_failure = self.is_drawdown_beyond_mdd(drawdown, time_now=TimeUtil.millis_to_datetime(position.close_ms))

            if mdd_failure:
                self.position_manager.close_open_positions_for_miner(hotkey)
                self.append_elimination_row(hotkey, drawdown, mdd_failure)
                return True, position_return, max_cuml_return_so_far

        # Replay of closed positions complete.
        return False, cuml_return, max_cuml_return_so_far

    def _update_open_position_returns_and_persist_to_disk(self, hotkey, open_position, signal_closing_prices) -> Position:
        """
        Setting the latest returns and persisting to disk for accurate MDD calculation and logging in get_positions

        Won't account for a position that was added in the time between mdd_check being called and this function
        being called. But that's ok as we will process such new positions the next round.
        """
        trade_pair_id = open_position.trade_pair.trade_pair_id
        realtime_price = signal_closing_prices[open_position.trade_pair]

        with self.position_manager.position_locks.get_lock(hotkey, trade_pair_id):
            # Position could have updated in the time between mdd_check being called and this function being called
            position = self.position_manager.get_miner_position_from_disk_using_position_in_memory(open_position)
            # It's unlikely but this position could have just closed.
            # That's ok since we haven't started MDD calculations yet.

            # Log return before calling set_returns
            #bt.logging.info(f"current return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{open_position.return_at_close}]. Position: {position}")
            if position.is_open_position:
                position.set_returns(realtime_price, open_position.get_net_leverage())
                self.position_manager.save_miner_position_to_disk(position)
            #bt.logging.info(f"updated return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{position.return_at_close}]. position: {position}")
            return position

    def _update_portfolio_debug_counters(self, portfolio_ret):
        self.max_portfolio_value_seen = max(self.max_portfolio_value_seen, portfolio_ret)
        self.min_portfolio_value_seen = min(self.min_portfolio_value_seen, portfolio_ret)

    def _search_for_miner_dd_failures(self, hotkey, sorted_positions, signal_closing_prices) -> bool:
        if len(sorted_positions) == 0:
            return False
        # Already eliminated
        if self._hotkey_in_eliminations(hotkey):
            self.n_miners_skipped_already_eliminated += 1
            return False

        self.n_miners_mdd_checked += 1
        open_positions = []
        closed_positions = []
        for position in sorted_positions:
            if position.is_closed_position:
                closed_positions.append(position)
            else:
                updated_position = self._update_open_position_returns_and_persist_to_disk(hotkey, position, signal_closing_prices)
                if updated_position.is_closed_position:
                    closed_positions.append(updated_position)
                else:
                    open_positions.append(updated_position)

        elimination_occurred, return_with_closed_positions, max_cuml_return_so_far = self._replay_all_closed_positions(hotkey, closed_positions)
        if elimination_occurred:
            return True

        # Enforce only one open position per trade pair
        seen_trade_pairs = set()
        return_with_open_positions = return_with_closed_positions
        for open_position in open_positions:
            #bt.logging.info(f"current return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{open_position.return_at_close}]")
            if open_position.trade_pair.trade_pair_id in seen_trade_pairs:
                debug_positions = [p for p in open_positions if p.trade_pair.trade_pair_id == open_position.trade_pair.trade_pair_id]
                raise ValueError(f"Miner [{hotkey}] has multiple open positions for trade pair [{open_position.trade_pair}]. Please restore cache. Affected positions: {debug_positions}")
            else:
                seen_trade_pairs.add(open_position.trade_pair.trade_pair_id)

            #bt.logging.success(f"current return with fees for [{open_position.position_uuid}] is [{open_position.return_at_close}]")
            return_with_open_positions *= open_position.return_at_close

        self._update_portfolio_debug_counters(return_with_open_positions)

        for position in closed_positions:
            seen_trade_pairs.add(position.trade_pair.trade_pair_id)

        dd_with_open_positions = self.calculate_drawdown(return_with_open_positions, max_cuml_return_so_far)
        self.portfolio_max_dd_all_positions = max(self.portfolio_max_dd_closed_positions, dd_with_open_positions)
        # Log the dd for this miner and the positions trade_pairs they are in as well as total number of positions
        bt.logging.trace(f"MDD checker -- current return for [{hotkey}]'s portfolio is [{return_with_open_positions}]. max_portfolio_drawdown: {self.portfolio_max_dd_all_positions}. Seen trade pairs: {seen_trade_pairs}. n positions open [{len(open_positions)} / {len(sorted_positions)}]")

        mdd_failure = self.is_drawdown_beyond_mdd(dd_with_open_positions)
        if mdd_failure:
            self.position_manager.close_open_positions_for_miner(hotkey)
            self.append_elimination_row(hotkey, dd_with_open_positions, mdd_failure)

        return bool(mdd_failure)









                

