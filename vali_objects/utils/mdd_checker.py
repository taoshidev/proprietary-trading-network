# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import time
from typing import List, Dict

from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker

from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource


class MDDChecker(CacheController):

    def __init__(self, config, metagraph, position_manager, eliminations_lock, running_unit_tests=False,
                 live_price_fetcher=None, shutdown_dict=None):
        super().__init__(config, metagraph, running_unit_tests=running_unit_tests)
        self.last_price_fetch_time_ms = None
        self.n_miners_skipped_already_eliminated = 0
        self.n_eliminations_this_round = 0
        self.n_miners_mdd_checked = 0
        self.price_correction_enabled = True
        self.portfolio_max_dd_closed_positions = 0
        self.portfolio_max_dd_all_positions = 0
        secrets = ValiUtils.get_secrets()
        self.position_manager = position_manager
        assert self.running_unit_tests == self.position_manager.running_unit_tests
        self.all_trade_pairs = [trade_pair for trade_pair in TradePair]
        if live_price_fetcher is None:
            self.live_price_fetcher = LivePriceFetcher(secrets=secrets)
        else:
            self.live_price_fetcher = live_price_fetcher
        self.eliminations_lock = eliminations_lock
        self.reset_debug_counters()
        self.shutdown_dict = shutdown_dict
        self.n_poly_api_requests = 0

    def reset_debug_counters(self, reset_global_counters=True):
        self.portfolio_max_dd_closed_positions = 0
        self.portfolio_max_dd_all_positions = 0
        if reset_global_counters:
            self.n_miners_skipped_already_eliminated = 0
            self.n_eliminations_this_round = 0
            self.n_miners_mdd_checked = 0
            self.max_portfolio_value_seen = 0
            self.min_portfolio_value_seen = float("inf")


    def get_candle_data(self, hotkey_positions) -> Dict[TradePair, List[PriceSource]]:
        required_trade_pairs_for_candles = set()
        trade_pair_to_last_order_time_ms = {}
        for sorted_positions in hotkey_positions.values():
            for position in sorted_positions:
                # Only need live price for open positions
                if position.is_open_position:
                    required_trade_pairs_for_candles.add(position.trade_pair)

        now = TimeUtil.now_in_millis()
        candle_data = self.live_price_fetcher.get_candles(trade_pairs=list(required_trade_pairs_for_candles),
                                                  start_time_ms=self.last_price_fetch_time_ms if self.last_price_fetch_time_ms else now,
                                                  end_time_ms=now)
        self.n_poly_api_requests += len(required_trade_pairs_for_candles)

        self.last_price_fetch_time_ms = now
        return candle_data

    
    def mdd_check(self):
        self.n_poly_api_requests = 0
        if not self.refresh_allowed(ValiConfig.MDD_CHECK_REFRESH_TIME_MS):
            time.sleep(1)
            return

        if self.shutdown_dict:
            return
        bt.logging.info("running mdd checker")
        self.reset_debug_counters()
        self._refresh_eliminations_in_memory()

        hotkey_to_positions = self.position_manager.get_all_miner_positions_by_hotkey(
            self.metagraph.hotkeys, sort_positions=True,
            eliminations=self.eliminations
        )
        candle_data = self.get_candle_data(hotkey_to_positions)
        any_eliminations = False
        for hotkey, sorted_positions in hotkey_to_positions.items():
            if self.shutdown_dict:
                return
            self.reset_debug_counters(reset_global_counters=False)
            if self._search_for_miner_dd_failures(hotkey, sorted_positions, candle_data):
                any_eliminations = True
                self.n_eliminations_this_round += 1

        if any_eliminations:
            with self.eliminations_lock:
                self._write_eliminations_from_memory_to_disk()

        bt.logging.info(f"mdd checker completed for {self.n_miners_mdd_checked} miners. n_eliminations_this_round: "
                        f"{self.n_eliminations_this_round}. Max portfolio value seen: {self.max_portfolio_value_seen}. "
                        f"Min portfolio value seen: {self.min_portfolio_value_seen}. n_poly_api_requests: {self.n_poly_api_requests}")
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
                self.position_manager.handle_eliminated_miner(hotkey, {})
                self.append_elimination_row(hotkey, drawdown, mdd_failure)
                return True, position_return, max_cuml_return_so_far

        # Replay of closed positions complete.
        return False, cuml_return, max_cuml_return_so_far

    def _update_position_returns_and_persist_to_disk(self, hotkey, position, candle_data_dict) -> Position:
        """
        Setting the latest returns and persisting to disk for accurate MDD calculation and logging in get_positions

        Won't account for a position that was added in the time between mdd_check being called and this function
        being called. But that's ok as we will process such new positions the next round.
        """

        def _get_sources_for_order(order, trade_pair, is_last_order):
            # Only fall back to REST if the order is the latest. Don't want to get slowed down
            # By a flurry of recent orders.
            ws_only = not is_last_order
            self.n_poly_api_requests += 0 if ws_only else 1
            sources = self.live_price_fetcher.fetch_prices([trade_pair],
                                                        {trade_pair: order.processed_ms},
                                                        ws_only=ws_only).get(trade_pair, (None, None))[1]
            return sources

        trade_pair = position.trade_pair
        trade_pair_id = trade_pair.trade_pair_id
        orig_return = position.return_at_close
        orig_avg_price = position.average_entry_price
        orig_iep = position.initial_entry_price
        now_ms = TimeUtil.now_in_millis()
        with self.position_manager.position_locks.get_lock(hotkey, trade_pair_id):
            # Position could have updated in the time between mdd_check being called and this function being called
            position = self.position_manager.get_miner_position_from_disk_using_position_in_memory(position)
            n_orders_updated = 0
            for i, order in enumerate(reversed(position.orders)):
                if not self.price_correction_enabled:
                    break

                order_age = now_ms - order.processed_ms
                if order_age > RecentEventTracker.OLDEST_ALLOWED_RECORD_MS:
                    break  # No need to check older records

                sources = _get_sources_for_order(order, position.trade_pair, is_last_order=i == 0)
                if not sources:
                    bt.logging.error(f"Unexpectedly could not find any new price sources for order"
                                     f" {order.order_uuid} in {hotkey} {position.trade_pair.trade_pair}. If this"
                                     f"issue persist, alert the team.")
                    continue
                if PriceSource.update_order_with_newest_price_sources(order, sources, hotkey, position.trade_pair.trade_pair):
                    n_orders_updated += 1

            # Rebuild the position with the newest price
            if n_orders_updated:
                position.rebuild_position_with_updated_orders()
                bt.logging.warning(f"Retroactively updated {n_orders_updated} order prices for {position.miner_hotkey} {position.trade_pair.trade_pair}  "
                                    f"return_at_close changed from {orig_return:.8f} to {position.return_at_close:.8f} "
                                    f"avg_price changed from {orig_avg_price:.8f} to {position.average_entry_price:.8f} "
                                   f"initial_entry_price changed from {orig_iep:.8f} to {position.initial_entry_price:.8f}")

            # Log return before calling set_returns
            #bt.logging.info(f"current return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{open_position.return_at_close}]. Position: {position}")
            realtime_price = self.live_price_fetcher.parse_price_from_candle_data(candle_data_dict.get(trade_pair), trade_pair)
            if position.is_open_position and realtime_price is not None:
                orig_return = position.return_at_close
                position.set_returns(realtime_price)
                n_orders_updated += orig_return != position.return_at_close
            if n_orders_updated:
                is_liquidated = position.current_return == 0
                self.position_manager.save_miner_position_to_disk(position, delete_open_position_if_exists=is_liquidated)

            #bt.logging.info(f"updated return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{position.return_at_close}]. position: {position}")
            return position

    def _update_portfolio_debug_counters(self, portfolio_ret):
        self.max_portfolio_value_seen = max(self.max_portfolio_value_seen, portfolio_ret)
        self.min_portfolio_value_seen = min(self.min_portfolio_value_seen, portfolio_ret)

    def _search_for_miner_dd_failures(self, hotkey, sorted_positions, candle_data) -> bool:
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
            if self.shutdown_dict:
                return False
            # Perform needed updates
            if position.is_open_position or position.newest_order_age_ms <= RecentEventTracker.OLDEST_ALLOWED_RECORD_MS:
                position = self._update_position_returns_and_persist_to_disk(hotkey, position, candle_data)

            if position.is_closed_position:
                closed_positions.append(position)
            else:
                open_positions.append(position)

        elimination_occurred, return_with_closed_positions, max_cuml_return_so_far = self._replay_all_closed_positions(hotkey, closed_positions)
        if elimination_occurred:
            return True

        # Enforce only one open position per trade pair
        seen_trade_pairs = set()
        return_with_open_positions = return_with_closed_positions
        trade_pair_to_price_source_used_for_elimination_check = {}
        open_position_trade_pairs = []
        for open_position in open_positions:
            if self.shutdown_dict:
                return False
            #bt.logging.info(f"current return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{open_position.return_at_close}]")
            if open_position.trade_pair.trade_pair_id in seen_trade_pairs:
                debug_positions = [p for p in open_positions if p.trade_pair.trade_pair_id == open_position.trade_pair.trade_pair_id]
                raise ValueError(f"Miner [{hotkey}] has multiple open positions for trade pair [{open_position.trade_pair}]. Please restore cache. Affected positions: {debug_positions}")
            else:
                seen_trade_pairs.add(open_position.trade_pair.trade_pair_id)
                open_position_trade_pairs.append(open_position.trade_pair)

            #bt.logging.success(f"current return with fees for [{open_position.position_uuid}] is [{open_position.return_at_close}]")
            parse_min = open_position.position_type == OrderType.LONG
            candle_price, corresponding_source = self.live_price_fetcher.parse_extreme_price_in_window(candle_data, open_position, parse_min=parse_min)
            if candle_price is None:  # Market closed for this trade pair. keep return the same
                unrealized_return_with_fees = open_position.return_at_close
            else:
                trade_pair_to_price_source_used_for_elimination_check[open_position.trade_pair] = corresponding_source
                unrealized_return = open_position.calculate_unrealized_pnl(candle_price)
                unrealized_return_with_fees = open_position.calculate_return_with_fees(unrealized_return)
            return_with_open_positions *= unrealized_return_with_fees

        self._update_portfolio_debug_counters(return_with_open_positions)

        for position in closed_positions:
            seen_trade_pairs.add(position.trade_pair.trade_pair_id)

        dd_with_open_positions = self.calculate_drawdown(return_with_open_positions, max_cuml_return_so_far)
        self.portfolio_max_dd_all_positions = max(self.portfolio_max_dd_closed_positions, dd_with_open_positions)
        # Log the dd for this miner and the positions trade_pairs they are in as well as total number of positions
        bt.logging.trace(f"MDD checker -- current return for [{hotkey}]'s portfolio is [{return_with_open_positions}]. max_portfolio_drawdown: {self.portfolio_max_dd_all_positions}. Seen trade pairs: {seen_trade_pairs}. n positions open [{len(open_positions)} / {len(sorted_positions)}]")

        mdd_failure = self.is_drawdown_beyond_mdd(dd_with_open_positions)
        if mdd_failure:
            self.position_manager.handle_eliminated_miner(hotkey, trade_pair_to_price_source_used_for_elimination_check,
                                                          open_position_trade_pairs=open_position_trade_pairs)
            self.append_elimination_row(hotkey, dd_with_open_positions, mdd_failure)

        return bool(mdd_failure)









                

