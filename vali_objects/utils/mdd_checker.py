# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import traceback
import time
from typing import List, Tuple

from data_generator.twelvedata_service import TwelveDataService
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.recent_event_tracker import RecentEventTracker

from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource


class MDDChecker(CacheController):

    def __init__(self, config, metagraph, position_manager, eliminations_lock, running_unit_tests=False, live_price_fetcher=None):
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

    def reset_debug_counters(self, reset_global_counters=True):
        self.portfolio_max_dd_closed_positions = 0
        self.portfolio_max_dd_all_positions = 0
        if reset_global_counters:
            self.n_miners_skipped_already_eliminated = 0
            self.n_eliminations_this_round = 0
            self.n_miners_mdd_checked = 0
            self.max_portfolio_value_seen = 0
            self.min_portfolio_value_seen = float("inf")

    def get_price_correction_data(self, sorted_positions: List[Position]):
        required_trade_pairs_for_price_corrections = set()
        trade_pair_to_last_order_time_ms = {}
        for position in sorted_positions:
            # Fresh orders need to attempt price corrections
            if position.newest_order_age_ms < RecentEventTracker.OLDEST_ALLOWED_RECORD_MS:
                trade_pair_to_last_order_time_ms[position.trade_pair] = position.orders[-1].processed_ms
                required_trade_pairs_for_price_corrections.add(position.trade_pair)

        trade_pair_to_price_sources = (
            self.live_price_fetcher.get_latest_prices(trade_pairs=list(required_trade_pairs_for_price_corrections),
                                                      trade_pair_to_last_order_time_ms=trade_pair_to_last_order_time_ms))

        debug = {}
        for k, v in trade_pair_to_price_sources.items():
            if v is not None and v[1]:
                debug[k.trade_pair_id] = str([x.debug_str(trade_pair_to_last_order_time_ms[k]) for x in v[1]])
        if debug:
            bt.logging.info(f"mdd checker successfully retrieved price correction data: {debug}")
        return trade_pair_to_price_sources


    def get_candle_data(self, hotkey_positions) -> Tuple[dict, dict]:
        required_trade_pairs_for_candles = set()
        trade_pair_to_last_order_time_ms = {}
        for sorted_positions in hotkey_positions.values():
            for position in sorted_positions:
                # Only need live price for open positions
                if position.is_open_position:
                    required_trade_pairs_for_candles.add(position.trade_pair)

        now = TimeUtil.now_in_millis()
        # If we're going to use candles, we need at least 2 seconds to have elapsed.

        candle_data = self.live_price_fetcher.get_candles(trade_pairs=list(required_trade_pairs_for_candles),
                                                  start_time_ms=self.last_price_fetch_time_ms if self.last_price_fetch_time_ms else now,
                                                  end_time_ms=now)

        self.last_price_fetch_time_ms = now
        return candle_data

    
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
        candle_data = self.get_candle_data(hotkey_to_positions)

        any_eliminations = False
        for hotkey, sorted_positions in hotkey_to_positions.items():
            self.reset_debug_counters(reset_global_counters=False)
            if self._search_for_miner_dd_failures(hotkey, sorted_positions, candle_data):
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

    def _parse_price_from_candle_data(self, candle_data, trade_pair):
        if trade_pair not in candle_data:
            raise ValueError(f"Trade pair [{trade_pair}] not in candle data. Candle data keys: {candle_data.keys()}")

        dat = candle_data[trade_pair]
        if dat is None:
            # Market is closed for this trade pair
            return None
        if isinstance(dat, float) or isinstance(dat, int):
            return float(dat)

        # Get the newest price in the window
        price = None
        for a in dat:
            #bt.logging.info(f"in _parse_price_from_closing_prices. timestamp: {a.timestamp}, close: {a.close}")
            if a.close is not None:
                price = a.close

        #bt.logging.info(f"in _parse_price_from_closing_prices. price: {price}. trade pair {trade_pair.trade_pair_id}")
        return price

    def _parse_extreme_price_in_window(self, signal_closing_prices, open_position, parse_min=True):
        trade_pair = open_position.trade_pair
        dat = signal_closing_prices[trade_pair]
        if dat is None:
            # Market is closed for this trade pair
            return None
        if isinstance(dat, float) or isinstance(dat, int):
            # From TwelveData
            return float(dat)
        # Handle the case where an order gets placed in between MDD checks.
        min_allowed_timestamp_ms = open_position.orders[-1].processed_ms
        price = None
        for a in dat:
            candle_epoch_ms = a.end_ms
            if candle_epoch_ms < min_allowed_timestamp_ms:
                continue
            #bt.logging.info(f"in _parse_min_price_in_window. timestamp: {a.timestamp}, close: {a.close}")
            if parse_min:
                if a.low is not None:
                    price = a.low if price is None else min(price, a.low)
            else:
                if a.high is not None:
                    price = a.high if price is None else max(price, a.high)
        #print(f"in _parse_min_price_in_window min_price: {min_price}. trade_pair {trade_pair.trade_pair_id}")
        return float(price) if price else None


    def _update_position_returns_and_persist_to_disk(self, hotkey, position, trade_pair_to_price_sources, candle_data) -> Position:
        """
        Setting the latest returns and persisting to disk for accurate MDD calculation and logging in get_positions

        Won't account for a position that was added in the time between mdd_check being called and this function
        being called. But that's ok as we will process such new positions the next round.
        """
        trade_pair_id = position.trade_pair.trade_pair_id
        realtime_price = self._parse_price_from_candle_data(candle_data, position.trade_pair)
        if realtime_price is None:  # market closed. Don't update return
            # TODO: fall back to any high resolution WS data that may exist? Resolve spikes in Poly WS data here. Utilize price correction data in MDD?
            return position

        attempt_retro_price_update = True
        order_uuid_before_refresh = position.orders[-1].order_uuid
        with self.position_manager.position_locks.get_lock(hotkey, trade_pair_id):
            # Position could have updated in the time between mdd_check being called and this function being called
            position = self.position_manager.get_miner_position_from_disk_using_position_in_memory(position)
            if position.orders[-1].order_uuid != order_uuid_before_refresh:
                # Position has changed since we last refreshed it. Don't try to retro update price
                attempt_retro_price_update = False
            changed = False
            n_orders_updated = 0
            for order in position.orders:
                if not attempt_retro_price_update:
                    break
                order_age = order.get_order_age(order)
                if order_age <= RecentEventTracker.OLDEST_ALLOWED_RECORD_MS:
                    temp = trade_pair_to_price_sources.get(position.trade_pair, (None, None))
                    new_sources = temp[1]
                    orig_price = order.price
                    orig_return = position.return_at_close
                    if self.price_correction_enabled and new_sources:
                        updated = PriceSource.update_order_with_newest_price_sources(order, new_sources, hotkey,
                                                                                     position.trade_pair.trade_pair)
                        if updated:
                            changed = True
                            n_orders_updated += 1
                            bt.logging.warning(
                                f"Retroactively updated order price for {position.miner_hotkey} {position.trade_pair.trade_pair} from "
                                f"{orig_price} to {order.price} rac b/a {orig_return:.8f}/{position.return_at_close:.8f}")



            # Rebuild the position with the newest price
            if changed:
                position.rebuild_position_with_updated_orders()
                bt.logging.warning( f"Retroactively updated {n_orders_updated} order prices for {position.miner_hotkey} {position.trade_pair.trade_pair}  "
                                    f"return_at_close changed from {orig_return:.8f} to {position.return_at_close:.8f}")

            # Log return before calling set_returns
            #bt.logging.info(f"current return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{open_position.return_at_close}]. Position: {position}")
            if position.is_open_position:
                orig_return = position.return_at_close
                position.set_returns(realtime_price)
                changed |= orig_return != position.return_at_close
            if changed:
                self.position_manager.save_miner_position_to_disk(position, delete_open_position_if_exists=False)

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
        trade_pair_to_price_sources = self.get_price_correction_data(sorted_positions)
        for position in sorted_positions:
            # Perform needed updates
            if position.is_open_position or position.newest_order_age_ms <= RecentEventTracker.OLDEST_ALLOWED_RECORD_MS:
                position = self._update_position_returns_and_persist_to_disk(hotkey, position, trade_pair_to_price_sources, candle_data)

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
        for open_position in open_positions:
            #bt.logging.info(f"current return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{open_position.return_at_close}]")
            if open_position.trade_pair.trade_pair_id in seen_trade_pairs:
                debug_positions = [p for p in open_positions if p.trade_pair.trade_pair_id == open_position.trade_pair.trade_pair_id]
                raise ValueError(f"Miner [{hotkey}] has multiple open positions for trade pair [{open_position.trade_pair}]. Please restore cache. Affected positions: {debug_positions}")
            else:
                seen_trade_pairs.add(open_position.trade_pair.trade_pair_id)

            #bt.logging.success(f"current return with fees for [{open_position.position_uuid}] is [{open_position.return_at_close}]")
            parse_min = open_position.position_type == OrderType.LONG
            candle_price = self._parse_extreme_price_in_window(candle_data, open_position, parse_min=parse_min)
            if candle_price is None:  # Market closed for this trade pair. keep return the same
                unrealized_return_with_fees = open_position.return_at_close
            else:
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
            self.position_manager.close_open_positions_for_miner(hotkey)
            self.append_elimination_row(hotkey, dd_with_open_positions, mdd_failure)

        return bool(mdd_failure)









                

