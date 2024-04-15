# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import shutil
import time
import traceback
import uuid
from pickle import UnpicklingError
from typing import List, Dict, Union
import bittensor as bt
from pathlib import Path

from copy import deepcopy
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.position import Position
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import OrderStatus, Order
from vali_objects.utils.position_utils import PositionUtils


class PositionManager(CacheController):
    def __init__(self, config=None, metagraph=None, running_unit_tests=False, perform_price_adjustment=False, live_price_fetcher=None, perform_order_corrections=False, perform_fee_structure_update=False):
        super().__init__(config=config, metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.init_cache_files()
        self.position_locks = PositionLocks()
        self.live_price_fetcher = live_price_fetcher
        self.recalibrated_position_uuids = set()
        if perform_price_adjustment:
            self.perform_price_recalibration()
        if perform_order_corrections:
            try:
                self.apply_order_corrections()
            except Exception as e:
                bt.logging.error(f"Error applying order corrections: {e}")
        if perform_fee_structure_update:
            self.ensure_latest_fee_structure_applied()



    def give_erronously_eliminated_miners_another_shot(self):
        # The MDD Checker will immediately eliminate miners if they exceed the maximum drawdown
        eliminations = self.get_miner_eliminations_from_disk()
        new_eliminations = []
        for e in eliminations:
            if e['hotkey'] == '':
                bt.logging.warning('Removed elimination for hotkey ', e['hotkey'])
            else:
                new_eliminations.append(e)

        self.write_eliminations_to_disk(new_eliminations)

    def apply_order_corrections(self):
        """
        This is our mechanism for manually synchronizing validator orders in situations where a bug prevented an
        order from filling. We are working on a more robust automated synchronization/recovery system.

        11/4/2024 - Metagraph synchronization was set to 5 minutes preventing a new miner from having their orders
        processed by all validators. After verifying that this miner's order should have been sent to all validators,
        we increased the metagraph update frequency to 1 minute to prevent this from happening again. This override
        will correct the order status for this miner.

        4/13/2024 - Price recalibration incorrectly applied to orders made after TwelveData websocket prices were
        implemented. This regressed pricing since the websocket prices are more accurate.

        Errantly closed out open CADCHF positions during a recalibration. Delete these positions that adversely affected
        miners

        One miner was eliminated due to a faulty candle from polygon at the close. We are investigating a workaround
        and have several candidate solutions.

        miner couldn't close position due to temporary bug. deleted position completely.

        # 4/15/23 Verified high lag on order price using Twelve Data

        """

        def correct_for_tp(positions: List[Position], idx, prices, tp, timestamp_ms=None):
            nonlocal n_attempts, n_corrections, unique_corrections
            n_attempts += 1
            pos = None
            i = -1

            for p in positions:
                if p.trade_pair == tp:
                    pos = p
                    i += 1
                    if i == idx:
                        break

            if pos and timestamp_ms:
                # check if the timestamp_ms is outside of 5 minutes of the position's open_ms
                delta_time_min = abs(timestamp_ms - pos.open_ms) / 1000.0 / 60.0
                if delta_time_min > 5.0:
                    bt.logging.warning(
                        f"Timestamp ms: {timestamp_ms} is more than 5 minutes away from position open ms: {pos.open_ms}. delta_time_min {delta_time_min}")
                    return

            if not prices:
                # del position
                if pos:
                    self.delete_position_from_disk(pos)
                    unique_corrections.add(pos.position_uuid)
                    n_corrections += 1
                    return

            elif i == idx and pos and len(prices) == len(pos.orders):
                self.delete_position_from_disk(pos)
                new_orders = []
                for i, o in enumerate(pos.orders):
                    o.price = prices[i]
                    new_orders.append(o)
                pos.orders = new_orders
                old_return = pos.return_at_close
                pos.rebuild_position_with_updated_orders()
                self.save_miner_position_to_disk(pos)
                bt.logging.info(f"Corrected position for trade pair {tp.trade_pair_id}. Old return {old_return}, New return: {pos.return_at_close}")
                unique_corrections.add(pos.position_uuid)
                n_corrections += 1
                return


        self.give_erronously_eliminated_miners_another_shot()
        n_corrections = 0
        n_attempts = 0
        unique_corrections = set()
        hotkey_to_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)
        for miner_hotkey, positions in hotkey_to_positions.items():
            """
                    
            if miner_hotkey == '5DX8tSyGrx1QuoR1wL99TWDusvmmWgQW5su3ik2Sc8y8Mqu3':
                n_corrections += self.correct_for_tp(positions, 0, [151.83500671, 151.792], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                n_corrections += self.correct_for_tp(positions, 0, [0.66623, 0.66634], TradePair.CADCHF, unique_corrections)

            if miner_hotkey == '5D4zieKMoRVm477oUyMTZAWZ9orzpiJM8K6ufQQjryiXwpGU':
                n_corrections += self.correct_for_tp(positions, 0, [0.66634, 0.6665], TradePair.CADCHF, unique_corrections)

            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                n_corrections += self.correct_for_tp(positions, 0, None, TradePair.CADCHF, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 0, [151.841, 151.773], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 1, [151.8, 152.302], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5Ec93qtHkKprEaA5EWXrmPmWppMeMiwaY868bpxfkH5ocBxi':
                n_corrections += self.correct_for_tp(positions, 0, [151.808, 151.844], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 1, [151.817, 151.84], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 2, [151.839, 151.809], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 3, [151.772, 151.751], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 4, [151.77, 151.748], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5Ct1J2jNxb9zeHpsj547BR1nZk4ZD51Bb599tzEWnxyEr4WR':
                n_corrections += self.correct_for_tp(positions, 0, None, TradePair.CADCHF, unique_corrections)
                
            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                correct_for_tp(positions, 2, None, TradePair.EURCHF, timestamp_ms=1712950839925)
            if miner_hotkey == '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
                correct_for_tp(positions, 0, [0.66242, 0.66464], TradePair.CADCHF)
            if miner_hotkey == '5D4zieKMoRVm477oUyMTZAWZ9orzpiJM8K6ufQQjryiXwpGU':
                correct_for_tp(positions, 0, [111.947, 111.987], TradePair.CADJPY)
            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                correct_for_tp(positions, 0, [151.727, 151.858, 153.0370, 153.0560, 153.0720, 153.2400, 153.2280, 153.2400], TradePair.USDJPY)
            if miner_hotkey == '5DfhKZckZwjCqEcBUsW7jwzA5APCdj5SgZbfK6zzS9bMPuHn':
                correct_for_tp(positions, 0, [111.599, 111.55999756, 111.622], TradePair.CADJPY)
                
            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                correct_for_tp(positions, 0, [151.73, 151.862, 153.047, 153.051, 153.071, 153.241, 153.225, 153.235], TradePair.USDJPY)
            if miner_hotkey == '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx':
                correct_for_tp(positions, 0, None, TradePair.ETHUSD, timestamp_ms=1713102534971)
            """
            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                correct_for_tp(positions, 1, [100.192, 100.711, 100.379], TradePair.AUDJPY)


        bt.logging.warning(f"Applied {n_corrections} order corrections out of {n_attempts} attempts. unique positions corrected: {len(unique_corrections)}")


    def ensure_latest_fee_structure_applied(self):
        hotkey_to_positions = self.get_all_disk_positions_for_all_miners(only_open_positions=False, sort_positions=True)
        n_positions_seen = 0
        n_positions_updated = 0
        n_positions_updated_significantly = 0
        n_positions_stayed_the_same = 0
        significant_deltas = []
        for hotkey, positions in hotkey_to_positions.items():
            for position in positions:
                # skip open positions as their returns will be updated in the next MDD check.
                # Also once positions closes, it will make it past this check next validator boot
                if position.is_open_position:
                    continue
                n_positions_seen += 1
                # Ensure this position is using the latest fee structure. If not, recalculate and persist the new return to disk
                old_return_at_close = position.return_at_close
                new_return_at_close = position.calculate_return_with_fees(position.current_return)
                if old_return_at_close != new_return_at_close:
                    n_positions_updated += 1
                    if abs(old_return_at_close - new_return_at_close) / old_return_at_close > 0.03:
                        n_positions_updated_significantly += 1
                        bt.logging.info(f"Updating return_at_close for position {position.position_uuid} trade pair "
                                        f"{position.trade_pair.trade_pair_id} from {old_return_at_close} to {new_return_at_close}")
                        #significant_deltas.append((old_return_at_close, new_return_at_close, position.to_json_string()))
                    position.return_at_close = new_return_at_close
                    self.save_miner_position_to_disk(position, delete_open_position_if_exists=False)
                else:
                    n_positions_stayed_the_same += 1
        bt.logging.info(f"Updated {n_positions_updated} positions. {n_positions_updated_significantly} positions "
                        f"were updated significantly. {n_positions_stayed_the_same} stayed the same. {n_positions_seen} positions were seen in total. Significant deltas: {significant_deltas}")

    def close_open_position_for_miner(self, hotkey: str, trade_pair: TradePair):
        with self.position_locks.get_lock(hotkey, trade_pair.trade_pair_id):
            open_position = self.get_open_position_for_a_miner_trade_pair(hotkey, trade_pair.trade_pair_id)
            if open_position:
                bt.logging.info(f"Closing open position for hotkey: {hotkey} and trade_pair: {trade_pair.trade_pair_id}")
                open_position.close_out_position(TimeUtil.now_in_millis())
                self.save_miner_position_to_disk(open_position)

    def close_open_positions_for_miner(self, hotkey: str):
        for trade_pair in TradePair:
            self.close_open_position_for_miner(hotkey, trade_pair)

    def recalculate_return_at_close_and_write_corrected_position_to_disk(self, position: Position, hotkey:str):
        # TODO LOCK and how to handle open positions?
        tp = position.trade_pair
        if tp not in (TradePair.CADJPY, TradePair.USDJPY, TradePair.CHFJPY):
            return position.return_at_close

        any_changes = False
        disposable_clone = deepcopy(position)
        deltas = []
        new_orders = []
        for o in disposable_clone.orders:
            timestamp_ms = o.processed_ms
            old_price = o.price
            new_price = self.live_price_fetcher.get_close_at_date(tp, timestamp_ms)

            if new_price is not None and new_price != old_price:
                # IF the order recent, don't recalibrate
                if o.processed_ms > 1712901063000:
                    time_lo = TimeUtil.timestamp_ms_to_eastern_time_str(o.processed_ms)
                    bt.logging.warning(
                        f"Skipping recalibration for trade pair {tp.trade_pair_id}."
                        f" Last order is recent. {time_lo}. Price would have changed from {old_price} to {new_price}.")
                else:
                    any_changes = True
                    deltas.append((old_price, new_price))
                    o.price = new_price
            new_orders.append(o)


        for i in range(len(position.orders)):
            orders = new_orders[:i+1]
            disposable_clone.orders = orders
            disposable_clone.position_type = None
            disposable_clone._update_position()
            if disposable_clone.orders[0].leverage < 0:
                order_type = OrderType.SHORT
            else:
                order_type = OrderType.LONG
            if 0:#len(orders) > 1:
                prev_order = orders[-2]
                o = orders[-1]
                window_start_ms = prev_order.processed_ms
                window_end_ms = o.processed_ms
                candles = self.live_price_fetcher.get_candles([tp], window_start_ms,
                                                              window_end_ms) if window_start_ms else None
                fs = TimeUtil.millis_to_formatted_date_str(window_start_ms)
                fe = TimeUtil.millis_to_formatted_date_str(window_end_ms)
                if tp in candles and candles[tp] and isinstance(candles[tp], list):
                    min_price_seen = min([x.low for x in candles[tp]])
                    max_price_seen = max([x.high for x in candles[tp]])
                    unrealized_return = disposable_clone.calculate_unrealized_pnl(min_price_seen if order_type == OrderType.LONG else max_price_seen)
                    unrealized_return_with_fees = disposable_clone.calculate_return_with_fees(unrealized_return)
                    drawdown = self.calculate_drawdown(unrealized_return_with_fees, 1.0)
                    bt.logging.warning(f"Drawdown is {drawdown} for hotkey {hotkey} trade pair {tp.trade_pair_id}. Between time: {fs} and {fe}. p1: {prev_order.price} p2: {o.price}, min: {min_price_seen}.")
                    if drawdown < ValiConfig.MAX_TOTAL_DRAWDOWN:
                        bt.logging.error(f"Drawdown is {drawdown} for hotkey {hotkey} trade pair {tp.trade_pair_id}. Unrealized return: {unrealized_return_with_fees}.")


        assert len(disposable_clone.orders) == len(position.orders)
        if any_changes:
            with self.position_locks.get_lock(hotkey, tp.trade_pair_id):
                self.save_miner_position_to_disk(disposable_clone, delete_open_position_if_exists=False)
            bt.logging.info(f"Recalculated return_at_close for position {position.position_uuid}."
                            f" Trade pair {position.trade_pair.trade_pair_id} New value: "
                            f"{disposable_clone.return_at_close}. Original value: {position.return_at_close}")
            bt.logging.info(f"Corrected n={len(deltas)} prices for position {position.position_uuid} Trade pair {tp.trade_pair_id}. Deltas (before/after): {deltas}")

        return disposable_clone.return_at_close


    def get_all_disk_positions_for_all_miners(self, **args):
        all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(
            ValiBkpUtils.get_miner_dir()
        )
        return self.get_all_miner_positions_by_hotkey(all_miner_hotkeys, **args)

    def close_open_orders_for_suspended_trade_pairs(self):
        tps_to_eliminate = []
        all_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)
        eliminations = self.get_miner_eliminations_from_disk()
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
        bt.logging.info(f"Found {len(eliminations)} eliminations on disk.")
        for hotkey, positions in all_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            # Closing all open positions for the specified trade pair
            for position in positions:
                if position.is_closed_position:
                    continue
                if position.trade_pair in tps_to_eliminate:
                    bt.logging.info(
                        f"Position {position.position_uuid} for hotkey {hotkey} and trade pair {position.trade_pair.trade_pair_id} has been closed")
                    self.close_open_position_for_miner(hotkey, position.trade_pair)

    def perform_price_recalibration(self, time_per_batch_s=90):
        try:
            t0 = time.time()
            if not self.live_price_fetcher.polygon_available:
                bt.logging.error("Polygon API not detected. Skipping price recalibration. Your validator will fall out of consensus.")
                return
            self.perform_price_recalibration_arap(time_per_batch_s)
            bt.logging.info(f"Price recalibration complete for {len(self.recalibrated_position_uuids)} positions in {time.time() - t0} seconds.")
        except Exception as e:
            bt.logging.error(f"Error performing price recalibration: {e}")
            bt.logging.error(traceback.format_exc())

    def perform_price_recalibration_arap(self, time_per_batch_s):
        t0 = time.time()
        all_positions = self.get_all_disk_positions_for_all_miners(sort_positions=True, only_open_positions=False)
        eliminations = self.get_miner_eliminations_from_disk()
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
        hotkeys_eliminated_to_current_return = {}
        hotkeys_with_return_modified = set()
        skipped_eliminated = 0
        n_positions_total = 0
        n_positions_modified = 0
        RETURNS_MODIFIED = []
        for hotkey, positions in all_positions.items():
            max_portfolio_return = 1.0
            cur_portfolio_return = 1.0
            most_recent_elimination_idx = None
            n_positions_total += len(positions)

            if len(positions) == 0:
                continue

            if hotkey in eliminated_hotkeys:
                skipped_eliminated += len(positions)
                continue

            dd_to_log = None
            for i, position in enumerate(positions):
                if position.position_uuid in self.recalibrated_position_uuids:
                    continue
                if time.time() - t0 > time_per_batch_s:
                    return
                original_return = position.return_at_close
                new_return = self.recalculate_return_at_close_and_write_corrected_position_to_disk(position, hotkey)
                if original_return != new_return:
                    hotkeys_with_return_modified.add(hotkey)
                    n_positions_modified += 1
                    RETURNS_MODIFIED.append((original_return, new_return, position.trade_pair.trade_pair_id, hotkey))
                if position.is_open_position:
                    continue
                self.recalibrated_position_uuids.add(position.position_uuid)
                cur_portfolio_return *= new_return
                max_portfolio_return = max(max_portfolio_return, cur_portfolio_return)
                drawdown = self.calculate_drawdown(cur_portfolio_return, max_portfolio_return)
                mdd_failure = self.is_drawdown_beyond_mdd(drawdown,
                                                           time_now=TimeUtil.millis_to_datetime(position.close_ms))
                if mdd_failure:
                    dd_to_log = drawdown
                    most_recent_elimination_idx = i

            return_with_open_positions = cur_portfolio_return
            for p in positions:
                if p.is_open_position:
                    return_with_open_positions *= p.return_at_close

            drawdown = self.calculate_drawdown(return_with_open_positions, max_portfolio_return)
            mdd_failure = self.is_drawdown_beyond_mdd(drawdown)
            if mdd_failure:
                most_recent_elimination_idx = len(positions) - 1
                dd_to_log = drawdown

            if most_recent_elimination_idx is not None:
                return_as_shown_on_dash = 1.0
                for p in positions:
                    return_as_shown_on_dash *= p.return_at_close
                hotkeys_eliminated_to_current_return[hotkey] = return_as_shown_on_dash
                bt.logging.warning(
                    f"MDD failure occurred at position {most_recent_elimination_idx} out of {len(positions)} positions for hotkey "
                    f"{hotkey}. Drawdown: {dd_to_log}. MDD failure: {mdd_failure}. Portfolio return: {return_with_open_positions}. ")


        bt.logging.info(f"Found n= {len(hotkeys_eliminated_to_current_return)} hotkeys to eliminate. After modifying returns for n = {len(hotkeys_with_return_modified)} hotkeys.")
        bt.logging.info(f"Total initial positions: {n_positions_total}, Modified {n_positions_modified}, Skipped {skipped_eliminated} eliminated.")
        bt.logging.warning(f"Position returns modified: {RETURNS_MODIFIED}")
        for rm in RETURNS_MODIFIED:
            if abs(rm[0] - rm[1]) / rm[0] > -1:
                bt.logging.warning(f"    Original return: {rm[0]}, New return: {rm[1]}, Trade pair: {rm[2]} hotkey: {rm[3]}")

        for k, v in sorted(hotkeys_eliminated_to_current_return.items(), key=lambda x: x[1]):
            bt.logging.info(f"hotkey: {k}. return as shown on dash: {v}")
        return hotkeys_eliminated_to_current_return

    def sort_by_close_ms(self, _position):
        return (
            _position.close_ms if _position.is_closed_position else float("inf")
        )

    def get_return_per_closed_position(self, positions: List[Position]) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []
        for position in positions:
            if position.is_open_position:
                continue
            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms
            closed_position_returns.append(position.return_at_close)

        cumulative_return = 1
        per_position_return = []

        # calculate the return over time at each position close
        for value in closed_position_returns:
            cumulative_return *= value
            per_position_return.append(cumulative_return)
        return per_position_return
    
    def get_return_per_closed_position_augmented(
            self, 
            positions: List[Position], 
            evaluation_time_ms: Union[None, int] = None,
            closed_only: bool = False
        ) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []

        for position in positions:
            # if bool on for closed only
            if position.is_open_position and closed_only:
                continue

            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms

            # this value will probably be around 0, or between -1 and 1
            logged_return_at_close = PositionUtils.log_transform(position.return_at_close)

            # this should be even closer to 0 with an absolute value less than the logged_return_at_close
            if evaluation_time_ms is None:
                raise ValueError("evaluation_time_ms must be provided if augmented is True")

            dampened_return_at_close = PositionUtils.dampen_return(
                logged_return_at_close, 
                position.open_ms, 
                position.close_ms, 
                evaluation_time_ms
            )
            closed_position_returns.append(dampened_return_at_close)

        consistency_penalty = PositionUtils.compute_consistency_penalty(
            positions, 
            evaluation_time_ms
        )

        # cumulative_return_logged = 0
        # per_position_return_logged = []

        # # calculate the return over time at each position close
        # for value in closed_position_returns:
        #     cumulative_return_logged += value
        #     per_position_return_logged.append(value)

        return [ x * consistency_penalty for x in closed_position_returns ]

    def get_all_miner_positions(self,
                                miner_hotkey: str,
                                only_open_positions: bool = False,
                                sort_positions: bool = False,
                                acceptable_position_end_ms: int = None
                                ) -> List[Position]:

        miner_dir = ValiBkpUtils.get_miner_all_positions_dir(miner_hotkey, running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)

        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions):
            bt.logging.trace(f"miner_dir: {miner_dir}, n_positions: {len(positions)}")

        if acceptable_position_end_ms is not None:
            positions = [
                position
                for position in positions
                if position.open_ms > acceptable_position_end_ms
            ]

        if only_open_positions:
            positions = [
                position for position in positions if position.is_open_position
            ]

        if sort_positions:
            positions = sorted(positions, key=self.sort_by_close_ms)

        return positions

    def get_all_miner_positions_by_hotkey(self, hotkeys: List[str], eliminations: List = None, **args) -> Dict[
        str, List[Position]]:
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()
        return {
            hotkey: self.get_all_miner_positions(hotkey, **args)
            for hotkey in hotkeys
            if hotkey not in eliminated_hotkeys
        }

    @staticmethod
    def positions_are_the_same(position1: Position, position2: Position | dict) -> (bool, str):
        # Iterate through all the attributes of position1 and compare them to position2.
        # Get attributes programmatically.
        comparing_to_dict = isinstance(position2, dict)
        for attr in dir(position1):
            attr_is_property = isinstance(getattr(type(position1), attr, None), property)
            if attr.startswith("_") or callable(getattr(position1, attr)) or (comparing_to_dict and attr_is_property):
                continue

            value1 = getattr(position1, attr)
            # Check if position2 is a dict and access the value accordingly.
            if comparing_to_dict:
                # Use .get() to avoid KeyError if the attribute is missing in the dictionary.
                value2 = position2.get(attr)
            else:
                value2 = getattr(position2, attr, None)

            if value1 != value2:
                return False, f"{attr} is different. {value1} != {value2}"
        return True, ""

    def get_miner_position_from_disk_using_position_in_memory(self, memory_position: Position) -> Position:
        # Position could have changed to closed
        fp = self.get_filepath_for_position(hotkey=memory_position.miner_hotkey,
                                            trade_pair_id=memory_position.trade_pair.trade_pair_id,
                                            position_uuid=memory_position.position_uuid,
                                            is_open=memory_position.is_open_position)
        try:
            return self.get_miner_position_from_disk(fp)
        except Exception as e:
            bt.logging.info(f"Error getting position from disk: {e}")
            if memory_position.is_open_position:
                bt.logging.warning(f"Attempting to get closed position from disk for memory position {memory_position}")
                fp = self.get_filepath_for_position(hotkey=memory_position.miner_hotkey,
                                                    trade_pair_id=memory_position.trade_pair.trade_pair_id,
                                                    position_uuid=memory_position.position_uuid,
                                                    is_open=False
                                                )
                return self.get_miner_position_from_disk(fp)
            else:
                raise e

    def get_miner_position_from_disk(self, file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        try:
            file_string = ValiBkpUtils.get_file(file)
            ans = Position.parse_raw(file_string)
            #bt.logging.info(f"vali_utils get_miner_position: {ans}")
            return ans
        except FileNotFoundError:
            raise ValiFileMissingException("Vali position file is missing")
        except UnpicklingError:
            raise ValiBkpCorruptDataException("position data is not pickled")
        except UnicodeDecodeError as e:
            raise ValiBkpCorruptDataException(f" Error {e} You may be running an old version of the software. Confirm with the team if you should delete your cache.")

    def get_recently_updated_miner_hotkeys(self):
        """
        Identifies and returns a list of directories that have been updated in the last 3 days.
        """
        # Define the path to the directory containing the directories to check
        query_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        # Get the current time
        current_time = time.time()
        # List of directories updated in the last 24 hours
        updated_directory_names = []
        # Get the names of all directories in query_dir
        directory_names = self.get_directory_names(query_dir)
        # Loop through each directory name
        for item in directory_names:
            item_path = Path(query_dir) / item  # Construct the full path
            # Get the last modification time of the directory
            root_last_modified_time_s = self._get_file_mod_time_s(item_path)
            latest_modification_time_s = self._get_latest_file_modification_time_s(item_path, root_last_modified_time_s)
            # Check if the directory was updated in the last 3 days
            if current_time - latest_modification_time_s < 259200:  # 3 days in seconds
                updated_directory_names.append(item)

        return updated_directory_names

    def _get_latest_file_modification_time_s(self, dir_path, root_last_modified_time):
        """
        Recursively finds the max modification time of all files within a directory.
        """
        latest_mod_time_s = root_last_modified_time
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                mod_time = self._get_file_mod_time_s(file_path)
                latest_mod_time_s = max(latest_mod_time_s, mod_time)

        return latest_mod_time_s

    def _get_file_mod_time_s(self, file_path):
        try:
            return os.path.getmtime(file_path)
        except OSError:  # Handle the case where the file is inaccessible
            return 0

    def delete_open_position_if_exists(self, position: Position) -> None:
        # See if we need to delete the open position file
        open_position = self.get_open_position_for_a_miner_trade_pair(position.miner_hotkey,
                                                                      position.trade_pair.trade_pair_id)
        if open_position:
            self.delete_position_from_disk(open_position)

    def verify_open_position_write(self, miner_dir, updated_position):
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
        # Print all files found for dir
        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions) == 0:
            return  # First time open position is being saved
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {updated_position.miner_hotkey} and trade_pair."
                                                   f" {updated_position.trade_pair.trade_pair_id}. Please restore cache. Positions: {positions}")
        elif len(positions) == 1:
            if positions[0].position_uuid != updated_position.position_uuid:
                raise ValiRecordsMisalignmentException(f"Open position for miner {updated_position.miner_hotkey} and trade_pair."
                                                       f" {updated_position.trade_pair.trade_pair_id} does not match the updated position."
                                                       f" Please restore cache. Position: {positions[0]} Updated position: {updated_position}")

    def save_miner_position_to_disk(self, position: Position, delete_open_position_if_exists=True) -> None:
        miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                     position.trade_pair.trade_pair_id,
                                                                     order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                                                                     running_unit_tests=self.running_unit_tests)
        if position.is_closed_position and delete_open_position_if_exists:
            self.delete_open_position_if_exists(position)
        elif position.is_open_position:
            self.verify_open_position_write(miner_dir, position)

        ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)

    def clear_all_miner_positions_from_disk(self):
        # Clear all files and directories in the directory specified by dir
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def get_number_of_eliminations(self):
        return len(self.get_miner_eliminations_from_disk())

    def get_number_of_plagiarism_scores(self):
        return len(self.get_plagiarism_scores_from_disk())

    def get_number_of_miners_with_any_positions(self):
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        ret = 0
        try:
            for file in os.listdir(dir):
                file_path = os.path.join(dir, file)
                ret += os.path.isdir(file_path)
            bt.logging.info(f"Number of miners with any positions: {ret}. Positions dir: {dir}")
        except FileNotFoundError:
            bt.logging.info(f"Directory for miners doesn't exist [{dir}].")
        return ret

    def get_extreme_position_order_processed_on_disk_ms(self):
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        min_time = float("inf")
        max_time = 0
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                continue
            hotkey = file
            # Read all positions in this directory
            positions = self.get_all_miner_positions(hotkey)
            for p in positions:
                for o in p.orders:
                    min_time = min(min_time, o.processed_ms)
                    max_time = max(max_time, o.processed_ms)
        return min_time, max_time



    def get_open_position_for_a_miner_trade_pair(self, hotkey: str, trade_pair_id: str) -> Position | None:
        dir = ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=OrderStatus.OPEN,
                                                               running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(dir)
        # Print all files found for dir
        positions = [self.get_miner_position_from_disk(file) for file in all_files]
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {hotkey} and trade_pair."
                                                   f" {trade_pair_id}. Please restore cache. Positions: {positions}")
        return positions[0] if len(positions) == 1 else None

    def get_filepath_for_position(self, hotkey, trade_pair_id, position_uuid, is_open):
        order_status = OrderStatus.CLOSED if not is_open else OrderStatus.OPEN
        return ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=order_status,
                                                               running_unit_tests=self.running_unit_tests) + position_uuid

    def delete_position_from_disk(self, p:Position):
        hotkey = p.miner_hotkey
        trade_pair_id = p.trade_pair.trade_pair_id
        position_uuid = p.position_uuid
        is_open = p.is_open_position
        filepath = self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, is_open)
        os.remove(filepath)
        bt.logging.info(f"Deleted position from disk: {filepath}")

