from collections import defaultdict
import os
import statistics
import time
import json
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.order import ORDER_SRC_LIMIT_CANCELLED, ORDER_SRC_LIMIT_FILLED, ORDER_SRC_LIMIT_UNFILLED, Order
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder

import bittensor as bt

class LimitOrderManager(CacheController):
    def __init__(self, position_manager, live_price_fetcher, shutdown_dict=None, running_unit_tests=False, ipc_manager=None):
        super().__init__(running_unit_tests=running_unit_tests)
        self.position_manager: PositionManager = position_manager
        self.elimination_manager: EliminationManager = position_manager.elimination_manager
        self.live_price_fetcher: LivePriceFetcher = live_price_fetcher

        self.shutdown_dict = shutdown_dict

        if ipc_manager:
            self.limit_orders = ipc_manager.dict()
        else:
            self.limit_orders = {}

        self._read_limit_orders_from_disk()
        self.triggered_order_times = {}
        self.last_fill_time = defaultdict(lambda: defaultdict(int))
        self._reset_counters()

    def save_limit_order(self, miner_hotkey, limit_order, position_locks):
        if miner_hotkey not in self.limit_orders:
            self.limit_orders[miner_hotkey] = []
        orders = list(self.limit_orders[miner_hotkey])

        unfilled_limit_orders = [order for order in orders if order.src == ORDER_SRC_LIMIT_UNFILLED]

        if len(unfilled_limit_orders) >= ValiConfig.MAX_UNFILLED_LIMIT_ORDERS:
            raise SignalException(
                f"miner has too many unfilled limit orders "
                f"[{len(unfilled_limit_orders)}] > [{ValiConfig.MAX_UNFILLED_LIMIT_ORDERS}]"
            )

        trade_pair = limit_order.trade_pair
        order_uuid = limit_order.order_uuid

        with (position_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id)):
            position = self._get_position_for(miner_hotkey, limit_order)
            if not position and limit_order.order_type == OrderType.FLAT:
                raise SignalException(f"No position found for FLAT order")

            self._write_to_disk(miner_hotkey, limit_order)

            orders.append(limit_order)
            self.limit_orders[miner_hotkey] = orders

        bt.logging.info(f"Saved [{miner_hotkey}] limit order [{order_uuid}]")
        bt.logging.info(f"checking for trigger price")
        # If the limit price is triggered when the order is placed, fill it immediately with market price.
        # Otherwise, it's filled with limit price.
        price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, limit_order.processed_ms)
        if not price_sources:
            return

        trigger_price = self._evaluate_trigger_price(limit_order.order_type, position, price_sources[0], limit_order.limit_price)
        if trigger_price:
            self._fill_limit_order(miner_hotkey, limit_order, position, price_sources[0], trigger_price, position_locks)

    def check_limit_orders(self, position_locks):
        if not self.refresh_allowed(ValiConfig.LIMIT_ORDER_CHECK_REFRESH_MS):
            time.sleep(1)
            return

        self._reset_counters()

        now_ms = TimeUtil.now_in_millis()

        for order_uuid, order_timestamp in list(self.triggered_order_times.items()):
            if order_timestamp < now_ms - ValiConfig.LIMIT_ORDER_PRICE_BUFFER_MS:
                del self.triggered_order_times[order_uuid]

        all_price_sources = self._get_price_sources()
        for miner_hotkey, orders in self.limit_orders.items():
            for order in orders:
                trade_pair = order.trade_pair

                time_since_last_fill = now_ms - self.last_fill_time[miner_hotkey][trade_pair]
                if time_since_last_fill < ValiConfig.LIMIT_ORDER_FILL_INTERVAL_MS:
                    continue

                price_sources = all_price_sources.get(trade_pair)
                position = self._get_position_for(miner_hotkey, order)

                order_price_source = self._evaluate_fill_price_source(order, position, price_sources)
                if order_price_source:
                    self._fill_limit_order(miner_hotkey, order, position, order_price_source, order.limit_price, position_locks)

        bt.logging.info(f"Limit orders evaluated: {self._limit_orders_evaluated}, filled: {self._limit_orders_filled}")

        self.set_last_update_time(skip_message=False)

    def _get_price_sources(self):
        end_ms = TimeUtil.now_in_millis()
        start_ms = end_ms - ValiConfig.LIMIT_ORDER_PRICE_BUFFER_MS - ValiConfig.LIMIT_ORDER_CHECK_REFRESH_MS

        price_sources = {}
        for _, orders in self.limit_orders.items():
            for order in orders:
                tp = order.trade_pair

                if tp not in price_sources:
                    price_sources[tp] = self.live_price_fetcher.get_ws_price_sources_in_window(tp, start_ms, end_ms)

        return price_sources

    def _get_position_for(self, hotkey, order):
        trade_pair_id = order.trade_pair.trade_pair_id
        return self.position_manager.get_open_position_for_a_miner_trade_pair(hotkey, trade_pair_id)

    def _evaluate_fill_price_source(self, order, position, price_sources):
        if order.src != ORDER_SRC_LIMIT_UNFILLED:
            return None

        if not price_sources or len(price_sources) == 0:
            return None

        self._limit_orders_evaluated += 1
        limit_price = order.limit_price
        order_type = order.order_type

        for i, price_source in enumerate(price_sources):
            trigger_price = self._evaluate_trigger_price(order_type, position, price_source, limit_price)
            if trigger_price is None:
                continue

            self.triggered_order_times[order.order_uuid] = price_source.start_ms

            window_end_ms = price_source.start_ms + ValiConfig.LIMIT_ORDER_PRICE_BUFFER_MS
            future_prices = set([ps.open for ps in price_sources[i+1:] if ps.start_ms <= window_end_ms])
            if len(future_prices) < ValiConfig.MIN_UNIQUE_PRICES_FOR_LIMIT_FILL:
                continue

            median_future_price = statistics.median(list(future_prices))

            tolerance = trigger_price * ValiConfig.LIMIT_ORDER_PRICE_BUFFER_TOLERANCE
            if abs(trigger_price - median_future_price) <= tolerance:
                return price_source

            elif ((order_type == OrderType.LONG and median_future_price < trigger_price) or
                  (order_type == OrderType.SHORT and median_future_price > trigger_price)):
                return price_source

            position_type = position.position_type if position else None
            if position_type == OrderType.LONG:
                return price_source if median_future_price > trigger_price else None
            elif position_type == OrderType.SHORT:
                return price_source if median_future_price < trigger_price else None

        return None

    def _evaluate_trigger_price(self, order_type, position, ps, limit_price):
        bid_price = ps.bid if ps.bid > 0 else ps.open
        ask_price = ps.ask if ps.ask > 0 else ps.open

        position_type = position.position_type if position else None

        buy_type = order_type == OrderType.LONG or (order_type == OrderType.FLAT and position_type == OrderType.SHORT)
        sell_type = order_type == OrderType.SHORT or (order_type == OrderType.FLAT and position_type == OrderType.LONG)

        if buy_type:
            return ask_price if ask_price <= limit_price else None
        elif sell_type:
            return bid_price if bid_price >= limit_price else None
        else:
            return None

    def _fill_limit_order(self, miner_hotkey, order, position, price_source, fill_price, position_locks):
        trade_pair = order.trade_pair
        with (position_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id)):
            fill_time = price_source.start_ms
            order.src = ORDER_SRC_LIMIT_FILLED
            order.price_sources = [price_source]
            order.price = fill_price
            order.bid = price_source.bid
            order.ask = price_source.ask
            order.slippage = PriceSlippageModel.calculate_slippage(order.bid, order.ask, order)
            order.processed_ms = fill_time

            try:
                if not position:
                    position = Position(
                        miner_hotkey=miner_hotkey,
                        position_uuid=order.order_uuid,
                        open_ms=fill_time,
                        trade_pair=trade_pair
                    )
                self.position_manager.enforce_num_open_order_limit(miner_hotkey, order)
                net_portfolio_leverage = self.position_manager.calculate_net_portfolio_leverage(miner_hotkey)
                position.add_order(order, net_portfolio_leverage)
                self.position_manager.save_miner_position(position)

                if order.order_uuid in self.triggered_order_times:
                    del self.triggered_order_times[order.order_uuid]

                self._close_order(miner_hotkey, order, ORDER_SRC_LIMIT_FILLED, fill_time)

                self._limit_orders_filled += 1
                self.last_fill_time[miner_hotkey][trade_pair] = fill_time

                bt.logging.info(f"Filling limit order {order.order_uuid} with price {order.price}")

            except ValueError as e:
                bt.logging.info(f"Could not add limit order [{order.order_uuid}] to position {e}. Cancelling Order")
                self._close_order(miner_hotkey, order, ORDER_SRC_LIMIT_CANCELLED, fill_time)

    def _close_order(self, miner_hotkey, order, src, time_ms):
        order_uuid = order.order_uuid
        trade_pair_id = order.trade_pair.trade_pair_id

        if order.order_uuid in self.triggered_order_times:
            raise SignalException(f"Cannot close limit order [{order.order_uuid}] about to be filled")

        unfilled_dir = ValiBkpUtils.get_limit_orders_dir(miner_hotkey, trade_pair_id, "unfilled", self.running_unit_tests)
        closed_filename = unfilled_dir + order_uuid

        if not os.path.exists(closed_filename):
            bt.logging.warning(f"Closed unfilled limit order not found on disk [{order_uuid}]")

        order.src = src
        order.processed_ms = time_ms

        self._write_to_disk(miner_hotkey, order)
        os.remove(closed_filename)

        # update for ipc manager?
        self.limit_orders[miner_hotkey] = list(self.limit_orders[miner_hotkey])

        bt.logging.info(f"Successfully closed limit order [{order_uuid}] [{trade_pair_id}] for [{miner_hotkey}]")

    def cancel_limit_order(self, miner_hotkey, trade_pair, cancel_order_uuid, now_ms, position_locks):
        with (position_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id)):
            orders_to_cancel = []
            if not cancel_order_uuid:
                orders_to_cancel = [
                    order for order in self.limit_orders.get(miner_hotkey, [])
                    if order.src == ORDER_SRC_LIMIT_UNFILLED and order.trade_pair == trade_pair
                ]
            else:
                orders_to_cancel = [
                    order for order in self.limit_orders.get(miner_hotkey, [])
                    if order.order_uuid == cancel_order_uuid
                ]

            if not orders_to_cancel:
                raise SignalException(f"No unfilled limit orders found to cancel for {miner_hotkey} with {trade_pair.trade_pair_id}")

            for order in orders_to_cancel:
                self._close_order(miner_hotkey, order, ORDER_SRC_LIMIT_CANCELLED, now_ms)

    def _read_limit_orders_from_disk(self, hotkeys=None):
        if not hotkeys:
            hotkeys = ValiBkpUtils.get_directories_in_dir(
                ValiBkpUtils.get_miner_dir(self.running_unit_tests)
            )
        eliminated_hotkeys = self.elimination_manager.get_eliminations_from_memory()

        for hotkey in hotkeys:
            miner_orders = []

            if hotkey in eliminated_hotkeys:
                continue

            miner_order_dicts = ValiBkpUtils.get_limit_orders(hotkey, self.running_unit_tests)
            for order_dict in miner_order_dicts:
                try:
                    order = Order.from_dict(order_dict)
                    miner_orders.append(order)
                except Exception as e:
                    bt.logging.error(f"Error reading limit order from disk: {e}")
                    continue

            if miner_orders:
                sorted_orders = sorted(miner_orders, key=lambda o: o.processed_ms)
                self.limit_orders[hotkey] = sorted_orders

    def _reset_counters(self):
        self._limit_orders_evaluated = 0
        self._limit_orders_filled = 0

    def _write_to_disk(self, miner_hotkey, order):
        if not order:
            return
        try:
            trade_pair_id = order.trade_pair.trade_pair_id
            if order.src == ORDER_SRC_LIMIT_UNFILLED:
                status = "unfilled"
            else:
                status = "closed"

            order_dir = ValiBkpUtils.get_limit_orders_dir(miner_hotkey, trade_pair_id, status, self.running_unit_tests)
            os.makedirs(order_dir, exist_ok=True)

            filepath = order_dir + order.order_uuid
            ValiBkpUtils.write_file(filepath, order)
        except Exception as e:
            bt.logging.error(f"Error writing limit order to disk for miner hotkey {e}")

    def sync_limit_orders(self, sync_data):
        if not sync_data:
            return

        for miner_hotkey, orders_data in sync_data.items():
            if not orders_data:
                continue

            try:
                for data in orders_data:
                    order = Order.from_dict(data)
                    self._write_to_disk(miner_hotkey, order)
            except Exception as e:
                print(f"Could not sync limit orders {e}")

        self._read_limit_orders_from_disk()

    def to_dashboard_dict(self, miner_hotkey):
        orders = self.limit_orders.get(miner_hotkey)
        if not orders:
            return None

        order_list = []
        for order in orders:
            data = {
                "trade_pair": [order.trade_pair.trade_pair_id, order.trade_pair.trade_pair],
                "order_type": str(order.order_type),
                "processed_ms": order.processed_ms,
                "limit_price": order.limit_price,
                "price": order.price,
                "leverage": order.leverage,
                "src": order.src
            }
            order_list.append(data)

        return order_list
