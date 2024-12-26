# developer: trdougherty

import numpy as np
import copy
import math

from vali_objects.position import Position
from vali_objects.vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderType

from time_util.time_util import TimeUtil


class PositionUtils:
    @staticmethod
    def cumulative_leverage_position(
            positions: list[Position],
            evaluation_time_ms: int = None
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
            evaluation_time_ms: int - the evaluation time in milliseconds
        """
        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        return PositionUtils.translate_current_leverage(
            positions,
            evaluation_time_ms
        )

    @staticmethod
    def translate_current_leverage(
            positions: list[Position],
            evaluation_time_ms: int = None
    ) -> list[Position]:
        """
        Adjusts the leverage of each position based on order types and adds a new order with the final leverage at the end.
        """
        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        positions_copy = copy.deepcopy(positions)
        for position in positions_copy:
            running_leverage = 0
            new_orders = []
            for order in position.orders:
                running_leverage += order.leverage

                if order.order_type == OrderType.FLAT:
                    running_leverage = 0  # Reset leverage if order type is FLAT

                order.leverage = running_leverage

            # Create and append a new order with the final running leverage
            new_order = copy.deepcopy(position.orders[-1])
            new_order.processed_ms = evaluation_time_ms
            new_order.leverage = running_leverage
            if new_order.order_type != OrderType.FLAT:
                new_orders.append(new_order)

            position.orders.extend(new_orders)  # Append all new orders after the loop

        return positions_copy

    @staticmethod
    def average_leverage(positions: list[Position]) -> float:
        """
        Computes the time-weighted average leverage of a list of positions.

        Args:
            positions: list[Position] - the list of positions

        Returns:
            float - the time-weighted average leverage
        """
        if not positions:
            return 0.0

        total_time = 0
        total_timeleverage = 0

        for position in positions:
            if len(position.orders) < 2:
                continue

            last_time = position.orders[0].processed_ms
            running_leverage = position.orders[0].leverage

            for i in range(1, len(position.orders)):
                current_time = position.orders[i].processed_ms
                time_delta = current_time - last_time
                total_time += time_delta
                total_timeleverage += time_delta * abs(running_leverage)
                last_time = current_time
                running_leverage += position.orders[i].leverage

        if total_time == 0:
            return 0.0

        return total_timeleverage / total_time

    @staticmethod
    def total_duration(
            positions: list[Position]
    ) -> int:
        """
        Args:
            positions: list[Position] - the list of positions
        """
        time_deltas = []

        for position in positions:
            if position.is_closed_position:
                time_deltas.append(position.close_ms - position.open_ms)

        return sum(time_deltas)

    @staticmethod
    def flatten(
            positions: dict[str, list[Position]]
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
        """
        positions_list = []
        for miner_key, miner_positions in positions.items():
            for position in miner_positions:
                positions_list.append(position)

        return positions_list

    @staticmethod
    def running_leverage_computation(
            positions: list[Position]
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
        """
        positions_copy = copy.deepcopy(positions)
        for position in positions_copy:
            for order in position.orders:
                order.leverage = np.clip(order.leverage, 0, 1)

        return positions

    @staticmethod
    def to_state_list(
            positions: list[Position],
            current_time: int,
            constrain_lookback: bool = True
    ) -> tuple[list[str], list[str], list[dict]]:
        """
        Args:
            positions: list[Position] - the positions
            current_time: int - the current time
            constrain_lookback: bool - whether to constrain the lookback

        Return:
            return: list[dict] - the order list
        """
        order_list = []

        miners = set()
        trade_pairs = set()

        if constrain_lookback:
            start_time = current_time - ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        else:
            start_time = 0

        for position in positions:
            order_start = 0
            order_end = 0
            order_leverage = 0
            order_tradepair = None
            order_minerid = position.miner_hotkey

            if len(position.orders) == 0:
                continue

            for order_number, order in enumerate(position.orders):
                if order_number == 0:
                    order_start = order.processed_ms
                    order_leverage = order.leverage
                    order_tradepair = order.trade_pair.trade_pair_id
                    order_orderid = order.order_uuid
                    continue

                order_end = order.processed_ms

                if order_start >= start_time:
                    miners.add(order_minerid)
                    trade_pairs.add(order_tradepair)
                    order_list.append({
                        "miner_id": order_minerid,
                        "trade_pair": order_tradepair,
                        "leverage": order_leverage,
                        "start": order_start,
                        "end": order_end,
                        "order_id": order_orderid
                    })

                order_start = order_end
                order_leverage = order.leverage
                order_orderid = order.order_uuid

        return (
            sorted(list(miners)),
            sorted(list(trade_pairs)),
            order_list
        )

    @staticmethod
    def condense_positions(
            positions: list[Position],
            time_window: int = None
    )   -> list[Position]:
        """
        Looks at a rolling window of length time_window and condenses positions within this window into one
        position where each position is now represented as an order. Useful for martingale calculations.

        Args:
            positions: list[Position] - the positions of a miner
            time_window: int - the time window to consider possible martingale positions in

        Return:
            return: list[Position] - New list of positions that
        """

        if time_window is None:
            time_window = ValiConfig.MARTINGALE_TIME_WINDOW_MS

        # Need at least two positions for this to work
        if len(positions) <= 1:
            return []

        start_time = positions[0].open_ms

        new_positions = []
        new_position = copy.deepcopy(positions[0])
        new_orders = []
        new_position_type = positions[0].orders[0].order_type

        # For each new condensed position, use the max return seen to understand the benefit the miner has received
        max_return = positions[0].return_at_close
        for i in range(len(positions)):

            # Current position must match position type for the idea of cumulative leverage to work properly
            position_time = positions[i].open_ms

            position_type = None
            if len(positions[i].orders) > 0:
                position_type = positions[i].orders[0].order_type

            if position_time < start_time + time_window and position_type is not None and position_type == new_position_type:

                # Use the average time-weighted position leverage to understand the overall risk of the position
                position_average_leverage = PositionUtils.average_leverage([positions[i]])
                if position_type == OrderType.SHORT:
                    position_average_leverage = -1 * position_average_leverage

                max_return = max(max_return, new_position.return_at_close)

                new_order = copy.deepcopy(positions[i].orders[0])
                new_order.leverage = position_average_leverage

                # Maybe we should use average price here
                new_order.price = positions[i].orders[0].price
                new_orders.append(new_order)

            # If there are more positions, and they don't fall in the time window
            # start a new "position"
            if i < len(positions) - 1:
                if positions[i + 1].open_ms > start_time + time_window and len(positions[i + 1].orders) > 0:
                    new_position.orders = new_orders
                    new_orders = []
                    new_position.return_at_close = max_return
                    new_positions.append(new_position)

                    new_position = copy.deepcopy(positions[i + 1])
                    new_position_type = positions[i + 1].orders[0].order_type

                    max_return = positions[i + 1].return_at_close
                    start_time = positions[i + 1].open_ms
            else:
                new_position.orders = new_orders
                new_position.return_at_close = max_return
                new_positions.append(new_position)

        return new_positions

