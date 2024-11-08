# developer: trdougherty
import numpy as np
import copy

from vali_objects.position import Position
from vali_objects.vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderType

from time_util.time_util import TimeUtil


class PositionUtils:
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
