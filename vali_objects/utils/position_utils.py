from typing import List, Dict

from vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderTypeEnum
from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.order import Order


class PositionUtils:
    @staticmethod
    def get_return_per_closed_position(positions: List[Position]) -> List[float]:
        closed_position_returns = [
            position.return_at_close
            for position in positions
            if position.is_closed_position
        ]
        cumulative_return = 1
        per_position_return = []

        # calculate the return over time at each position close
        for value in closed_position_returns:
            cumulative_return *= value
            per_position_return.append(cumulative_return)
        return per_position_return

    @staticmethod
    def get_all_miner_positions(
        miner_hotkey: str,
        only_open_positions: bool = False,
        sort_positions: bool = False,
        acceptable_position_end_ms: int = None,
    ) -> List[Position]:
        def _sort_by_close_ms(_position):
            # Treat None values as largest possible value
            return (
                _position.close_ms if _position.close_ms is not None else float("inf")
            )

        miner_dir = ValiBkpUtils.get_miner_position_dir(miner_hotkey)
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)

        positions = [ValiUtils.get_miner_positions(file) for file in all_files]

        if acceptable_position_end_ms is not None:
            positions = [
                position
                for position in positions
                if position.open_ms > acceptable_position_end_ms
            ]

        if only_open_positions:
            positions = [
                position for position in positions if position.close_ms is None
            ]

        if sort_positions:
            positions = sorted(positions, key=_sort_by_close_ms)
        return positions

    @staticmethod
    def get_all_miner_positions_by_hotkey(
        hotkeys: List[str], eliminations: List[str] = None, **args
    ) -> Dict[str, List[Position]]:
        return {
            hotkey: PositionUtils.get_all_miner_positions(hotkey, **args)
            for hotkey in hotkeys
            if eliminations is None
            or (eliminations is not None and hotkey not in eliminations)
        }

    @staticmethod
    def is_order_similar_to_positional_orders(
        check_order: Order,
        hotkey: str = None,
        hotkeys: List[str] = None,
        orders: List[Order] = None,
        **args,
    ):
        if orders is None and hotkeys is None:
            raise ValueError(f"either hotkeys or orders need to be provided.")

        trade_pair_fee = ValiConfig.TRADE_PAIR_FEES[check_order.trade_pair]

        if orders is None:
            if hotkey is None:
                raise ValueError("miner hotkey must be provided.")

            miner_positions_by_hotkey = PositionUtils.get_all_miner_positions_by_hotkey(
                hotkeys, **args
            )
            # don't include their own hotkey
            orders = [
                porder
                for key, positions in miner_positions_by_hotkey.items()
                for position in positions
                for porder in position.orders
                if key != hotkey
            ]

        # check to see if there is a similar order to the miner's in the time window
        # based on ranged values
        for order in orders:
            if (
                order.trade_pair == check_order.trade_pair
                and order.processed_ms
                > check_order.processed_ms - ValiConfig.ORDER_SIMILARITY_WINDOW_MS
                and check_order.price * (1 - trade_pair_fee)
                <= order.price
                <= check_order.price * (1 + trade_pair_fee)
            ):
                return True
        return False

    @staticmethod
    def is_position_similar_to_existing_positions(
        check_position: Position,
        hotkey: str = None,
        hotkeys: List[str] = None,
        miner_positions_by_hotkey: List[str, List[Position]] = None,
        **args,
    ):
        if miner_positions_by_hotkey is None or (hotkeys is None and miner_positions_by_hotkey is None):
            raise ValueError("need to provide positions by hotkey or hotkeys and miner hotkey")

        # get miner positions by hotkey if not provided
        if miner_positions_by_hotkey is None:
            miner_positions_by_hotkey = PositionUtils.get_all_miner_positions_by_hotkey(
                hotkeys, **args
            )
            del miner_positions_by_hotkey[hotkey]


