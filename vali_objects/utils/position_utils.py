from typing import List, Dict

from vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderType
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
        hotkeys: List[str], eliminations: Dict = None, **args
    ) -> Dict[str, List[Position]]:
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()
        return {
            hotkey: PositionUtils.get_all_miner_positions(hotkey, **args)
            for hotkey in hotkeys
            if hotkey not in eliminated_hotkeys
        }

   

