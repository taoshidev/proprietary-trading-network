from ast import List
from sympy import Order

from vali_config import ValiConfig
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position import Position
from vali_objects.utils.challenge_utils import ChallengeBase
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

class PlagiarismDetector(ChallengeBase):
    def __init__(self, config, metagraph):
        super().__init__(config, metagraph)

    def is_order_similar_to_positional_orders(self,
        position_open_ms: int,
        check_order: Order,
        hotkey: str = None,
        hotkeys: List[str] = None,
        **args,
    ):

        trade_pair_fee = ValiConfig.TRADE_PAIR_FEES[check_order.trade_pair]

        if hotkey is None:
            raise ValueError("miner hotkey must be provided.")

        miner_positions_by_hotkey = PositionUtils.get_all_miner_positions_by_hotkey(
            hotkeys, **args
        )
        # don't include their own hotkey
        orders = {
            porder.order_uuid: {"order": porder, "position": position}
            for key, positions in miner_positions_by_hotkey.items()
            for position in positions
            for porder in position.orders
            if key != hotkey
        }

        # check to see if there is a similar order to the miner's in the time window
        # based on ranged values
        for order_uuid, order_info in orders.items():
            if (
                order_info["position"].open_ms < position_open_ms
                and order_info["order"].trade_pair == check_order.trade_pair
                and order_info["order"].processed_ms
                > check_order.processed_ms - ValiConfig.ORDER_SIMILARITY_WINDOW_MS
                and check_order.price * (1 - trade_pair_fee)
                <= order_info["order"].price
                <= check_order.price * (1 + trade_pair_fee)
            ):
                return True
        return False


    def check_plagiarism(self, open_position: Position,
                                   signal_to_order: Order, 
                                   miner_hotkey: str,
                                     eliminations: list,
                                       metagraph) -> None:
        # check to see if order is similar to existing order
        is_similar_order = self.is_order_similar_to_positional_orders(
                open_position.open_ms,
                signal_to_order,
                hotkey=miner_hotkey,
                hotkeys=metagraph.hotkeys,
            )
        
        miner_copying_json = ValiUtils.get_vali_json_file(
            ValiBkpUtils.get_miner_copying_dir()
        )
        # If this is a new miner, use the initial value 0. 
        current_hotkey_mc = miner_copying_json.get(miner_hotkey, 0)
        if is_similar_order:
            current_hotkey_mc += ValiConfig.MINER_COPYING_WEIGHT
            if current_hotkey_mc > 1:
                eliminations.append(miner_hotkey)
                # updating both elims and miner copying
                miner_copying_json[miner_hotkey] = current_hotkey_mc
                ValiBkpUtils.write_file(
                    ValiBkpUtils.get_eliminations_dir(), eliminations
                )
                raise SignalException(
                    f"miner eliminated for signal copying [{miner_hotkey}]."
                )
        else:
            if current_hotkey_mc > 0:
                current_hotkey_mc -= ValiConfig.MINER_COPYING_WEIGHT
                # updating miner copying file
                miner_copying_json[miner_hotkey] = current_hotkey_mc

        ValiBkpUtils.write_file(
            ValiBkpUtils.get_miner_copying_dir(),
            miner_copying_json,
        )
        bt.logging.info(
            f"updated miner copying - [{miner_copying_json[miner_hotkey]}]"
        )

