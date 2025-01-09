import hashlib
import pickle
import numpy as np
from typing import Union
from vali_objects.vali_dataclasses.order import Order
from vali_objects.position import Position
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedger
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO


def get_time_in_range(percent, start, end):
    return int(start + ((end - start) * percent))


def hash_object(obj):
    serialized_obj = pickle.dumps(obj)
    hash_obj = hashlib.sha256()
    hash_obj.update(serialized_obj)
    hashed_str = hash_obj.hexdigest()
    return hashed_str[:10]


def order_generator(
        order_type=OrderType.LONG,
        leverage=1.0,
        n_orders: int = 10,
        processed_ms: int = 1710521764446
) -> list[Order]:
    order_list = []
    for _ in range(n_orders):
        sample_order = Order(
            order_type=order_type,
            leverage=leverage,
            price=3000,
            trade_pair=TradePair.BTCUSD,
            processed_ms=processed_ms,
            order_uuid="1000"
        )

        order_list.append(sample_order)

    return order_list


def position_generator(
        open_time_ms,
        trade_pair,
        close_time_ms: Union[None, int] = None,
        return_at_close=1.0,
        orders: list[Order] = [],
        miner_hotkey: str = 'miner0'
):
    generated_position = Position(
        miner_hotkey=miner_hotkey,
        position_uuid=hash_object((
            open_time_ms,
            close_time_ms,
            return_at_close,
            trade_pair
        )),
        orders=orders,
        net_leverage=0.0,
        open_ms=open_time_ms,
        trade_pair=trade_pair,
    )

    if close_time_ms is not None:
        generated_position.close_out_position(
            close_ms=close_time_ms
        )
        generated_position.return_at_close = return_at_close

    return generated_position


def generate_ledger(
        value=None,
        start_time=0,
        nterms=None,
        end_time=ValiConfig.TARGET_LEDGER_WINDOW_MS,
        gain=None,
        loss=None,
        open_ms=None,
        mdd=1.0
):
    # Check for invalid input combinations
    if value is None and (gain is None or loss is None):
        raise ValueError("Either 'value' or both 'gain' and 'loss' must be provided")

    # If value is provided and gain/loss are not, set gain/loss based on value
    if value is not None and gain is None and loss is None:
        gain = value
        loss = -value

    # If gain or loss is provided without value, just use the provided gain/loss
    if gain is not None and loss is not None and value is None:
        pass  # gain and loss are already set correctly

    # Determine the number of terms if not provided
    if nterms is None:
        nterms = (end_time - start_time) // ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    # Generate checkpoint times
    checkpoint_times = [start_time + i * ValiConfig.TARGET_CHECKPOINT_DURATION_MS for i in range(nterms)]
    checkpoint_time_accumulation = np.diff(checkpoint_times, prepend=start_time)

    checkpoint_list = []
    for i in range(len(checkpoint_times)):
        if open_ms is None:
            checkpoint_open_ms = checkpoint_time_accumulation[i]
        else:
            checkpoint_open_ms = open_ms

        checkpoint_list.append(
            checkpoint_generator(
                last_update_ms=checkpoint_times[i],
                gain=gain,
                loss=loss,
                prev_portfolio_ret=1.0,
                open_ms=checkpoint_open_ms,
                accum_ms=checkpoint_open_ms,
                mdd=mdd
            )
        )

    base_ledger = ledger_generator(checkpoints=checkpoint_list)
    return {TP_ID_PORTFOLIO: base_ledger, TradePair.BTCUSD.trade_pair_id: base_ledger}


def ledger_generator(
        target_cp_duration: int = ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
        target_ledger_window_ms: float = ValiConfig.TARGET_LEDGER_WINDOW_MS,
        checkpoints=None,
):
    if checkpoints is None:
        checkpoints = []
    return PerfLedger(
        target_cp_duration_ms=target_cp_duration,
        target_ledger_window_ms=target_ledger_window_ms,
        cps=checkpoints
    )


def checkpoint_generator(
        last_update_ms: int = 0,
        prev_portfolio_ret: float = 1.0,
        accum_ms: int = 0,
        open_ms: int = 0,
        n_updates: int = 0,
        gain: float = 0.0,
        loss: float = 0.0,
        mdd: float = 1.0
):
    return PerfCheckpoint(
        last_update_ms=last_update_ms,
        prev_portfolio_ret=prev_portfolio_ret,
        accum_ms=accum_ms,
        open_ms=open_ms,
        n_updates=n_updates,
        gain=gain,
        loss=loss,
        mdd=mdd
    )
def add_orders_to_position(
        position: Position,
        order_type: OrderType,
        trade_pair: TradePair = TradePair.BTCUSD,
        leverages: list[float] = [],
        prices: list[float] = [],
        times: list[int] = [],
        order_uuid: int = 1000 ):
    assert len(leverages) == len(prices), "The lengths of 'leverages' and 'prices' do not match."
    uuid_counter = 0
    for i in range(len(leverages)):
        uuid = order_uuid + uuid_counter
        uuid_counter += 1
        if leverages[i] > 0:
            order_type = OrderType.LONG
        elif leverages[i] < 0:
            order_type = OrderType.SHORT
        else:
            order_type = OrderType.FLAT

        order = Order(
            order_type=order_type,
            leverage=leverages[i],
            price=prices[i],
            trade_pair=trade_pair,
            processed_ms=times[i],
            order_uuid=uuid
        )
        position.add_order(order)