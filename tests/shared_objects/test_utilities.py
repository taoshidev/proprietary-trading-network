import hashlib
import pickle
import numpy as np
from typing import Union
from vali_objects.vali_dataclasses.order import Order
from vali_objects.position import Position
from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.perf_ledger import PerfCheckpoint, PerfLedger
from vali_config import ValiConfig

def get_time_in_range(percent, start, end):
    return int(start + ((end - start) * percent))

def hash_object(obj):
    serialized_obj = pickle.dumps(obj)
    hash_obj = hashlib.sha256()
    hash_obj.update(serialized_obj)
    hashed_str = hash_obj.hexdigest()
    return hashed_str[:10]

def order_generator(
    order_type = OrderType.LONG,
    leverage = 1.0,
    n_orders:int = 10,
    processed_ms:int = 1710521764446
) -> list[Order]:
    order_list = []
    for _ in range(n_orders):
        sample_order = Order(
            order_type = order_type,
            leverage = leverage,
            price = 3000,
            trade_pair = TradePair.BTCUSD,
            processed_ms = processed_ms,
            order_uuid = "1000"
        )
        
        order_list.append(sample_order)

    return order_list

def position_generator(
    open_time_ms, 
    trade_pair,
    close_time_ms: Union[None, int] = None,
    return_at_close = 1.0,
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
            close_ms = close_time_ms
        )
        generated_position.return_at_close = return_at_close

    return generated_position

def generate_ledger(
        nterms, 
        value, 
        start_time=0, 
        end_time=ValiConfig.SET_WEIGHT_MINIMUM_TOTAL_CHECKPOINT_DURATION_MS,
        gain=None,
        loss=None,
        open_ms=None
    ):

    if gain is None and loss is None:
        gain = value
        loss = -value

    checkpoint_list = []
    checkpoint_times = np.linspace(start_time, end_time, nterms, dtype=int).tolist()
    checkpoint_time_accumulation = np.diff(checkpoint_times, prepend=0)

    for i in range(nterms):
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
                open_ms=checkpoint_open_ms
            )
        )

    return ledger_generator(checkpoints=checkpoint_list)

def ledger_generator(
    target_cp_duration: int = 21600000,
    target_ledger_window_ms: float = 2592000000,
    checkpoints: list[PerfCheckpoint] = [],
):
    return PerfLedger(
        target_cp_duration=target_cp_duration,
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
    loss: float = 0.0
):
    return PerfCheckpoint(
        last_update_ms=last_update_ms,
        prev_portfolio_ret=prev_portfolio_ret,
        accum_ms=accum_ms,
        open_ms=open_ms,
        n_updates=n_updates,
        gain=gain,
        loss=loss
    )