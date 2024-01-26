# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List

from vali_objects.dataclasses.signal import Signal


class BaseProtocol(bt.Synapse):
    request_uuid: str = Field(..., allow_mutation=False)
    stream_id: str = Field(..., allow_mutation=False)
    samples: typing.Optional[bt.Tensor] = None
    topic_id: typing.Optional[int] = Field(..., allow_mutation=False)


class SendSignal(bt.Synapse):
    signals: List[typing.Dict]
    received: typing.Optional[bool] = None


def convert_to_send_signal(signals = List[Signal]):
    converted_signals = [{"order_type": signal.order_type, "leverage": signal.leverage} for signal in signals]
    return SendSignal(signals=converted_signals)

