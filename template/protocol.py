# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List, Type

from vali_objects.vali_dataclasses.signal import Signal


class BaseProtocol(bt.Synapse):
    request_uuid: str = Field(..., allow_mutation=False)
    stream_id: str = Field(..., allow_mutation=False)
    samples: typing.Optional[bt.Tensor] = None
    topic_id: typing.Optional[int] = Field(..., allow_mutation=False)


class SendSignal(bt.Synapse):
    signals: List[typing.Dict]
    successfully_processed: typing.Optional[bool] = None
    error_message: typing.Optional[str] = None


