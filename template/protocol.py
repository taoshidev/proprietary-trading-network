# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List, Type

from vali_objects.vali_dataclasses.signal import Signal


class SendSignal(bt.Synapse):
    signal: typing.Dict
    successfully_processed: typing.Optional[bool] = None
    error_message: typing.Optional[str] = None


class GetPositions(bt.Synapse):
    positions: List[typing.Dict]
