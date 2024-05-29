# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List

class SendSignal(bt.Synapse):
    signal: typing.Dict = Field({}, title="Signal", allow_mutation=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", allow_mutation=True)
    error_message: str = Field("", title="Error Message", allow_mutation=True)
    validator_hotkey: str = Field("", title="Hotkey set by validator", allow_mutation=True)
    miner_order_uuid: str = Field("", title="Order UUID set by miner", allow_mutation=True)

class GetPositions(bt.Synapse):
    positions: List[typing.Dict] = Field([], title="Positions", allow_mutation=True)
    successfully_processed: bool = Field(False, title="Successfully Processed", allow_mutation=True)
    error_message: str = Field("", title="Error Message", allow_mutation=True)
