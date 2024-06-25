# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List

class SendSignal(bt.Synapse):
    signal: typing.Dict = Field(default_factory=dict, title="Signal", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    validator_hotkey: str = Field("", title="Hotkey set by validator", frozen=False)
    miner_order_uuid: str = Field("", title="Order UUID set by miner", frozen=False)
    required_hash_fields: List[str] = Field(
        ["signal"],
        title="Required Hash Fields",
        description="A list of fields required for the hash."
    )

class GetPositions(bt.Synapse):
    positions: List[typing.Dict] = Field(default_factory=list, title="Positions", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    required_hash_fields: List[str] = Field(
        ["positions"],
        title="Required Hash Fields",
        description="A list of fields required for the hash."
    )

class ValidatorCheckpoint(bt.Synapse):
    checkpoint: str = Field("", title="Checkpoint", allow_mutation=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", allow_mutation=True)
    error_message: str = Field("", title="Error Message", allow_mutation=True)
    validator_receive_hotkey: str = Field("", title="Hotkey set by receiving validator", allow_mutation=True)