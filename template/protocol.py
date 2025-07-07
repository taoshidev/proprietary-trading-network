# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List

class SendSignal(bt.Synapse):
    signal: typing.Dict = Field(default_factory=dict, title="Signal", frozen=False, max_length=4096)
    repo_version: str = Field("N/A", title="Repo version (use the same meta.json file as validator)", frozen=False, max_length=256)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    validator_hotkey: str = Field("", title="Hotkey set by validator", frozen=False, max_length=256)
    order_json: str = Field("", title="New Order JSON set by validator", frozen=False)
    miner_order_uuid: str = Field("", title="Order UUID set by miner", frozen=False, max_length=256)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)

SendSignal.required_hash_fields = ["signal"]

class GetPositions(bt.Synapse):
    positions: List[typing.Dict] = Field(default_factory=list, title="Positions", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
    version: int = Field(0, title="Version", frozen=False)

GetPositions.required_hash_fields = ["positions"]

class ValidatorCheckpoint(bt.Synapse):
    checkpoint: str = Field("", title="Checkpoint", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    validator_receive_hotkey: str = Field("", title="Hotkey set by receiving validator", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
ValidatorCheckpoint.required_hash_fields = ["checkpoint"]

class GetDashData(bt.Synapse):
    data: typing.Dict = Field(default_factory=dict, title="Dashboard Data", frozen=False)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)
GetDashData.required_hash_fields = ["data"]

class DepositCollateral(bt.Synapse):
    extrinsic_data: str = Field("", title="Encoded Extrinsic Data", frozen=False)
    amount: int = Field(0, title="Deposit Amount in Theta", frozen=False)
    miner_address: str = Field("", title="Miner SS58 Address", frozen=False, max_length=256)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    transaction_hash: str = Field("", title="EVM Transaction Hash", frozen=False, max_length=256)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)

DepositCollateral.required_hash_fields = ["extrinsic_data", "amount", "miner_address"]

class WithdrawCollateral(bt.Synapse):
    amount: int = Field(0, title="Withdrawal Amount in Theta", frozen=False)
    miner_address: str = Field("", title="Miner SS58 Address", frozen=False, max_length=256)
    successfully_processed: bool = Field(False, title="Successfully Processed", frozen=False)
    error_message: str = Field("", title="Error Message", frozen=False, max_length=4096)
    returned_amount: int = Field(0, title="Actual Amount Withdrawn in Theta", frozen=False)
    computed_body_hash: str = Field("", title="Computed Body Hash", frozen=False)

WithdrawCollateral.required_hash_fields = ["amount", "miner_address"]
