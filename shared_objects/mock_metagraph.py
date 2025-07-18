from typing import List

from bittensor import Balance


class MockAxonInfo:
    ip: str

    def __init__(self, ip: str):
        self.ip = ip

class MockNeuron:
    axon_info: MockAxonInfo
    stake: Balance

    def __init__(self, axon_info: MockAxonInfo, stake: Balance):
        self.axon_info = axon_info
        self.stake = stake


class MockMetagraph():
    neurons: List[MockNeuron]
    hotkeys: List[str]
    uids: List[int]
    block_at_registration: List[int]

    def __init__(self, hotkeys, neurons = None):
        self.hotkeys = hotkeys
        self.neurons = neurons
        self.uids = []
        self.block_at_registration = []