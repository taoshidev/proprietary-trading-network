import threading
from typing import List
from bittensor import Balance

from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager

class MockMDDChecker(MDDChecker):
    def __init__(self, metagraph, position_manager, live_price_fetcher):
        lock = threading.Lock()
        super().__init__(None, metagraph, position_manager, lock, running_unit_tests=True,
                         live_price_fetcher=live_price_fetcher)

    # Lets us bypass the wait period in MDDChecker
    def get_last_update_time_ms(self):
        return 0

class MockPlagiarismDetector(PlagiarismDetector):
    def __init__(self, metagraph):
        super().__init__(None, metagraph, running_unit_tests=True)

    # Lets us bypass the wait period in PlagiarismDetector
    def get_last_update_time_ms(self):
        return 0
    
class MockChallengePeriodManager(ChallengePeriodManager):
    def __init__(self, metagraph):
        super().__init__(None, metagraph, running_unit_tests=True)

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
    def __init__(self, hotkeys, neurons = None):
        self.hotkeys = hotkeys
        self.neurons = neurons

