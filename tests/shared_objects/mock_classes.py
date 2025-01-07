import threading
from typing import List
from bittensor import Balance

from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from shared_objects.cache_controller import CacheController


class MockMDDChecker(MDDChecker):
    def __init__(self, metagraph, position_manager, live_price_fetcher):
        lock = threading.Lock()
        super().__init__(None, metagraph, position_manager, lock, running_unit_tests=True,
                         live_price_fetcher=live_price_fetcher)

    # Lets us bypass the wait period in MDDChecker
    def get_last_update_time_ms(self):
        return 0


class MockCacheController(CacheController):
    def __init__(self, metagraph):
        super().__init__(None, metagraph, running_unit_tests=True)


class MockPositionManager(PositionManager):
    def __init__(self, metagraph, perf_ledger_manager):
        super().__init__(None, metagraph, live_price_fetcher=None, running_unit_tests=True,
                         perf_ledger_manager=perf_ledger_manager)


class MockPerfLedgerManager(PerfLedgerManager):
    def __init__(self, metagraph):
        super().__init__(metagraph, live_price_fetcher=None, running_unit_tests=True)


class MockPlagiarismDetector(PlagiarismDetector):
    def __init__(self, metagraph):
        super().__init__(None, metagraph, running_unit_tests=True)

    # Lets us bypass the wait period in PlagiarismDetector
    def get_last_update_time_ms(self):
        return 0


class MockChallengePeriodManager(ChallengePeriodManager):
    def __init__(self, metagraph, position_manager):
        super().__init__(None, metagraph, running_unit_tests=True, position_manager=position_manager)

class MockLivePriceFetcher(LivePriceFetcher):
    def __init__(self, secrets, disable_ws):
        super().__init__(secrets=secrets, disable_ws=disable_ws)

    def get_latest_price(self, trade_pair: TradePair, time_ms=None):
        return [0, []]

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

