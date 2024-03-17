from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager


class MockMDDChecker(MDDChecker):
    def __init__(self, metagraph):
        position_manger = PositionManager(metagraph=metagraph, running_unit_tests=True)
        super().__init__(None, metagraph, position_manger, running_unit_tests=True)

    # Lets us bypass the wait period in MDDChecker
    def get_last_update_time_ms(self):
        return 0

class MockPlagiarismDetector(PlagiarismDetector):
    def __init__(self, metagraph):
        super().__init__(None, metagraph, running_unit_tests=True)

    # Lets us bypass the wait period in PlagiarismDetector
    def get_last_update_time_ms(self):
        return 0
class MockMetagraph():
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys
