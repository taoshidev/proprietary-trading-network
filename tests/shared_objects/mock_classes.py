from vali_objects.utils.MDDChecker import MDDChecker
from vali_objects.utils.PlagiarismDetector import PlagiarismDetector


class MockMDDChecker(MDDChecker):
    def __init__(self, metagraph):
        super().__init__(None, metagraph)

    # Lets us bypass the wait period in MDDChecker
    def get_last_update_time_ms(self):
        return 0

class MockPlagiarismDetector(PlagiarismDetector):
    def __init__(self, metagraph):
        super().__init__(None, metagraph)

    # Lets us bypass the wait period in PlagiarismDetector
    def get_last_update_time_ms(self):
        return 0
class MockMetagraph():
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys
