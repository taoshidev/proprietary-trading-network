# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

class IncorrectLiveResultsCountException(Exception):
    def __init__(self, message):
        super().__init__(self, message)