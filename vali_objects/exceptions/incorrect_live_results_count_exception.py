# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

class IncorrectLiveResultsCountException(Exception):
    def __init__(self, message):
        super().__init__(self, message)