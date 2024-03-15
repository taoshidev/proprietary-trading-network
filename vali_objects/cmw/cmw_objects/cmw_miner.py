# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

class CMWMiner:
    def __init__(self, miner_id):
        self.miner_id = miner_id
        self.wins = 0
        self.win_value = 0
        self.unscaled_scores = []
        self.win_scores = []

    def set_wins(self, wins):
        self.wins = wins
        return self

    def set_win_value(self, win_value):
        self.win_value = win_value
        return self

    def set_unscaled_scores(self, unscaled_scores):
        self.unscaled_scores = unscaled_scores
        return self

    def set_win_scores(self, win_scores):
        self.win_scores = win_scores
        return self

    def add_unscaled_score(self, score):
        self.unscaled_scores.append(score)

    def add_win(self):
        self.wins += 1

    def add_win_score(self, win_score):
        self.win_scores.append(win_score)
