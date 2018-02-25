import sys
sys.path.append('..')
from collections import Counter


class Player():
    def __init__(self, game, n):
        self.captured = {'b': 0, 'g': 0, 'w': 0}
        self.game = game
        self.n = n

    def get_action():
        pass

class Human(Player):
    pass

class Random(Player):
    pass

