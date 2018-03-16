import sys
sys.path.append('.')
import unittest
import numpy as np

from mcts import Node, MCTS
from zertz.ZertzGame import ZertzGame as Game

class DumbNN(object):
    def __init__(self, game):
        self.game = game

    def predict(self, board_state, action_filter):
        # always predicts the uniform distribution over valid moves
        v = self.game.get_game_ended(board_state)
        placement, capture = self.game.get_valid_actions(board_state)
        if np.any(placement):
            placement = placement.astype(np.float32)
            placement[(1, 0, 8)] += 100
            placement = placement.astype(np.float32) / np.sum(placement)
        else:
            capture = capture.astype(np.float32) / np.sum(capture)
        return placement, capture, v

class TestMCTS(unittest.TestCase):
    def test_mcts(self):
        # set up
        rings = 19
        marbles = {'w': 10, 'g': 10, 'b': 10}
        win_con = [{'w': 2}, {'w': 1, 'g': 1, 'b': 1}]
        t = 3
        game = Game(rings, marbles, win_con, t)
        nnet = DumbNN(game)

        # take some actions
        #(('PUT', 'w', (4, 4)), ('REM', (4, 3)))
        game.get_next_state((0, 24, 23), 'PUT')
        #(('PUT', 'b', (3, 4)), ('REM', (4, 2)))
        game.get_next_state((2, 19, 22), 'PUT')
        #(('PUT', 'g', (2, 3)), ('REM', (1, 3)))
        game.get_next_state((1, 13, 8), 'PUT')
        #(('PUT', 'b', (1, 1)), ('REM', (3, 1)))
        game.get_next_state((1, 6, 16), 'PUT')
        #(('PUT', 'b', (2, 1)), ('REM', (0, 2)))
        game.get_next_state((2, 11, 2), 'PUT')
        #(('PUT', 'w', (3, 3)), ('REM', (0, 0)))
        game.get_next_state((0, 18, 0), 'PUT')

        # do MCTS
        board_state, player_value = game.get_current_state()
        print(board_state[0] + board_state[1] + board_state[2]*2 + board_state[3]*3)
        ai = MCTS(game, nnet, 1, 50)
        ai.reset(player_value)
        ai.get_action_prob(board_state, temp=0)

    def test_mcts2(self):
        # set up
        rings = 19
        marbles = {'w': 10, 'g': 10, 'b': 10}
        win_con = [{'w': 2}, {'g': 2}, {'b': 2}, {'w': 1, 'g': 1, 'b': 1}]
        t = 3
        game = Game(rings, marbles, win_con, t)
        nnet = DumbNN(game)

        # take some actions
        #Human:   PUT g B1 B4
        game.get_next_state((1, 16, 1), 'PUT')
        #AI:      PUT b D3 C5
        game.get_next_state((2, 13, 2), 'PUT')
        #Human:   PUT b E1 C4
        game.get_next_state((2, 24, 7), 'PUT')
        #AI:      PUT w B2 D1
        game.get_next_state((0, 11, 23), 'PUT')
        #Human:   CAP g B1 w B3
        game.get_next_state((3, 3, 1), 'CAP')
        #AI:      PUT g A3 D4
        #game.get_next_state((1, 0, 8), 'PUT')
        #Human:   CAP g A3 g C3
        #game.get_next_state((5, 0, 0), 'CAP')
        #Human:   CAP g C3 b E3

        # do MCTS
        board_state, player_value = game.get_current_state()
        print(board_state[0] + board_state[1] + board_state[2]*2 + board_state[3]*3)
        print(board_state[-1])
        ai = MCTS(game, nnet, 1, 6)
        ai.reset(player_value)
        ai.get_action_prob(board_state, temp=0)

if __name__ == '__main__':
    unittest.main()

