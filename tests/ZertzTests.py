import sys
sys.path.append('..')
import unittest
import numpy as np

from zertz.ZertzLogic import Board
from zertz.ZertzGame import ZertzGame


class TestZertzLogic(unittest.TestCase):
    def test_init(self):
        board = Board(37)
        self.assertEqual(board.board_width, 7)
        self.assertEqual(board.rings, 37)
        self.assertEqual(np.sum(board.board_state), 37)

    def test_removable(self):
        board = Board(19)
        self.assertTrue(board._is_removable((0, 0)))
        self.assertTrue(board._is_removable((0, 1)))
        self.assertTrue(board._is_removable((0, 2)))
        self.assertTrue(board._is_removable((1, 3)))
        self.assertTrue(board._is_removable((2, 4)))
        self.assertFalse(board._is_removable((1, 2)))
        self.assertFalse(board._is_removable((1, 1)))
        self.assertFalse(board._is_removable((2, 2)))

    def test_neighbors(self):
        board = Board(19)
        center = (2, 2)
        neighbors = [(3,2), (2,1), (1,1), (1,2), (2,3), (3,3)]
        for index in board._get_neighbors(center):
            self.assertTrue(index in neighbors)

    def test_jump_dst(self):
        board = Board(19)
        center = (2, 2)
        destinations = [(4,2), (2,0), (0,0), (0,2), (2,4), (4,4)]
        for index in board._get_neighbors(center):
            dst = board._get_jump_dst(center, index)
            self.assertTrue(dst in destinations)

    def test_get_middle(self):
        board = Board(19)
        center = (2, 2)
        destinations = [(4,2), (2,0), (0,0), (0,2), (2,4), (4,4)]
        middles = [(3,2), (2,1), (1,1), (1,2), (2,3), (3,3)]
        for dst, mid in zip(destinations, middles):
            self.assertTrue(mid == board._get_middle_ring(center, dst))

    def test_put_action(self):
        board = Board(19)
        board.take_action((('PUT', 'w', (4, 4)), ('REM', (4, 3))), None)
        board.take_action((('PUT', 'b', (3, 4)), ('REM', (4, 2))), None)
        self.assertEqual(board.board_state[4, 4], 2)
        self.assertEqual(board.board_state[4, 3], 0)
        self.assertEqual(board.board_state[3, 4], 4)
        self.assertEqual(board.board_state[4, 2], 0)
        self.assertTrue(board._is_removable((3, 2)))

    def test_capture_moves(self):
        board = Board(19)
        board.take_action((('PUT', 'w', (4, 4)), ('REM', (4, 3))), None)
        board.take_action((('PUT', 'b', (3, 4)), ('REM', (4, 2))), None)
        board.take_action((('PUT', 'g', (2, 3)), ('REM', (1, 3))), None)
        board.take_action((('PUT', 'b', (1, 1)), ('REM', (3, 1))), None)
        board.take_action((('PUT', 'b', (2, 1)), ('REM', (0, 2))), None)
        board.take_action((('PUT', 'w', (3, 3)), ('REM', (0, 0))), None)
        actions = [(('CAP', 'b', (2, 1)), ('b', (0, 1))),
                   (('CAP', 'b', (3, 4)), ('g', (1, 2)), ('b', (1, 0)), ('b', (3, 2))),
                   (('CAP', 'b', (3, 4)), ('w', (3, 2)), ('b', (1, 0)), ('b', (1, 2))),
                   (('CAP', 'w', (4, 4)), ('w', (2, 2)), ('b', (2, 0))),
                   (('CAP', 'w', (4, 4)), ('w', (2, 2)), ('g', (2, 4))),
                   (('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('b', (2, 0)))]
        for move in board._get_capture_moves():
            self.assertTrue(move in actions)

class TestZertzGame(unittest.TestCase):
    def test_init(self):
        game = ZertzGame(19)
        self.assertEqual(game.initial_rings, 19)
        self.assertEqual(np.sum(game.board.board_state), 19)
        self.assertEqual(len(game.players), 2)
        self.assertEqual(game.cur_player, 0)

    def test_get_actions(self):
        game = ZertzGame(19)
        self.assertEqual(len(game.get_valid_actions()), 648)
        game = ZertzGame(37)
        self.assertEqual(len(game.get_valid_actions()), 1944)

        
if __name__ == '__main__':
    unittest.main()
