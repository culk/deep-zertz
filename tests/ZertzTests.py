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
        
if __name__ == '__main__':
    unittest.main()
