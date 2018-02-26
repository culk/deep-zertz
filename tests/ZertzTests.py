import sys
sys.path.append('..')
import unittest
import numpy as np

from zertz.ZertzLogic import Board
from zertz.ZertzGame import ZertzGame
from zertz.ZertzPlayer import Player


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

    def test_get_capture_moves(self):
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

    def test_separated_board_simple(self):
        board = Board(19)
        board.take_action((('PUT', 'b', (0, 0)), ('REM', (3, 1))), None)
        board.take_action((('PUT', 'b', (0, 2)), ('REM', (2, 0))), None)
        board.take_action((('PUT', 'b', (2, 4)), ('REM', (2, 1))), None)
        board.take_action((('PUT', 'b', (4, 2)), ('REM', (3, 2))), None)
        self.assertEqual(board.board_state[4, 2], 4)
        player = Player(None, 1)
        board.take_action((('PUT', 'b', (2, 2)), ('REM', (4, 3))), player)
        self.assertEqual(board.board_state[4, 2], 0)
        self.assertEqual(player.captured['b'], 1)

    def test_separated_board_complex(self):
        board = Board(19)
        board.take_action((('PUT', 'b', (0, 0)), ('REM', (3, 1))), None)
        board.take_action((('PUT', 'b', (0, 2)), ('REM', (2, 0))), None)
        board.take_action((('PUT', 'b', (2, 4)), ('REM', (2, 1))), None)
        board.take_action((('PUT', 'w', (4, 2)), ('REM', (3, 2))), None)
        board.take_action((('PUT', 'b', (2, 2)), ('REM', (4, 4))), None)
        board.take_action((('PUT', 'b', (1, 3)), ('REM', (3, 4))), None)
        self.assertEqual(board.board_state[4, 2], 2)
        player = Player(None, 1)
        board.take_action((('PUT', 'b', (0, 1)), ('REM', (3, 3))), player)
        self.assertEqual(player.captured['w'], 0)
        self.assertEqual(board.board_state[4, 2], 2)
        self.assertEqual(board.board_state[4, 3], 1)
        board.take_action((('PUT', 'w', (4, 3)), ('REM', (1, 0))), player)
        self.assertEqual(board.board_state[4, 2], 0)
        self.assertEqual(player.captured['w'], 2)

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
        game.get_next_state((('PUT', 'w', (4, 4)), ('REM', (4, 3))))
        game.get_next_state((('PUT', 'b', (3, 4)), ('REM', (4, 2))))
        game.get_next_state((('PUT', 'g', (2, 3)), ('REM', (1, 3))))
        game.get_next_state((('PUT', 'b', (1, 1)), ('REM', (3, 1))))
        game.get_next_state((('PUT', 'b', (2, 1)), ('REM', (0, 2))))
        self.assertEqual(len(game.get_valid_actions()), 6)
        game = ZertzGame(1)
        self.assertEqual(len(game.get_valid_actions()), 3)
        put, rem = game.get_valid_actions()[0]
        _, rem_index = rem
        self.assertTrue(rem_index is None)

    def test_take_actions(self):
        game = ZertzGame(19)
        game.get_next_state((('PUT', 'w', (4, 4)), ('REM', (4, 3))))
        game.get_next_state((('PUT', 'b', (3, 4)), ('REM', (4, 2))))
        game.get_next_state((('PUT', 'g', (2, 3)), ('REM', (1, 3))))
        game.get_next_state((('PUT', 'b', (1, 1)), ('REM', (3, 1))))
        supply, board, player = game.get_next_state((('PUT', 'b', (2, 1)), ('REM', (0, 2))))
        self.assertTrue(np.all(supply[:3] == [5, 7, 7]))
        self.assertEqual(np.sum(board), 26)
        self.assertEqual(player, -1)
        supply, board, player = game.get_next_state((('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('b', (2, 0))))
        self.assertTrue(np.all(supply[-3:] == [0, 1, 2]))
        self.assertEqual(np.sum(board), 18)
        self.assertTrue(board[2, 0] == 2)
        self.assertTrue(board[4, 4] == 1)
        self.assertTrue(board[2, 3] == 1)
        self.assertTrue(board[1, 1] == 4)
        self.assertEqual(player, 1)

    def test_game_end(self):
        game = ZertzGame(1)
        self.assertEqual(game.get_game_ended(), 0)
        action = game.get_valid_actions()[0]
        game.get_next_state(action)
        self.assertEqual(game.get_game_ended(), 1)
        game = ZertzGame(7)
        self.assertEqual(game.get_game_ended(), 0)
        game.players[1].captured = {'w': 4, 'b': 0, 'g': 0}
        game.board.supply['w'] -= 4
        self.assertEqual(game.get_game_ended(), -1)


        
if __name__ == '__main__':
    unittest.main()
