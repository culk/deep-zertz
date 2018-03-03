import sys
sys.path.append('.')
import unittest
import numpy as np

from zertz.ZertzLogic import Board
from zertz.ZertzGame import ZertzGame
from zertz.ZertzPlayer import Player


class TestZertzLogic(unittest.TestCase):
    def test_init(self):
        board = Board(37)
        self.assertEqual(board.width, 7)
        self.assertEqual(board.rings, 37)
        self.assertEqual(np.sum(board.state[0]), 37)
        self.assertEqual(board.state.shape[0], 15)
        self.assertTrue(np.all(board.state[5] == 6))
        self.assertTrue(np.all(board.state[6] == 8))
        self.assertTrue(np.all(board.state[7] == 10))
        self.assertEqual(np.sum(board.state), 1213)
        board = Board(37, t=7)
        self.assertEqual(board.state.shape[0], 39)
        self.assertEqual(np.sum(board.state), 1213)

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
        board = Board(19, t=2)
        board.take_action((('PUT', 'w', (4, 4)), ('REM', (4, 3))), None)
        self.assertEqual(board.state[18, 0, 0], 1)
        self.assertEqual(board.state[1, 4, 4], 1)
        self.assertEqual(board.state[0, 4, 3], 0)
        self.assertEqual(board.state[9, 0, 0], 5)
        board.take_action((('PUT', 'b', (3, 4)), ('REM', (4, 2))), None)
        self.assertEqual(board.state[18, 0, 0], 0)
        self.assertEqual(board.state[3, 3, 4], 1)
        self.assertEqual(board.state[0, 4, 2], 0)
        self.assertEqual(board.state[11, 0, 0], 9)
        self.assertEqual(board.state[5, 4, 4], 1)
        self.assertEqual(board.state[4, 4, 3], 0)
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
                   (('CAP', 'b', (3, 4)), ('g', (1, 2)), ('b', (1, 0)), ('b', (3, 2)), ('w', (3, 4))),
                   (('CAP', 'b', (3, 4)), ('w', (3, 2)), ('b', (1, 0)), ('b', (1, 2)), ('g', (3, 4))),
                   (('CAP', 'w', (4, 4)), ('w', (2, 2)), ('b', (2, 0))),
                   (('CAP', 'w', (4, 4)), ('w', (2, 2)), ('g', (2, 4)), ('b', (4, 4))),
                   (('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('b', (2, 0))),
                   (('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('w', (4, 4)))]
        for move in board._get_capture_moves():
            self.assertTrue(move in actions)

    def test_separated_board_simple(self):
        # Captured black marble goes to player 1
        board = Board(19)
        board.take_action((('PUT', 'b', (0, 0)), ('REM', (3, 1))), None)
        board.take_action((('PUT', 'b', (0, 2)), ('REM', (2, 0))), None)
        board.take_action((('PUT', 'b', (2, 4)), ('REM', (2, 1))), None)
        board.take_action((('PUT', 'b', (4, 2)), ('REM', (3, 2))), None)
        self.assertEqual(board.state[3, 4, 2], 1)
        board.take_action((('PUT', 'b', (2, 2)), ('REM', (4, 3))), None)
        self.assertEqual(board.state[3, 4, 2], 0)
        self.assertEqual(board.state[0, 4, 2], 0)
        self.assertEqual(board.state[10, 0, 0], 1)
        self.assertEqual(board.state[13, 0, 0], 0)
        # Captured black marble goes to player 2
        board = Board(19)
        board.take_action((('PUT', 'b', (0, 0)), ('REM', (3, 1))), None)
        board.take_action((('PUT', 'b', (0, 2)), ('REM', (2, 0))), None)
        board.take_action((('PUT', 'b', (2, 4)), ('REM', (2, 1))), None)
        board.take_action((('PUT', 'b', (4, 2)), ('REM', (3, 2))), None)
        board.take_action((('PUT', 'b', (1, 0)), ('REM', (0, 1))), None)
        self.assertEqual(board.state[3, 4, 2], 1)
        board.take_action((('PUT', 'b', (2, 2)), ('REM', (4, 3))), None)
        self.assertEqual(board.state[3, 4, 2], 0)
        self.assertEqual(board.state[0, 4, 2], 0)
        self.assertEqual(board.state[10, 0, 0], 0)
        self.assertEqual(board.state[13, 0, 0], 1)

    def test_separated_board_complex(self):
        board = Board(19)
        board.take_action((('PUT', 'b', (0, 0)), ('REM', (3, 1))), None)
        board.take_action((('PUT', 'b', (0, 2)), ('REM', (2, 0))), None)
        board.take_action((('PUT', 'b', (2, 4)), ('REM', (2, 1))), None)
        board.take_action((('PUT', 'w', (4, 2)), ('REM', (3, 2))), None)
        board.take_action((('PUT', 'b', (2, 2)), ('REM', (4, 4))), None)
        board.take_action((('PUT', 'b', (1, 3)), ('REM', (3, 4))), None)
        self.assertEqual(board.state[1, 4, 2], 1)
        # This action creates a separable region but that region has empty rings
        board.take_action((('PUT', 'b', (0, 1)), ('REM', (3, 3))), None)
        self.assertEqual(board.state[0, 4, 2], 1)
        self.assertEqual(board.state[1, 4, 2], 1)
        self.assertEqual(board.state[0, 4, 3], 1)
        self.assertEqual(board.state[1, 4, 3], 0)
        self.assertEqual(board.state[8, 0, 0], 0)
        self.assertEqual(board.state[11, 0, 0], 0)
        # This action fills the last ring in separable region with a white marble
        board.take_action((('PUT', 'w', (4, 3)), ('REM', (1, 0))), None)
        self.assertEqual(board.state[0, 4, 2], 0)
        self.assertEqual(board.state[1, 4, 2], 0)
        self.assertEqual(board.state[0, 4, 3], 0)
        self.assertEqual(board.state[1, 4, 3], 0)
        # Result is two white marbles captured by player 2
        self.assertEqual(board.state[8, 0, 0], 0)
        self.assertEqual(board.state[11, 0, 0], 2)

class TestZertzGame(unittest.TestCase):
    def test_init(self):
        game = ZertzGame(19)
        self.assertEqual(game.initial_rings, 19)
        self.assertEqual(np.sum(game.board.state), 619)
        self.assertEqual(game.board._get_cur_player(), 0)
        game = ZertzGame(19, {'w': 2, 'g': 2, 'b': 2})
        self.assertEqual(game.initial_rings, 19)
        self.assertEqual(game.board.state[5, 0, 0], 2)
        self.assertEqual(game.board.state[6, 0, 0], 2)
        self.assertEqual(game.board.state[7, 0, 0], 2)
        self.assertEqual(np.sum(game.board.state), 169)

    def test_get_actions(self):
        game = ZertzGame(19)
        self.assertEqual(len(game.get_valid_actions()), 648)
        game = ZertzGame(37)
        self.assertEqual(len(game.get_valid_actions()), 1944)
        state = game.get_next_state((('PUT', 'w', (4, 4)), ('REM', (4, 3))))
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
        self.assertTrue(np.all(board[5:8, 0, 0] == [5, 7, 7]))
        self.assertEqual(np.sum(board), 519)
        self.assertEqual(player, -1)
        supply, board, player = game.get_next_state((('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('b', (2, 0))))
        self.assertTrue(np.all(board[11:14, 0, 0] == [0, 1, 2]))
        self.assertEqual(np.sum(board), 566)
        self.assertTrue(np.all(board[:4, 2, 0] == [1, 1, 0, 0]))
        self.assertTrue(np.all(board[:4, 4, 4] == [1, 0, 0, 0]))
        self.assertTrue(np.all(board[:4, 2, 3] == [1, 0, 0, 0]))
        self.assertTrue(np.all(board[:4, 1, 1] == [1, 0, 0, 1]))
        self.assertEqual(player, 1)

    def test_game_end(self):
        game = ZertzGame(1)
        self.assertEqual(game.get_game_ended(), 0)
        action = game.get_valid_actions()[0]
        game.get_next_state(action)
        self.assertEqual(game.get_game_ended(), 1)
        game = ZertzGame(7)
        self.assertEqual(game.get_game_ended(), 0)
        # TODO: remove player
        game.players[1].captured = {'w': 4, 'b': 0, 'g': 0}
        game.board.state[11] += 4
        #game.board.supply['w'] -= 4
        game.board.state[5] -= 4
        self.assertEqual(game.get_game_ended(), -1)

        
if __name__ == '__main__':
    unittest.main()
