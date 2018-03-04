import sys
sys.path.append('.')
import unittest
import numpy as np

from zertz.ZertzLogic import Board
from zertz.ZertzGame import ZertzGame


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
        #(('PUT', 'w', (4, 4)), ('REM', (4, 3)))
        board.take_action((0, 24, 23), 'PUT')
        self.assertEqual(board.state[18, 0, 0], 1)
        self.assertEqual(board.state[1, 4, 4], 1)
        self.assertEqual(board.state[0, 4, 3], 0)
        self.assertEqual(board.state[9, 0, 0], 5)
        #(('PUT', 'b', (3, 4)), ('REM', (4, 2)))
        board.take_action((2, 19, 22), 'PUT')
        self.assertEqual(board.state[18, 0, 0], 0)
        self.assertEqual(board.state[3, 3, 4], 1)
        self.assertEqual(board.state[0, 4, 2], 0)
        self.assertEqual(board.state[11, 0, 0], 9)
        self.assertEqual(board.state[5, 4, 4], 1)
        self.assertEqual(board.state[4, 4, 3], 0)
        self.assertTrue(board._is_removable((3, 2)))

    def test_get_capture_moves(self):
        board = Board(19)
        #(('PUT', 'w', (4, 4)), ('REM', (4, 3)))
        board.take_action((0, 24, 23), 'PUT')
        #(('PUT', 'b', (3, 4)), ('REM', (4, 2)))
        board.take_action((2, 19, 22), 'PUT')
        #(('PUT', 'g', (2, 3)), ('REM', (1, 3)))
        board.take_action((1, 13, 8), 'PUT')
        #(('PUT', 'b', (1, 1)), ('REM', (3, 1)))
        board.take_action((2, 6, 16), 'PUT')
        #(('PUT', 'b', (2, 1)), ('REM', (0, 2)))
        board.take_action((2, 11, 2), 'PUT')
        #(('PUT', 'w', (3, 3)), ('REM', (0, 0)))
        board.take_action((0, 18, 0), 'PUT')
        #actions = [(('CAP', 'b', (2, 1)), ('b', (0, 1))),
        #           (('CAP', 'b', (3, 4)), ('g', (1, 2)), ('b', (1, 0)), ('b', (3, 2)), ('w', (3, 4))),
        #           (('CAP', 'b', (3, 4)), ('w', (3, 2)), ('b', (1, 0)), ('b', (1, 2)), ('g', (3, 4))),
        #           (('CAP', 'w', (4, 4)), ('w', (2, 2)), ('b', (2, 0))),
        #           (('CAP', 'w', (4, 4)), ('w', (2, 2)), ('g', (2, 4)), ('b', (4, 4))),
        #           (('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('b', (2, 0))),
        #           (('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('w', (4, 4)))]
        test_actions = np.zeros((6, 5, 5), dtype=bool)
        options = [(3, 4, 4), (1, 3, 4), (2, 3, 4), (3, 2, 1), (2, 4, 4)]
        for option in options:
            test_actions[option] = True
        valid_actions, action_type = board.get_valid_moves()
        self.assertEqual(action_type, 'CAP')
        self.assertTrue(np.all(test_actions == valid_actions))
        # Make one capture jump
        board.take_action((2, 4, 4), 'CAP')
        options = [(1, 2, 2), (4, 2, 2)]
        test_actions = np.zeros((6, 5, 5), dtype=bool)
        for option in options:
            test_actions[option] = True
        valid_actions, action_type = board.get_valid_moves()
        self.assertTrue(np.sum(board.state[board._CAPTURE_LAYER]) == 1)
        self.assertEqual(board.state[board._CAPTURE_LAYER, 2, 2], 1)
        self.assertEqual(action_type, 'CAP')
        self.assertTrue(np.all(test_actions == valid_actions))

    def test_separated_board_simple(self):
        # Captured black marble goes to player 1
        board = Board(19)
        #(('PUT', 'b', (0, 0)), ('REM', (3, 1)))
        board.take_action((2, 0, 16), 'PUT')
        #(('PUT', 'b', (0, 2)), ('REM', (2, 0)))
        board.take_action((2, 2, 10), 'PUT')
        #(('PUT', 'b', (2, 4)), ('REM', (2, 1)))
        board.take_action((2, 14, 11), 'PUT')
        #(('PUT', 'b', (4, 2)), ('REM', (3, 2)))
        board.take_action((2, 22, 17), 'PUT')
        self.assertEqual(board.state[3, 4, 2], 1)
        #(('PUT', 'b', (2, 2)), ('REM', (4, 3)))
        board.take_action((2, 12, 23), 'PUT')
        self.assertEqual(board.state[3, 4, 2], 0)
        self.assertEqual(board.state[0, 4, 2], 0)
        self.assertEqual(board.state[10, 0, 0], 1)
        self.assertEqual(board.state[13, 0, 0], 0)
        # Captured black marble goes to player 2
        board = Board(19)
        #(('PUT', 'b', (0, 0)), ('REM', (3, 1)))
        board.take_action((2, 0, 16), 'PUT')
        #(('PUT', 'b', (0, 2)), ('REM', (2, 0)))
        board.take_action((2, 2, 10), 'PUT')
        #(('PUT', 'b', (2, 4)), ('REM', (2, 1)))
        board.take_action((2, 14, 11), 'PUT')
        #(('PUT', 'b', (4, 2)), ('REM', (3, 2)))
        board.take_action((2, 22, 17), 'PUT')
        #(('PUT', 'b', (1, 0)), ('REM', (0, 1)))
        board.take_action((2, 5, 1), 'PUT')
        self.assertEqual(board.state[3, 4, 2], 1)
        #(('PUT', 'b', (2, 2)), ('REM', (4, 3)))
        board.take_action((2, 12, 23), 'PUT')
        self.assertEqual(board.state[3, 4, 2], 0)
        self.assertEqual(board.state[0, 4, 2], 0)
        self.assertEqual(board.state[10, 0, 0], 0)
        self.assertEqual(board.state[13, 0, 0], 1)

    def test_separated_board_complex(self):
        board = Board(19)
        #(('PUT', 'b', (0, 0)), ('REM', (3, 1)))
        board.take_action((2, 0, 16), 'PUT')
        #(('PUT', 'b', (0, 2)), ('REM', (2, 0)))
        board.take_action((2, 2, 10), 'PUT')
        #(('PUT', 'b', (2, 4)), ('REM', (2, 1)))
        board.take_action((2, 14, 11), 'PUT')
        #(('PUT', 'w', (4, 2)), ('REM', (3, 2)))
        board.take_action((0, 22, 17), 'PUT')
        #(('PUT', 'b', (2, 2)), ('REM', (4, 4)))
        board.take_action((2, 12, 24), 'PUT')
        #(('PUT', 'b', (1, 3)), ('REM', (3, 4)))
        board.take_action((2, 8, 19), 'PUT')
        self.assertEqual(board.state[1, 4, 2], 1)
        # This action creates a separable region but that region has empty rings
        #(('PUT', 'b', (0, 1)), ('REM', (3, 3)))
        board.take_action((2, 1, 18), 'PUT')
        self.assertEqual(board.state[0, 4, 2], 1)
        self.assertEqual(board.state[1, 4, 2], 1)
        self.assertEqual(board.state[0, 4, 3], 1)
        self.assertEqual(board.state[1, 4, 3], 0)
        self.assertEqual(board.state[8, 0, 0], 0)
        self.assertEqual(board.state[11, 0, 0], 0)
        # This action fills the last ring in separable region with a white marble
        #(('PUT', 'w', (4, 3)), ('REM', (1, 0)))
        board.take_action((0, 23, 5), 'PUT')
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
        self.assertEqual(game.board.get_cur_player(), 0)
        game = ZertzGame(19, {'w': 2, 'g': 2, 'b': 2})
        self.assertEqual(game.initial_rings, 19)
        self.assertEqual(game.board.state[5, 0, 0], 2)
        self.assertEqual(game.board.state[6, 0, 0], 2)
        self.assertEqual(game.board.state[7, 0, 0], 2)
        self.assertEqual(np.sum(game.board.state), 169)

    def test_get_actions(self):
        game = ZertzGame(19)
        actions, action_type = game.get_valid_actions()
        self.assertEqual(np.sum(actions), 648)
        self.assertEqual(action_type, 'PUT')
        game = ZertzGame(37)
        actions, action_type = game.get_valid_actions()
        self.assertEqual(np.sum(actions), 1944)
        self.assertEqual(action_type, 'PUT')
        #(('PUT', 'w', (4, 4)), ('REM', (4, 3)))
        state = game.get_next_state((0, 32, 31), 'PUT')
        #(('PUT', 'b', (3, 4)), ('REM', (4, 2)))
        game.get_next_state((2, 25, 30), 'PUT')
        #(('PUT', 'g', (2, 3)), ('REM', (1, 3)))
        game.get_next_state((1, 17, 10), 'PUT')
        #(('PUT', 'b', (1, 1)), ('REM', (3, 1)))
        game.get_next_state((2, 8, 22), 'PUT')
        #(('PUT', 'b', (2, 1)), ('REM', (0, 2)))
        game.get_next_state((2, 15, 2), 'PUT')
        actions, action_type = game.get_valid_actions()
        self.assertEqual(np.sum(actions), 5)
        self.assertEqual(action_type, 'CAP')
        game = ZertzGame(1)
        actions, action_type = game.get_valid_actions()
        self.assertEqual(np.sum(actions), 3)
        self.assertTrue(np.all(actions[:, :, 1]))
        self.assertEqual(action_type, 'PUT')

    def test_take_actions(self):
        game = ZertzGame(19)
        #(('PUT', 'w', (4, 4)), ('REM', (4, 3)))
        game.get_next_state((0, 24, 23), 'PUT')
        #(('PUT', 'b', (3, 4)), ('REM', (4, 2)))
        game.get_next_state((2, 19, 22), 'PUT')
        #(('PUT', 'g', (2, 3)), ('REM', (1, 3)))
        game.get_next_state((1, 13, 8), 'PUT')
        #(('PUT', 'b', (1, 1)), ('REM', (3, 1)))
        game.get_next_state((2, 6, 16), 'PUT')
        #(('PUT', 'b', (2, 1)), ('REM', (0, 2)))
        state, player_value = game.get_next_state((2, 11, 2), 'PUT')
        self.assertTrue(np.all(state[5:8, 0, 0] == [5, 7, 7]))
        self.assertEqual(np.sum(state), 519)
        self.assertEqual(player_value, -1)
        #(('CAP', 'w', (4, 4)), ('b', (2, 4)), ('g', (2, 2)), ('b', (2, 0)))
        state, player_value = game.get_next_state((3, 4, 4), 'CAP')
        self.assertEqual(player_value, -1)
        state, player_value = game.get_next_state((1, 2, 4), 'CAP')
        self.assertEqual(player_value, -1)
        state, player_value = game.get_next_state((1, 2, 2), 'CAP')
        self.assertTrue(np.all(state[11:14, 0, 0] == [0, 1, 2]))
        self.assertEqual(np.sum(state), 566)
        self.assertTrue(np.all(state[:4, 2, 0] == [1, 1, 0, 0]))
        self.assertTrue(np.all(state[:4, 4, 4] == [1, 0, 0, 0]))
        self.assertTrue(np.all(state[:4, 2, 3] == [1, 0, 0, 0]))
        self.assertTrue(np.all(state[:4, 1, 1] == [1, 0, 0, 1]))
        self.assertEqual(np.sum(state[game.board._CAPTURE_LAYER]), 0)
        self.assertEqual(player_value, 1)

    def test_game_end(self):
        game = ZertzGame(1)
        self.assertEqual(game.get_game_ended(), 0)
        game.get_next_state((2, 0, 1), 'PUT')
        self.assertEqual(game.get_game_ended(), 1)
        game = ZertzGame(7)
        self.assertEqual(game.get_game_ended(), 0)
        game.board.state[11] += 4
        game.board.state[5] -= 4
        self.assertEqual(game.get_game_ended(), -1)

        
if __name__ == '__main__':
    unittest.main()
