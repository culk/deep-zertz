import numpy as np

from .ZertzLogic import Board


# For full rules: http://www.gipf.com/zertz/rules/rules.html
# Class interface inspired by https://github.com/suragnair/alpha-zero-general

class ZertzGame():
    def __init__(self, rings=37, marbles=None, win_con=None, t=1, clone=None, clone_state=None):
        if clone is not None:
            # Creates an instance of ZertzGame with settings copied from clone and updated to 
            # have the same board state as clone_state
            self.initial_rings = clone.initial_rings
            self.t = clone.t
            self.win_con = clone.win_con
            self.board = Board(clone=clone.board)
            self.board.state = np.copy(clone_state)
            assert clone.board.state.shape[0] == clone_state.shape[0]
        else:
            # The size of the game board
            #   default: 37 rings (approximately 7x7 hex)
            self.initial_rings = rings
            self.t = t
            self.board = Board(self.initial_rings, marbles, self.t)

            # The win conditions (amount of each marble needed)
            #   default:
            #     -3 marbles of each color
            #     -4 white marbles
            #     -5 gray marbles
            #     -6 black marbles
            if win_con is None:
                # Use the default win conditions
                self.win_con = [{'w': 3, 'g': 3, 'b': 3},
                                {'w': 4}, {'g': 5}, {'b': 6}]
            else:
                self.win_con = win_con

    def get_cur_player_value(self):
        # Returns 1 if current player is player 0 and -1 if current player is player 1
        if self.board.state[-1, 0, 0] == 0:
            player_value = 1
        elif self.board.state[-1, 0, 0] == 1:
            player_value = -1
        return player_value

    def get_current_state(self):
        # Returns the game state which is a tuple of:
        #   - 3D matrix of size L x H x W (layers, board height, board width)
        #   - integer (1 or -1) giving the value of the current player
        board_state = np.copy(self.board.state)
        player_value = self.get_cur_player_value()
        return (board_state, player_value)

    def get_next_state(self, action, cur_state=None):
        # Input:
        #   - An action which consists of a marble placement and a ring to remove or a capture
        #   - Optional: cur_state = an arbitrary board state to use instead of the current game state
        # Returns the game state which is a tuple of:
        #   - 3D matrix of size L x H x W (layers, board height, board width)
        #   - integer (1 or -1) giving the value of the current player
        if cur_state is None:
            # Use the internal game state to determine the next state
            self.board.take_action(action)
            next_state = self.get_current_state()
        else:
            # Return the next state for an arbitrary marble supply, board and player
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            next_state = temp_game.get_next_state(action)
        return next_state

    def get_valid_actions(self, cur_state=None):
        # Returns a filtering matrix that can be used to filter and renormalize the policy
        # probability distribution. Capturing is compulsory so if there is a valid capture action
        # then the matrix of placement actions will all be False. Matrix shape depends on the action type.
        #   - for placement actions, shape is 3 x width^2 x (width^2 + 1) and action_type is 'PUT'
        #   - for capture actions, shape is 6 x width x width and action_type is 'CAP'
        #     - capture actions only end the current players turn if there are no more chain captures
        if cur_state is None:
            actions, action_type = self.board.get_valid_moves()
        else:
            # Return the valid actions for an arbitrary marble supply, board and player
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            actions, action_type = temp_game.get_valid_actions()
        return actions, action_type

    def get_capture_action_size(self):
        # Return the number of possible capture actions
        return 6 * self.board.width**2

    def get_placement_action_size(self):
        # Return the number of possible placement actions
        return 3 * self.width**2 * (self.width**2 + 1)

    def _is_game_over(self):
        # Return True if ended or False if not ended
        # Create lists to help index into the game state and win_con
        marble_types = ['w', 'g', 'b']
        captured_layers_start = [self.t * 4 + 4, self.t * 4 + 7]

        # Check if any player's captured marbles are enough to satisfy a win condition
        for win_con in self.win_con:
            # Build the list of required marble amounts
            required = np.zeros(3)
            for i, marble_type in enumerate(marble_types):
                if marble_type in win_con:
                    required[i] = win_con[marble_type]

            # Build the list of captured marble amounts for each player
            for layer_start in captured_layers_start:
                captured = self.board.state[layer_start: layer_start+3, 0, 0]
                if np.all(captured >= required):
                    return True

        # If board has every ring covered with a marble then the last player who played is winner
        if np.all(np.sum(self.board.state[:4], axis=0) != 1):
            return True
        return False

    def get_game_ended(self, cur_state=None):
        # Returns 1 if first player won and -1 if second player won.
        # If no players have won then return 0.
        if cur_state is None:
            if self._is_game_over():
                # The winner is the player that made the previous action
                if np.sum(self.board.state[self.board._CAPTURE_LAYER]) == 0:
                    return -1 * self.get_cur_player_value()
                else:
                    # The game is over in the middle of the players turn during a chain capture
                    # if they have enough marbles to meet a win condition.
                    return self.get_cur_player_value()
            return 0
        else:
            # Return if game is ended for an arbitrary game state
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            return temp_game.get_game_ended()

    def get_symmetries(self):
        # TODO: implement this later for training efficiency improvements
        # There are many symmetries in Zertz
        # First, there are rotational symmetry in that every board position can be rotated in
        # six different ways
        # Second, there are mirror symmetry with every rotation being able to be flipped
        # Third, there are translational symmetries once the board has gotten small enough that 
        # it can be shifted in one of the six directions and still be able to fit within the
        # original space.
        # Total board symmetries = 6 * 2 * (# of shift symmetries)
        symmetries = self.board.get_symmetries()
        return symmetries

