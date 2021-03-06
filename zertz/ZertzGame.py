import numpy as np
import copy

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
            self.marbles = copy.copy(clone.marbles)
            self.win_con = copy.copy(clone.win_con)
            self.board = Board(clone=clone.board)
            self.board.state = np.copy(clone_state)
            assert clone.board.state.shape[0] == clone_state.shape[0]
        else:
            # The size of the game board
            #   default: 37 rings (approximately 7x7 hex)
            self.initial_rings = rings
            self.t = t
            self.marbles = marbles
            self.board = Board(self.initial_rings, self.marbles, self.t)

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

    def __deepcopy__(self, memo):
        return ZertzGame(clone=self, clone_state=self.board.state)

    def reset_board(self):
        self.board = Board(self.initial_rings, self.marbles, self.t)

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

    def get_next_state(self, action, action_type, cur_state=None):
        # Input:
        #   - action which is an index into the action matrix
        #   - action_type which is 'PUT' for a placement action or 'CAP' for a capture action
        #   - Optional: cur_state = an arbitrary board state to use instead of the current game state
        # Returns the game state which is a tuple of:
        #   - 3D matrix of size L x H x W (layers, board height, board width)
        #   - integer (1 or -1) giving the value of the current player
        if cur_state is None:
            # Use the internal game state to determine the next state
            self.board.take_action(action, action_type)
            board_state, player_value = self.get_current_state()
        else:
            # Return the next state for an arbitrary marble supply, board and player
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            board_state, player_value = temp_game.get_next_state(action, action_type)
        return (board_state, player_value)

    def get_valid_actions(self, cur_state=None):
        # Returns two filtering matrices that can be used to filter and renormalize the policy
        # probability distributions. Capturing is compulsory so if there is a valid capture action
        # then the matrix of placement actions will all be False. Matrix shape depends on the action type.
        #   - for placement actions, shape is 3 x width^2 x (width^2 + 1)
        #   - for capture actions, shape is 6 x width x width
        #     - capture actions only end the current players turn if there are no more chain captures
        if cur_state is None:
            placement, capture = self.board.get_valid_moves()
        else:
            # Return the valid actions for an arbitrary marble supply, board and player
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            placement, capture = temp_game.get_valid_actions()
        return (placement, capture)

    def get_capture_action_size(self):
        # Return the number of possible capture actions
        return 6 * self.board.width**2

    def get_capture_action_shape(self):
        # Return the shape of the capture actions as a tuple
        return self.board.get_capture_shape()

    def get_placement_action_size(self):
        # Return the number of possible placement actions
        return 3 * self.board.width**2 * (self.board.width**2 + 1)

    def get_placement_action_shape(self):
        # Return the shape of the placement actions as a tuple
        return self.board.get_placement_shape()

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

    def get_symmetries(self, cur_state=None):
        # There are many symmetries in Zertz
        # First, there are rotational symmetry in that every board position can be rotated in
        # six different ways
        # Second, there are mirror symmetry with every rotation being able to be flipped
        # Third, there are translational symmetries once the board has gotten small enough that 
        # it can be shifted in one of the six directions and still be able to fit within the
        # original space.
        # Total board symmetries = 6 * 2 * (# of shift symmetries)
        # Total implemented currently = 4
        if cur_state is None:
            symmetries = self.board.get_state_symmetries()
        else:
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            symmetries = temp_game.get_symmetries()
        return symmetries

    def translate_action_symmetry(self, action_type, symmetry, actions):
        translated = np.copy(actions)
        if len(translated.shape) != 3:
            if action_type == 'PUT':
                translated = translated.reshape(self.get_placement_action_shape())
            elif action_type == 'CAP':
                translated = translated.reshape(self.get_capture_action_shape())
        if symmetry == 0: # mirror
            translated = self.board.mirror_action(action_type, translated)
        elif symmetry == 1: # rotated
            translated = self.board.rotate_action(action_type, translated)
        elif symmetry == 2: # mirror/rotated
            translated = self.board.mirror_action(action_type, translated)
            translated = self.board.rotate_action(action_type, translated)
        elif symmetry == 3: # opponent
            pass
        elif symmetry == 4: # opponent mirror
            pass
        elif symmetry == 5: # opponent rotated
            pass
        elif symmetry == 6: # opponent mirror/rotated
            pass
        return translated

    def str_to_action(self, action_str):
        # Translate an action string [i.e. 'PUT w A1 B2' or 'CAP b C4 g C2'] to a tuple/type
        args = action_str.split()
        action_type = args[0]
        if action_type == 'PUT':
            if len(args) == 4:
                marble_type, put_str, rem_str = args[1:]
            elif len(args) == 3:
                marble_type, put_str = args[1:]
                rem_str = None
            else:
                return '', None
            layer = self.board._MARBLE_TO_LAYER[marble_type] - 1
            y, x = self.board.str_to_index(put_str)
            put = y * self.board.width + x
            if rem_str is not None:
                y, x = self.board.str_to_index(rem_str)
                rem = y * self.board.width + x
            else:
                rem = self.board.width**2
            action = (layer, put, rem)
        elif action_type == 'CAP':
            if len(args) == 5:
                _a, src_str, _b, dst_str = args[1:]
            else:
                return '', None
            src = self.board.str_to_index(src_str)
            dst = self.board.str_to_index(dst_str)
            cap = self.board._get_middle_ring(src, dst)
            neighbors = self.board._get_neighbors(src)
            direction = neighbors.index(cap)
            action = (direction, src[0], src[1])
        else:
            action = None
        return action_type, action

    def action_to_str(self, action_type, action):
        # Translate an action tuple and type to a human readable string representation
        action_str = action_type + ' '
        if action_type == 'PUT':
            marble_type, put, rem = action
            marble_type = self.board._LAYER_TO_MARBLE[marble_type + 1]

            put_index = (put / self.board.width, put % self.board.width)
            put_str = self.board.index_to_str(put_index)

            if rem == self.board.width**2:
                rem_str = ''
            else:
                rem_index = rem / self.board.width, rem % self.board.width
                rem_str = self.board.index_to_str(rem_index)

            action_str = "{} {} {} {}".format(
                    action_type, marble_type, put_str, rem_str).rstrip()

        elif action_type == 'CAP':
            direction, y, x = action
            src = (y, x)
            src_marble = self.board._get_marble_type_at(src)
            src_str = self.board.index_to_str(src)

            dy, dx = self.board._DIRECTIONS[direction]
            cap = (y + dy, x + dx)
            cap_marble = self.board._get_marble_type_at(cap)

            dst = self.board._get_jump_dst(src, cap)
            dst_str = self.board.index_to_str(dst)

            action_str = "{} {} {} {} {}".format(
                    action_type, src_marble, src_str, cap_marble, dst_str)

        return action_str

    def print_state(self):
        # Print the board state and supplies to the console
        #   0 - empty space
        #   1 - ring
        #   2 - white marble
        #   3 - gray marble
        #   4 - black marble
        print "---------------"
        print "Board state:"
        print (self.board.state[0] + self.board.state[1] + self.board.state[2] * 2
                + self.board.state[3] * 3)
        print "---------------"
        print "Marble supply:"
        print self.board.state[-10:-1, 0, 0]
        print "---------------"

