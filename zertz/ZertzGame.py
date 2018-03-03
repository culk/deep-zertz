import numpy as np

from .ZertzLogic import Board
from .ZertzPlayer import Player


# For full rules: http://www.gipf.com/zertz/rules/rules.html
# Class interface inspired by https://github.com/suragnair/alpha-zero-general

class ZertzGame():
    def __init__(self, rings=37, marbles=None, win_con=None, t=1, clone=None, clone_state=None):
        if clone is not None:
            # Creates an instance of ZertzGame that is a copy of clone and updated to 
            # have the same state as clone_state
            supply, state, cur_player = clone_state
            marble_types = ['w', 'g', 'b']

            self.initial_rings = clone.initial_rings
            self.t = clone.t
            self.board = Board(clone=clone.board)
            self.board.supply = dict(zip(marble_types, supply[:3]))
            self.state = np.copy(state)

            self.players = [Player(self, player.n) for player in clone.players]
            self.player[0].captured = dict(zip(marble_types, supply[3:6]))
            self.player[1].captured = dict(zip(marble_types, supply[6:]))
            if cur_player == -1:
                self.cur_player = 1
            else:
                self.cur_player = 0
            self.win_con = clone.win_con
        else:
            # The size of the game board
            #   default: 37 rings (approximately 7x7 hex)
            self.initial_rings = rings
            self.t = t
            self.board = Board(self.initial_rings, marbles, self.t)
            self.players = [Player(self, i) for i in [1, -1]]
            self.cur_player = 0

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

    def _get_marble_state(self):
        type_to_i = {'w': 0, 'g': 1, 'b': 2}
        state = np.zeros(9)
        for marble_type in self.board.supply:
            state[type_to_i[marble_type]] = self.board.supply[marble_type]
        for marble_type in self.players[0].captured:
            state[type_to_i[marble_type] + 3] = self.players[0].captured[marble_type]
        for marble_type in self.players[1].captured:
            state[type_to_i[marble_type] + 6] = self.players[1].captured[marble_type]
        return state

    def get_current_state(self):
        # TODO: return a h x w x layers matrix, the number of layers is given by:
        #   - (# of marble types + 1) x (time history) binary to record previous board positions
        #   - 1 layer binary with a 1 at the index of a marble that needs to be used for capture
        #   - 9 layers, each same value one for each index in the supply
        #   - 1 layer of the same value for the current player
        # Returns the game state which is a tuple of:
        #   - 2D matrix of size self.board.width representing the board state
        #   - vector of length 9 containing marble counts in the order:
        #     (supply 'w', supply 'g', supply 'b', player 1 'w', ..., player 2 'b')
        #   - integer (0 or 1) giving the current player
        board_state = np.copy(self.board.state)
        marble_state = self._get_marble_state()
        return (marble_state, board_state, self.players[self.cur_player].n)

    def get_next_state(self, action, cur_state=None):
        # Input:
        #   - An action which consists of a marble placement and a ring to remove or a capture
        #   - Optional: An arbitrary state to use instead of the current game state
        #         in the form: cur_state = (marble_state, state, player_value)
        # Returns the game state which is a tuple of:
        #   - 2D matrix of size self.board.width representing the board state
        #   - vector of length 9 containing marble counts in the order:
        #     (supply 'w', supply 'g', supply 'b', player 1 'w', ..., player 2 'b')
        #   - integer (0 or 1) giving the current player
        if cur_state is None:
            # Use the internal game state to determine the next state
            self.board.take_action(action, self.players[self.cur_player])
            self.cur_player = (self.cur_player + 1) % 2
            next_state = self.get_current_state()
        else:
            # Return the next state for an arbitrary marble supply, board and player
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            next_state = temp_game.get_next_state(action)
        return next_state

    def get_valid_actions(self, cur_state=None):
        # TODO: returns a filtering matrix that can be used to filter and renormalize the policy
        # probability distribution. Matrix shape will depend on the type of action.
        #   - for placement actions, shape is 49 x 49 x 3 
        #   - for capture actions, shape is 7 x 7 x 6
        #     - capture actions will also be broken down into capture steps with a step for each jump
        # A move is in the form:
        #   ([Place/Capture], [marble type], [marble placement], [ring removed])
        # Capturing marbles is compulsory, check if a capture is possible and return list of options
        # Available moves is every combination of:
        #   -marble type in supply (up to 3)
        #   -open rings for marble placement (up to self.rings)
        #   -edge rings for removal (up to 18 for rings=37)
        if cur_state is None:
            actions = self.board.get_valid_moves(self.players[self.cur_player])
        else:
            # Return the valid actions for an arbitrary marble supply, board and player
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            actions = temp_game.get_valid_actions()
        return actions

    def _is_game_over(self):
        # Return True if ended or False if not ended
        # Check if any player's captured marbles are enough to satisfy a win condition
        for win_con in self.win_con:
            for player in self.players:
                possible_win = True
                for marble_type, required in win_con.items():
                    if player.captured[marble_type] >= required:
                        # Meets condition for this type of marble, keep checking
                        continue
                    else:
                        possible_win = False
                        break
                if possible_win:
                    return True
        # If board has every ring covered with a marble then the last player who played is winner
        if np.all(self.board.state != 1):
            return True
        return False

    def get_game_ended(self, cur_state=None):
        # Returns 1 if first player won and -1 if second player won.
        # If no players have won then return 0.
        if cur_state is None:
            if self._is_game_over():
                # The winner is the player that made the previous action
                return self.players[(self.cur_player + 1) % 2].n
            return 0
        else:
            # Return if game is ended for an arbitrary marble supply, board and player
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

