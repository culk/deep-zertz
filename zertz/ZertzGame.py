from .ZertzLogic import Board
from .ZertzPlayer import Player


# For full rules: http://www.gipf.com/zertz/rules/rules.html
# Class interface inspired by https://github.com/suragnair/alpha-zero-general

class ZertzGame():
    def __init__(self, rings=37, marbles=None, win_con=None):
        # the size of the game board
        #   default: 37 rings (approximately 7x7 hex)
        self.initial_rings = rings
        self.board = Board(self.initial_rings)
        self.players = [Player(self, i) for i in [1, -1]]
        self.cur_player = 0

        # the number of marbles that are available in the supply
        #   default: 6x white, 8x gray, 10x black
        if marbles is None:
            self.supply = {'w': 6, 'g': 8, 'b': 10}
        else:
            self.supply = marbles

        # the win conditions (amount of each marble needed)
        #   default:
        #     -3 marbles of each color
        #     -4 white marbles
        #     -5 gray marbles
        #     -6 black marbles
        if win_con is None:
            self.win_con = [{'w': 3, 'g': 3, 'b': 3},
                            {'w': 4}, {'g': 5}, {'b': 6}]
        else:
            self.win_con = win_con

    def _get_marble_state():
        type_to_i = {'w': 0, 'g': 1, 'b': 2}
        state = np.zeros(9)
        for marble_type in self.supply:
            state[type_to_i[marble_type]] = self.supply[marble_type]
        for marble_type in self.players[0].captured:
            state[type_to_i[marble_type] + 3] = self.players[0].captured[marble_type]
        for marble_type in self.players[1].captured:
            state[type_to_i[marble_type] + 6] = self.players[1].captured[marble_type]
        return state

    def get_next_state(self, action):
        # Input: an action which consists of a marble placement and a ring to remove or a capture
        # returns the game state which is a tuple of:
        #   - 2D matrix of size self.board.board_width representing the board state
        #   - vector of length 9 containing marble counts in the order:
        #     (supply 'w', supply 'g', supply 'b', player 1 'w', ..., player 2 'b')
        #   - integer (0 or 1) giving the current player
        self.board.take_action(action, self.players[self.cur_player])
        board_state = np.copy(self.board.board_state)
        marble_state = self._get_marble_state()
        self.cur_player = (self.cur_player + 1) % 2
        return (marble_state, board_state, self.cur_player)

    def get_valid_actions(self):
        # A move is in the form:
        #   ([Place/Capture], [marble type], [marble placement], [ring removed])
        # Capturing marbles is compulsory, check if a capture is possible and return list of options
        # Available moves is every combination of:
        #   -marble type in supply (up to 3)
        #   -open rings for marble placement (up to self.rings)
        #   -edge rings for removal (up to 18 for rings=37)
        actions = self.board.get_legal_moves()
        return actions

    def _is_game_over(self):
        # return True if ended or False if not ended
        # check if the player's captured marbles is enough to satisfy a win condition
        for win_con in self.win_con:
            possible_win = True
            for marble_type, required in win_con.items():
                if player.captured[marble_type] >= required:
                    # meets condition for this type of marble, keep checking
                    continue
                else:
                    possible_win = False
                    break
            if possible_win:
                return True
        # TODO: implement other checks
        # check if the board is in a state where the game cannot progress
        #   (not sure if possible, could be for different sized board states)
        #   - if board has every ring covered with a marble then the last player who played is winner
        #   - if both players start repeating the same sequence of movies the game is a tie (how to check)
        return False

    def get_game_ended(self):
        # returns 1 if 1st player won and -1 if second player won
        # if no players have won then return 0
        if self._is_game_over():
            return self.players[self.cur_player].n
        return 0

    def _get_rotational_symmetries(self):
        pass

    def _get_mirrior_symmetries(self):
        # flip the board and then generate rotational symmetries
        pass

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
        pass

