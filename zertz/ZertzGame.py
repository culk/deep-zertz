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


    def get_next_state(self, action):
        # an action consists of a marble placement and a ring to remove or a capture
        self.board.take_action(action, self.players[self.cur_player])
        self.cur_player = (self.cur_player + 1) % 2
        # TODO: what should actually be returned?
        # should to also include the players captured list because that could impact decisions
        return (self.supply, self.board.board_state, self.cur_player)

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
        else:
            return 0

    def get_symmetries(self):
        # TODO: implement this later for training efficiency improvements
        pass

