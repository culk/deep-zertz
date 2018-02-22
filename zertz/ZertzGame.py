# For full rules: http://www.gipf.com/zertz/rules/rules.html

# Class interface inspired by https://github.com/suragnair/alpha-zero-general

class ZertzGame():
    def __init__(self, rings=37, marbles=None, win_con=None):
        # the size of the game board
        #   default: 37 rings (approximately 7x7 hex)
        self.initial_rings = rings

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

    def get_next_state(self, board, player, action):
        # an action consists of a marble placement and a ring to remove
        pass

    def get_valid_moves(self, board, player):
        # A move is in the form:
        #   ([Place/Capture], [marble type], [marble placement], [ring removed])
        # Capturing marbles is compulsory, check if a capture is possible and return list of options
        # Available moves is every combination of:
        #   -marble type in supply (up to 3)
        #   -open rings for marble placement (up to self.rings)
        #   -edge rings for removal (up to 18 for rings=37)
        moves = []
        return moves

    def get_game_end(self, board, player):
        # return True if ended or False if not ended
        # check if the player's captured marbles is enough to satisfy a win condition
        for win_con in self.win_con:
            possible_win = True
            for marble, required in win_con.items():
                if player.captured[marble] >= required:
                    # meets condition for this type of marble, keep checking
                    continue
                else:
                    possible_win = False
                    break
            if possible_win:
                return True
                    
        # check if the board is in a state where the game cannot progress
        #   (not sure if possible, could be for different sized board states)
        #   - if board has every ring covered with a marble then the last player who played is winner
        #   - if both players start repeating the same sequence of movies the game is a tie (how to check)
        return False
