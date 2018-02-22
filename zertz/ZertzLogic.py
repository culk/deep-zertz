import numpy as np


class Board():
    # The zertz board is a hexagon and looks like this:
    #   Each location is a ring
    #   Each ring is adjacent to the rings above, below, left, right, up/left, and up/right
    # 
    #           D7
    #        C6 D6 E6
    #     B5 C5 D5 E5 F5
    #  A4 B4 C4 D4 E4 F4 G4
    #  A3 B3 C3 D3 E3 F3 G3
    #  A2 B2 C2 D2 E2 F2 G2
    #  A1 B1 C1 D1 E1 F1 G1
    #
    # The value of each space corresponds to:
    #   - 0 = no ring
    #   - 1 = ring
    #   - 2 = white marble
    #   - 3 = gray marble
    #   - 4 = black marble
    #
    # A placement action is a tuple in the form:
    #   ((placement, marble_type, ring), (remove, ring))
    #   i.e. (('PUT', 'b', 'E6'), ('REM', 'A2'))
    #     - this puts a marble at E6 and removes the ring A2
    #
    # Capture action is a tuple in the form:
    #   ((capture, marble_type, start_ring), (marble_type, dest_ring), (marble_type, dest_ring), etc.)
    #   - here, each marble after the first is a marble being captured
    #   - the final location of the marble doing the capturing is the final dest_ring
    #   - the marble being captured is the one on the ring between the start_ring and dest_ring
    #   i.e. (('CAP', 'b', 'A3'), ('b', 'C5'))
    #        or (('CAP', 'g', 'D6'), ('w', 'D4'), ('w', 'B2'))
    #     - this action uses the marble at D6 to capture the marbles at D5 and C3 before ending at B2
    _ACTION_VERBS = ['PUT', 'REM', 'CAP']
    _HEX_NUMBERS = [(1, 1), (7, 3), (19, 5), (37, 7), (61, 9), (91, 11), (127, 13)]
    _MARBLE_VALUES = {'b': 4, 'g': 3, 'w': 2}

    def __init__(self, rings=37):
        self.rings = rings

        # determine width of board
        self.board_width = 0
        for total, width in self._HEX_NUMBERS:
            # limiting to only boards that are perfect hexagons for now
            # TODO: implement for uneven number of rings
            if total == self.rings:
                self.board_width = width
        assert self.board_width != 0

        # initialize board as 2d array and fill with rings, only perfect hexagons are implemented for now
        # TODO: implement for uneven number of rings
        self.board_state = np.zeros((self.board_width, self.board_width))
        for i in range(self.board_width):
            # right now this has origin at top left (like numpy) but my diagram has the origin in bottom left
            j = np.abs(i - self.board_width // 2)
            self.board_state[:self.board_width - j, i] = 1

    def _get_middle_ring(self, src, dst):
        x1, y1 = src
        x2, y2 = dst
        diff = (x2 - x1, y2 - y1)
        dx, dy = diff
        return (x1 + dx / 2, y1 + dy / 2)

    def take_action(self, action, player):
        # modify the game state based on the action
        if action[0][0] == 'PUT':
            _, marble_type, put_index = action[0]
            _, rem_index = action[1]
            self.board_state[put_index] = _MARBLE_VALUES[marble_type]
            self.board_state[rem_index] = 0
        elif action[0][0] == 'CAP':
            # remove marble from its origin
            _, marble_type, src_index = action[0]
            self.board_state[src_index] = 1
            # iterate over the captured marbles
            for captured_type, dest_index in action[1:]:
                # give the captured marble to the player
                player.captured[captured_type] += 1
                captured_index = self._get_middle_ring(src_index, dest_index)
                # remove the captured marble from the board and move the capturing marble
                self.board_state[captured_index] = 1
                src_index = dst_index
            self.board_state[src_index] = _MARBLE_VALUES[marble_type]

    def get_legal_moves(self):
        # return a list of legal moves
        # check if there is a compulsory capture
        moves = self._get_capture_moves()
        if len(moves) != 0:
            return moves
        # build list of elligible moves
        for marble_type in self.supply.keys():
            if self.supply[marble_type] == 0:
                continue
            open_rings = self._get_open_rings()
            removable_rings = self._get_removable_rings()
            for put in open_rings:
                for rem in removable_rings:
                    if put == rem:
                        continue
                    moves.append((('PUT', marble_type, put), ('REM', rem)))
        return moves

    def _get_capture_moves(self):
        # check each square to see if a capture is possible
        # TODO: implement
        # how to implement this efficiently? maintain list of marbles currently on the board and iterate over those? or iterate over all spaces
        moves = []
        return moves

    def _get_open_rings(self):
        # return a list of indices to open rings
        return zip(*np.where(self.board_state == 1))

    def _get_removable_rings(self):
        # return a list of indices to rings that can be removed
        # TODO: implement
        # iterate over all open rings? efficient way to check if a ring is removable?
        pass

