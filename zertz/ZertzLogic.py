import numpy as np


class Board():
    # The zertz board is a hexagon and looks like this:
    #   A 2D array where each location is a ring
    #   Each ring is adjacent to the rings below, left, above/left, above, right, and down/right
    # 
    #  A4 B5 C6 D7
    #  A3 B4 C5 D6 E6
    #  A2 B3 C4 D5 E5 F5
    #  A1 B2 C3 D4 E4 F4 G4
    #     B1 C2 D3 E3 F3 G3
    #        C1 D2 E2 F2 G2
    #           D1 E1 F1 G1
    #
    # The value of each location corresponds to:
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
    # for mapping number of rings to board width
    _HEX_NUMBERS = [(1, 1), (7, 3), (19, 5), (37, 7), (61, 9), (91, 11), (127, 13)]
    _MARBLE_VALUES = {'b': 4, 'g': 3, 'w': 2}
    #              (down), (left ), (u / l ), ( up ), (right), (d /r)
    _DIRECTIONS = [(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]

    def __init__(self, rings=37, clone=None):
        if clone is not None:
            self.rings = clone.rings
            self.board_width = clone.board_width
            self.board_state = np.copy(clone.board_state)
        else:
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
            middle = self.board_width // 2
            for i in range(self.board_width):
                lb = min(0, i - middle)
                ub = min(5, middle + i + 1)
                self.board_state[lb:ub, i] = 1

    def _get_middle_ring(self, src, dst):
        # returns the index of the ring between src and dst
        x1, y1 = src
        x2, y2 = dst
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def take_action(self, action, player):
        # modify the game state based on the action
        if action[0][0] == 'PUT':
            _, marble_type, put_index = action[0]
            _, rem_index = action[1]
            self.board_state[put_index] = _MARBLE_VALUES[marble_type]
            # TODO: technically it is possible for a placement action to not have any valid 
            # rings to remove, in that case no rings are removed
            self.board_state[rem_index] = 0
            # TODO: check if removing the ring would separate the board
            #       capture separated marbles and update the board state
            # Idea: if there are at least three empty spaces bordering the removed space and 
            # they are not all next to each other then it separates
            # Still have to find the section that is smaller (less rings) to iterate over 
            # those indices and capture the marbles
            # To further complicate this, an isolated section of the board is only removed 
            # and captured if all of its rings have marbles
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

    def _get_neighbors(self, index):
        # return a list of indices that are adjacent to index on the board
        # the neighboring index may not be within the board space so it must be checked
        y, x = index
        neighbors = [(y + dy, x + dx) for dy, dx in self._DIRECTIONS]
        return neighbors

    def _is_adjacent(self, l1, l2):
        # return True if l1 and l2 are adjacent to each other on the hexagonal board
        return l2 in self.get_neighbors(l1)

    def _get_jump_dest(self, start, cap):
        # return the landing index after capturing the marble at cap from start
        # the landing index may not be within the board space so it must be checked
        sy, sx = start
        cy, cx = cap
        dy = cy - sy * 2
        dx = cx - sx * 2
        return (sy + dy, sx + dx)

    def _get_capture_moves(self):
        # check each square to see if a capture is possible
        def build_capture_chain(start, visited, occupied):
            # TODO: implement
            # need recursive function to build capture paths that can then be turned into actions
            # BAD CODE BELOW
            moves = []
            for i, index in enumerate(occupied):
                if index == start: continue
                if self._is_adjacent(index, start):
                    marble_type = self.board_state[index]
                    dest = self._get_jump_dest(start, index)
                    moves.append([[marble_type, dest], build_capture_chain(dest, visited + [i], occupied)])
                    # then would need to unravel the moves...
                    # this is messy and needs to be written better
            return moves
            # BAD CODE ABOVE

        # TODO: implement
        # how to implement this efficiently? maintain list of marbles currently on the board and iterate over those? or iterate over all spaces
        moves = []
        occupied_rings = zip(*np.where(self.board_state > 1))
        for i, index in enumerate(occupied_rings):
            move = build_capture_chain(index, [i], occupied_rings)
            # convert move to a tuple and label it 'CAP'
            moves.append(move)
        return moves

    def _get_open_rings(self):
        # return a list of indices to open rings
        open_rings = zip(*np.where(self.board_state == 1))
        return open_rings

    def _is_removable(self, index):
        # check if the ring at index is removable
        # a ring is removable if two of its neighbors in a row are missing and the ring itself is empty
        # check if the ring is empty
        if self.board_state[index] != 1:
            return False
        # build a list of the neighboring indices
        neighbors = self._get_neighbors(index)
        # add the first neighbor index to the end so that if the first and last are both empty then it still passes
        neighbors.append(neighbors[0])
        # track the number of consecutive empty neighboring rings
        adjacent_empty = 0
        for ny, nx in neighbors:
            if (0 <= ny < self.board_width
                    and 0 <= nx < self.board_width
                    and self.board_state[ny, nx] != 0):
                # if the neighbor index is in bounds and not removed then reset the empty counter
                adjacent_empty = 0
            else:
                adjacent_empty += 1
                if adjacent_empty >= 2:
                    return True
        return False

    def _get_removable_rings(self):
        # return a list of indices to rings that can be removed
        # TODO: can this be improved? Current running time is O(6*n) where n is number of open rings
        removable = [index for index in self._get_open_rings() if self._is_removable(index)]
        return removable

    def _get_rotational_symmetries(self):
        pass

    def _get_mirrior_symmetries(self):
        pass

    def get_symmetries(self):
        pass

