import sys
sys.path.append('..')
from collections import deque
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
    #   ((capture, marble_type, start_ring), (marble_type, dst_ring), (marble_type, dst_ring), etc.)
    #   - here, each marble after the first is a marble being captured
    #   - the final location of the marble doing the capturing is the final dst_ring
    #   - the marble being captured is the one on the ring between the start_ring and dst_ring
    #   i.e. (('CAP', 'b', 'A3'), ('b', 'C5'))
    #        or (('CAP', 'g', 'D6'), ('w', 'D4'), ('w', 'B2'))
    #     - this action uses the marble at D6 to capture the marbles at D5 and C3 before ending at B2
    _ACTION_VERBS = ['PUT', 'REM', 'CAP']
    # For mapping number of rings to board width
    _HEX_NUMBERS = [(1, 1), (7, 3), (19, 5), (37, 7), (61, 9), (91, 11), (127, 13)]
    _MARBLE_TO_INT = {'b': 4, 'g': 3, 'w': 2}
    _INT_TO_MARBLE = dict((v, k) for k, v in _MARBLE_TO_INT.items())
    #              (down), (left ), (u / l ), ( up ), (right), (d /r)
    _DIRECTIONS = [(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]

    def __init__(self, rings=37, marbles=None, clone=None):
        if clone is not None:
            self.rings = clone.rings
            self.board_width = clone.board_width
            self.board_state = np.copy(clone.board_state)
            self.supply = clone.supply
        else:
            self.rings = rings

            # Determine width of board from the number of rings
            self.board_width = 0
            for total, width in self._HEX_NUMBERS:
                # Currently limited to only boards that are perfect hexagons
                # TODO: implement for uneven number of rings
                if total == self.rings:
                    self.board_width = width
            assert self.board_width != 0

            # Initialize board as 2d array and fill with rings, only perfect hexagons are implemented for now
            # TODO: implement for uneven number of rings
            self.board_state = np.zeros((self.board_width, self.board_width))
            middle = self.board_width // 2
            for i in range(self.board_width):
                lb = max(0, i - middle)
                ub = min(self.board_width, middle + i + 1)
                self.board_state[lb:ub, i] = 1

            # The number of marbles that are available in the supply
            #   default: 6x white, 8x gray, 10x black
            if marbles is None:
                self.supply = {'w': 6, 'g': 8, 'b': 10}
            else:
                self.supply = marbles

    def _get_middle_ring(self, src, dst):
        # Return the index of the ring between src and dst
        x1, y1 = src
        x2, y2 = dst
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _get_neighbors(self, index):
        # Return a list of indices that are adjacent to index on the board.
        # The neighboring index may not be within the board space so it must be checked 
        # that it is inbounds (see _is_inbounds).
        y, x = index
        neighbors = [(y + dy, x + dx) for dy, dx in self._DIRECTIONS]
        return neighbors

    def _is_adjacent(self, l1, l2):
        # Return True if l1 and l2 are adjacent to each other on the hexagonal board
        return l2 in self._get_neighbors(l1)

    def _get_jump_dst(self, start, cap):
        # Return the landing index after capturing the marble at cap from start.
        # The landing index may not be within the board space so it must be checked 
        # that it is inbounds (see _is_inbounds).
        sy, sx = start
        cy, cx = cap
        dy = (cy - sy) * 2
        dx = (cx - sx) * 2
        return (sy + dy, sx + dx)

    def _is_inbounds(self, index):
        # Return True if the index is in bounds for board's width
        y, x = index
        return 0 <= y < self.board_width and 0 <= x < self.board_width


    def _get_regions(self):
        # Return a list of continuous regions on the board. A region consists of a list of indices.
        # If any index can be reached from any other index then this will return a list of length 1.
        regions = []
        not_visited = set(zip(*np.where(self.board_state >= 1)))
        while not_visited:
            # While there are indices that have not been added to a region start a new empty region
            region = []
            queue = deque()
            queue.appendleft(not_visited.pop())
            # Add all indices to the region that can be reached from the starting index
            while queue:
                index = queue.pop()
                region.append(index)
                # Add all neighbors to the queue and mark visited to add them to the same region
                for neighbor in self._get_neighbors(index):
                    if (neighbor in not_visited
                            and self._is_inbounds(neighbor)
                            and self.board_state[neighbor] != 0):
                        not_visited.remove(neighbor)
                        queue.appendleft(neighbor)
            regions.append(region)
        return regions

    def take_action(self, action, player):
        # Modify the game state based on the action
        if action[0][0] == 'PUT':
            _, marble_type, put_index = action[0]
            _, rem_index = action[1]
            # Place the marble on the board and remove it from the supply
            self.board_state[put_index] = self._MARBLE_TO_INT[marble_type]
            self.supply[marble_type] -= 1

            # Remove the ring from the board
            if rem_index is not None:
                self.board_state[rem_index] = 0
                # Check if it is possbile for the board to have been separated into regions. This is only 
                # possible if two of the empty neighbors are opposites.
                opposite_empty = False
                for neighbor in self._get_neighbors(rem_index)[:3]:
                    opposite = self._get_jump_dst(neighbor, rem_index)
                    if ((not self._is_inbounds(neighbor) or self.board_state[neighbor] == 0)
                            and (not self._is_inbounds(opposite) or self.board_state[opposite] == 0)):
                        opposite_empty = True
                        break
                if opposite_empty:
                    # Get list of regions
                    regions = self._get_regions()
                    # If the board has been separated into multiple regions then check if any are captured
                    if len(regions) > 1:
                        for region in regions:
                            captured = True
                            # A region is captured if every ring in the region is occupied by a marble
                            for index in region:
                                if self.board_state[index] == 1:
                                    captured = False
                                    break
                            if captured:
                                # Remove all rings in the captured region and give the marbles to the player
                                for index in region:
                                    captured_type = self._INT_TO_MARBLE[self.board_state[index]]
                                    player.captured[captured_type] += 1
                                    self.board_state[index] = 0
        elif action[0][0] == 'CAP':
            # Remove the marble doing the capturing from its origin ring
            _, marble_type, src_index = action[0]
            self.board_state[src_index] = 1
            for captured_type, dst_index in action[1:]:
                # Give the captured marble to the player
                player.captured[captured_type] += 1
                captured_index = self._get_middle_ring(src_index, dst_index)
                # Remove the captured marble from the board and update the capturing marbles location
                self.board_state[captured_index] = 1
                src_index = dst_index
            # Place the capturing marble in its final destination ring
            self.board_state[src_index] = self._MARBLE_TO_INT[marble_type]

    def get_valid_moves(self):
        # Return a list of moves that are valid with the current game state.
        # The current player has no impact on the list of valid moves.
        moves = self._get_capture_moves()
        if moves:
            return moves
        # If there are no forced captures then build list of placement moves
        for marble_type in self.supply.keys():
            if self.supply[marble_type] == 0:
                continue
            open_rings = self._get_open_rings()
            removable_rings = self._get_removable_rings()
            for put in open_rings:
                for rem in removable_rings:
                    if put != rem:
                        moves.append((('PUT', marble_type, put), ('REM', rem)))
                # If there are no removable rings then you are not required to remove one
                if not removable_rings or (len(removable_rings) == 1 and removable_rings[0] == put):
                    moves.append((('PUT', marble_type, put), ('REM', None)))
        return moves

    def _get_capture_moves(self):
        # Return a list of all possible capture moves in the form of tuples
        def build_capture_chain(start, visited, marbles):
            # Recursively build the capture chain for all capture options.
            # Returns a list of lists for all possible capture branches.
            moves = []
            for i, index in enumerate(marbles):
                if i not in visited and self._is_adjacent(start, index):
                    # if index is not visited and adjacent then get the landing index after the capture
                    dst = self._get_jump_dst(start, index)
                    if self._is_inbounds(dst) and self.board_state[dst] == 1:
                        # if the landing index is in bounds and on an empty ring
                        marble_type = self._INT_TO_MARBLE[self.board_state[index]]
                        captured = [(marble_type, dst)]
                        chains = build_capture_chain(dst, visited + [i], marbles)
                        if chains:
                            moves += [captured + chain for chain in chains]
                        else:
                            moves += [captured]
            return moves

        moves = []
        # Create list of the indices of all marbles
        occupied_rings = zip(*np.where(self.board_state > 1))
        for i, index in enumerate(occupied_rings):
            marble_type = self._INT_TO_MARBLE[self.board_state[index]]
            beginning = [('CAP', marble_type, index)]
            chains = build_capture_chain(index, [i], occupied_rings)
            for chain in chains:
                moves.append(tuple(beginning + chain))
        return moves

    def _get_open_rings(self):
        # Return a list of indices for all of the open rings
        open_rings = zip(*np.where(self.board_state == 1))
        return open_rings

    def _is_removable(self, index):
        # Check if the ring at index is removable. A ring is removable if two of its neighbors 
        # in a row are missing and the ring itself is empty.
        if self.board_state[index] != 1:
            return False
        neighbors = self._get_neighbors(index)
        # Add the first neighbor index to the end so that if the first and last are both empty then it still passes
        neighbors.append(neighbors[0])
        # Track the number of consecutive empty neighboring rings
        adjacent_empty = 0
        for neighbor in neighbors:
            if self._is_inbounds(neighbor) and self.board_state[neighbor] != 0:
                # If the neighbor index is in bounds and not removed then reset the empty counter
                adjacent_empty = 0
            else:
                adjacent_empty += 1
                if adjacent_empty >= 2:
                    return True
        return False

    def _get_removable_rings(self):
        # Return a list of indices to rings that can be removed
        removable = [index for index in self._get_open_rings() if self._is_removable(index)]
        return removable

    def _get_rotational_symmetries(self):
        pass

    def _get_mirrior_symmetries(self):
        pass

    def get_symmetries(self):
        # Return a list of symmetrical board_states by mirroring and rotating the board
        pass

