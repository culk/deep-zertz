from collections import deque
import copy
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
    _MARBLE_TO_LAYER = {'w': 1, 'g': 2, 'b': 3}
    _LAYER_TO_MARBLE = dict((v, k) for k, v in _MARBLE_TO_LAYER.items())
    _HEX_NUMBERS = [(1, 1), (7, 3), (19, 5), (37, 7), (61, 9), (91, 11), (127, 13)]
    #              (down), (left ), (u / l ), ( up ), (right), (d /r)
    _DIRECTIONS = [(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]

    def __init__(self, rings=37, marbles=None, t=1, clone=None):
        # Return a Board object to store the board state
        #   - State is a matrix with dimensions L x H x W, H = W = Board width, L = Layers:
        #     - (# of marble types + 1) x (time history) binary to record previous board positions
        #     - 1 layer binary with a 1 at the index of a marble that needs to be used for capture
        #     - 9 layers, each same value one for each index in the supply
        #     - 1 layer of the same value for the current player
        if clone is not None:
            self.rings = clone.rings
            self.width = clone.width
            self.t = clone.t
            self._CAPTURE_LAYER = clone._CAPTURE_LAYER
            self._MARBLE_TO_SUPPLY = copy.copy(clone._MARBLE_TO_SUPPLY)
            self.state = np.copy(clone.state)
        else:
            # Determine width of board from the number of rings
            self.rings = rings
            self.width = 0
            for total, width in self._HEX_NUMBERS:
                # Currently limited to only boards that are perfect hexagons
                # TODO: implement for uneven number of rings
                if total == self.rings:
                    self.width = width
            assert self.width != 0

            # Calculate the number of layers
            # 4 * t layers for all pieces going back t steps, 9 for supply, 1 for capture, 1 for player
            # Layer:
            #   -          0 = rings (binary)
            #   -          1 = white marbles (binary)
            #   -          2 = gray marbles (binary)
            #   -          3 = black marbles (binary)
            #   -        ...   (t - 1) * 4 more layers for ring and marble state on previous time steps
            #   -      t * 4 = capturing marble (binary)
            #   -  t * 4 + 1 = supply white marbles ([0, 10])
            #   -  t * 4 + 2 = supply gray marbles ([0, 10])
            #   -  t * 4 + 3 = supply black marbles ([0, 10])
            #   -  t * 4 + 4 = player 1 white marbles ([0, 10])
            #   -  t * 4 + 5 = player 1 gray marbles ([0, 10])
            #   -  t * 4 + 6 = player 1 black marbles ([0, 10])
            #   -  t * 4 + 7 = player 2 white marbles ([0, 10])
            #   -  t * 4 + 8 = player 2 gray marbles ([0, 10])
            #   -  t * 4 + 9 = player 2 black marbles ([0, 10])
            #   - t * 4 + 10 = current player (0 or 1)
            self.t = t
            layers = 4 * self.t + 11
            self._CAPTURE_LAYER = self.t * 4
            self._MARBLE_TO_SUPPLY = {'w': self.t * 4 + 1,
                                      'g': self.t * 4 + 2,
                                      'b': self.t * 4 + 3}

            # Initialize state as 3d array
            self.state = np.zeros((layers, self.width, self.width), dtype=np.uint8)

            # Place rings
            # TODO: implement for uneven number of rings
            middle = self.width // 2
            for i in range(self.width):
                lb = max(0, i - middle)
                ub = min(self.width, middle + i + 1)
                self.state[0, lb:ub, i] = 1

            # Set the number of each type of marble available in the supply
            #   default: 6x white, 8x gray, 10x black
            if marbles is None:
                self.state[self._MARBLE_TO_SUPPLY['w']] = 6
                self.state[self._MARBLE_TO_SUPPLY['g']] = 8
                self.state[self._MARBLE_TO_SUPPLY['b']] = 10
            else:
                self.state[self._MARBLE_TO_SUPPLY['w']] = marbles['w']
                self.state[self._MARBLE_TO_SUPPLY['g']] = marbles['g']
                self.state[self._MARBLE_TO_SUPPLY['b']] = marbles['b']

    def _get_middle_ring(self, src, dst):
        # Return the (y, x) index of the ring between src and dst
        y1, x1 = src
        y2, x2 = dst
        return ((y1 + y2) / 2, (x1 + x2) / 2)

    def _get_neighbors(self, index):
        # Return a list of (y, x) indices that are adjacent to index on the board.
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
        return 0 <= y < self.width and 0 <= x < self.width

    def _get_regions(self):
        # Return a list of continuous regions on the board. A region consists of a list of indices.
        # If any index can be reached from any other index then this will return a list of length 1.
        regions = []
        not_visited = set(zip(*np.where(self.state[0] == 1)))
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
                            and self.state[0][neighbor] != 0):
                        not_visited.remove(neighbor)
                        queue.appendleft(neighbor)
            regions.append(region)
        return regions

    def get_cur_player(self):
        return self.state[self.t * 4 + 10, 0, 0]

    def _next_player(self):
        self.state[self.t * 4 + 10] = (self.state[self.t * 4 + 10] + 1) % 2
    
    def _get_cur_player_supply_layer(self, marble_type):
        # Return the layer index for the captured marble type and the current player
        # Input: captured_type = 'w', 'g', or 'b'
        supply_layer = self._MARBLE_TO_SUPPLY[marble_type]
        if self.get_cur_player() == 0:
            supply_layer += 3
        else:
            supply_layer += 6
        return supply_layer

    def _get_marble_type_at(self, index):
        y, x = index
        marble_type = self._LAYER_TO_MARBLE[np.argmax(self.state[1:4, y, x]) + 1]
        return marble_type

    def take_action(self, action, action_type):
        # Input: action is an index into the action space matrix
        #        action_type is 'PUT' or 'CAP'
        # Push back the previous t states and copy the most recent state to the top 4 layers
        self.state[0: 4*self.t] = np.concatenate([self.state[0:4], self.state[0: 4*(self.t-1)]], axis=0)

        # Modify the most recent 4 layers of the state based on the action
        if action_type == 'PUT':
            self.take_placement_action(action)
        elif action_type == 'CAP':
            self.take_capture_action(action)

    def take_placement_action(self, action):
        # Placement actions have dimension (3 x w^2 x w^2 + 1)
        # Translate the action dimensions into marble_type, put_index, and rem_index
        type_index, put_loc, rem_loc = action
        marble_type = self._LAYER_TO_MARBLE[type_index + 1]
        put_index = (put_loc // self.width, put_loc % self.width)
        if rem_loc == self.width**2:
            rem_index = None
        else:
            rem_index = (rem_loc // self.width, rem_loc % self.width)

        # Place the marble on the board
        y, x = put_index
        assert np.sum(self.state[:4, y, x]) == 1
        put_layer = self._MARBLE_TO_LAYER[marble_type] 
        self.state[put_layer][put_index] = 1

        # Remove the marble from the supply
        supply_layer = self._MARBLE_TO_SUPPLY[marble_type]
        if self.state[supply_layer, 0, 0] >= 1:
            self.state[supply_layer] -= 1
        else:
            # If supply is empty then take the marble from those the player has captured
            supply_layer = self._get_cur_player_supply_layer(marble_type)
            assert self.state[supply_layer, 0, 0] >= 1
            self.state[supply_layer] -= 1

        # Remove the ring from the board
        if rem_index is not None:
            self.state[0][rem_index] = 0
            # Check if it is possbile for the board to have been separated into regions. This is only 
            # possible if two of the empty neighbors are opposites.
            opposite_empty = False
            for neighbor in self._get_neighbors(rem_index)[:3]:
                opposite = self._get_jump_dst(neighbor, rem_index)
                if ((not self._is_inbounds(neighbor) or self.state[0][neighbor] == 0)
                        and (not self._is_inbounds(opposite) or self.state[0][opposite] == 0)):
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
                        for y, x in region:
                            if np.sum(self.state[1:4, y, x]) == 0:
                                captured = False
                                break
                        if captured:
                            # Remove all rings in the captured region and give the marbles to the player
                            for index in region:
                                y, x = index
                                if np.sum(self.state[1:4, y, x]) == 1:
                                    captured_type = self._get_marble_type_at(index)
                                supply_layer = self._get_cur_player_supply_layer(captured_type)
                                self.state[supply_layer] += 1
                                # Set the ring and marble layers all to 0
                                self.state[0:4, y, x] = 0

        # Update current player
        self._next_player()

    def take_capture_action(self, action):
        # Capture actions have dimension (6 x w x w)
        # Translate the action dimensions into src_index, marble_type, cap_index and dst_index
        direction, y, x = action
        src_index = (y, x)
        marble_type = self._get_marble_type_at(src_index)
        dy, dx = self._DIRECTIONS[direction]
        cap_index = (y + dy, x + dx)
        dst_index = self._get_jump_dst(src_index, cap_index)
        y, x = cap_index

        if np.sum(self.state[1:4, y, x]) != 1:
            import pdb; pdb.set_trace()

        # Reset the capture layer
        self.state[self._CAPTURE_LAYER] = 0

        # Remove capturing marble from src_index and place it at dst_index
        marble_layer = self._MARBLE_TO_LAYER[marble_type] 
        self.state[marble_layer][src_index] = 0
        self.state[marble_layer][dst_index] = 1

        # Give the captured marble to the current player and remove it from the board
#        y, x = cap_index
#
#        if np.sum(self.state[1:4, y, x]) != 1:
#            import pdb; pdb.set_trace()
        assert np.sum(self.state[1:4, y, x]) == 1
        captured_type = self._get_marble_type_at(cap_index)
        supply_layer = self._get_cur_player_supply_layer(captured_type)
        self.state[supply_layer] += 1
        self.state[1:4, y, x] = 0
        
        # Update the capture layer if there is a forced chain capture
        neighbors = self. _get_neighbors(dst_index)
        for neighbor in neighbors:
            y, x = neighbor
            # Check each neighbor to see if it has a marble
            if self._is_inbounds(neighbor) and np.sum(self.state[1:4, y, x]) == 1:
                next_dst = self._get_jump_dst(dst_index, neighbor)
                y, x = next_dst
                if self._is_inbounds(next_dst) and np.sum(self.state[:4, y, x]) == 1:
                    # Set the captured layer to 1 at dst_index
                    self.state[self._CAPTURE_LAYER][dst_index] = 1
                    break

        # Update current player if there are no forced chain captures
        if np.sum(self.state[self._CAPTURE_LAYER]) == 0:
            self._next_player()

    def get_valid_moves(self):
        # Return two matrices that can be used to filter the placement and capture action policy
        # distribution for actions that are valid with the current game state.
        capture = self.get_capture_moves()
        if np.any(capture):
            # no placement move is allowed if there is a valid capture move
            placement = np.zeros((3, self.width**2, self.width**2 + 1), dtype=bool)
        else:
            placement = self.get_placement_moves()
        return (placement, capture)

    def get_placement_shape(self):
        # get shape of placement moves as a tuple
        return (3, self.width**2, self.width**2 + 1)

    def get_capture_shape(self):
        # get shape of capture moves as a tuple
        return (6, self.width, self.width)

    def get_placement_moves(self):
        # Return a boolean matrix of size (3 x w^2 x w^2 + 1) with the value True at
        # every index that corresponds to a valid placement action.
        # Marble types correspond to the following indices {'w':0, 'g':1, 'b':2}
        # A ring removal value of w^2 indicates no ring is removed
        moves = np.zeros((3, self.width**2, self.width**2 + 1), dtype=bool)

        # Build list of open and removable rings for marble placement and ring removal
        open_rings = list(self._get_open_rings())
        removable_rings = list(self._get_removable_rings())

        # Get list of marble types that can be placed. If supply is empty then
        # the player must use a captured marble.
        supply_start = self._MARBLE_TO_SUPPLY['w']
        if np.all(self.state[supply_start : supply_start+3, 0, 0] == 0):
            if self.state[14, 0, 0] == 0:
                supply_start += 3
            else:
                supply_start += 6

        # Assign 1 to all indices that are valid actions
        for m, marble_count in enumerate(self.state[supply_start: supply_start+3, 0, 0]):
            if marble_count == 0:
                continue
            for put_index in open_rings:
                put = put_index[0] * self.width + put_index[1]
                for rem_index in removable_rings:
                    rem = rem_index[0] * self.width + rem_index[1]
                    if put != rem:
                        moves[m, put, rem] = True
                # If there are no removable rings then you are not required to remove one
                if not removable_rings or (len(removable_rings) == 1 and removable_rings[0] == put_index):
                    rem = self.width**2
                    moves[m, put, rem] = True
        return moves

    def get_capture_moves(self):
        # Return a boolean matrix of size (6 x w x w) with the value True at
        # every index that corresponds to a valid capture action.
        # The six directions are given by self._DIRECTIONS
        moves = np.zeros((6, self.width, self.width), dtype=bool)

        # Create list of the indices of marbles that can be used to capture
        if np.sum(self.state[self._CAPTURE_LAYER]) == 1:
            occupied_rings = zip(*np.where(self.state[self._CAPTURE_LAYER] == 1))
        else:
            occupied_rings = zip(*np.where(np.sum(self.state[1:4], axis=0) == 1))

        # Update matrix with all possible capture directions from each capturing marble
        for src_index in occupied_rings:
            src_y, src_x = src_index
            neighbors = self._get_neighbors(src_index)
            for direction, neighbor in enumerate(neighbors):
                # Check each neighbor to see if it has a marble and the jump destination is empty
                y, x = neighbor
                if self._is_inbounds(neighbor) and np.sum(self.state[1:4, y, x]) == 1:
                    dst_index = self._get_jump_dst(src_index, neighbor)
                    y, x = dst_index
                    if self._is_inbounds(dst_index) and np.sum(self.state[:4, y, x]) == 1:
                        # Set this move as a valid action in the filter matrix
                        moves[direction, src_y, src_x] = True
        return moves

    def _get_open_rings(self):
        # Return a list of indices for all of the open rings
        open_rings = zip(*np.where(np.sum(self.state[:4], axis=0) == 1))
        return open_rings

    def _is_removable(self, index):
        # Check if the ring at index is removable. A ring is removable if two of its neighbors 
        # in a row are missing and the ring itself is empty.
        y, x = index
        if np.sum(self.state[0:4, y, x]) != 1:
            return False
        neighbors = self._get_neighbors(index)
        # Add the first neighbor index to the end so that if the first and last are both empty then it still passes
        neighbors.append(neighbors[0])
        # Track the number of consecutive empty neighboring rings
        adjacent_empty = 0
        for neighbor in neighbors:
            if self._is_inbounds(neighbor) and self.state[0][neighbor] == 1:
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

    def _get_rotational_symmetries(self, state=None):
        # Rotate the board 180 degrees
        if state is None:
            rotated_state = np.copy(self.state)
        else:
            rotated_state = np.copy(state)
        rotated_state = np.rot90(np.rot90(rotated_state, axes=(1, 2)), axes=(1, 2))
        return rotated_state

    def _get_mirror_symmetries(self, state=None):
        # Flip the board while maintaining adjacency
        if state is None:
            mirror_state = np.copy(self.state)
        else:
            mirror_state = np.copy(state)
        layers = mirror_state.shape[0]
        for i in xrange(layers):
            mirror_state[i] = mirror_state[i].T
        return mirror_state

    def get_state_symmetries(self):
        # Return a list of symmetrical states by mirroring and rotating the board
        symmetries = []
        symmetries.append((0, self._get_mirror_symmetries()))
        symmetries.append((1, self._get_rotational_symmetries()))
        symmetries.append((2, self._get_rotational_symmetries(symmetries[0][1])))
        return symmetries

    def mirror_action(self, action_type, translated):
        if action_type == 'CAP':
            # swap capture direction axes
            temp = np.copy(translated)
            translated[3], translated[1] = temp[1], temp[3]
            translated[4], translated[0] = temp[0], temp[4]

            # transpose location axes
            d = translated.shape[0]
            for i in xrange(d):
                translated[i] = translated[i].T

        elif action_type == 'PUT':
            temp = np.copy(translated)
            _, put, rem = translated.shape
            for p in xrange(put):
                # Translate the put index
                put_y, put_x = p / self.width, p % self.width
                new_p = put_x * self.width + put_y
                for r in xrange(rem - 1):
                    # Translate the rem index
                    rem_y, rem_x = r / self.width, r % self.width
                    new_r = rem_x * self.width + rem_y
                    translated[:, new_p, new_r] = temp[:, p, r]

                # The last rem index is the same
                translated[:, new_p, rem - 1] = translated[:, new_p, rem - 1]

        return translated

    def rotate_action(self, action_type, translated):

        mid = self.width / 2 # board width must be odd
        if action_type == 'CAP':
            # swap capture direction axes
            temp = np.copy(translated)
            translated[3], translated[0] = temp[0], temp[3]
            translated[4], translated[1] = temp[1], temp[4]
            translated[5], translated[2] = temp[2], temp[5]

            # rotate location axes
            temp = np.copy(translated)
            _, y, x = temp.shape
            for i in xrange(y):
                new_i = mid + (mid - i)
                for j in xrange(x):
                    new_j = mid + (mid - j)
                    translated[:, new_i, new_j] = temp[:, i, j]

        if action_type == 'PUT':
            temp = np.copy(translated)
            _, put, rem = translated.shape
            for p in xrange(put):
                # Translate the put index
                put_y, put_x = p / self.width, p % self.width
                put_y = mid + (mid - put_y)
                put_x = mid + (mid - put_x)
                new_p = put_y * self.width + put_x
                for r in xrange(rem - 1):
                    # Translate the rem index
                    rem_y, rem_x = r / self.width, r % self.width
                    rem_y = mid + (mid - rem_y)
                    rem_x = mid + (mid - rem_x)
                    new_r = rem_y * self.width + rem_x
                    translated[:, new_p, new_r] = temp[:, p, r]

                # The last rem index is the same
                translated[:, new_p, rem - 1] = translated[:, new_p, rem - 1]

        return translated

    def str_to_index(self, index_str):
        # Given a string like 'A1' return an index (y, x) based on the board shape
        letter, number = index_str

        # Calculate x
        letter = letter.upper()
        x = ord(letter) - 65 # ord('A') == 65

        # Calculate y
        mid = self.width // 2
        number = int(number)
        offset = max(mid - x, 0)
        y = (self.width) - (number + offset)

        return y, x

    def index_to_str(self, index):
        # Given an index (y, x) return a string like 'A1' based on the board shape
        y, x = index

        # Calculate letter
        letter = chr(x + 65) # chr(65) == 'A'

        # Calculate number
        mid = self.width // 2
        offset = max(mid - x, 0)
        number = str((self.width) - (y + offset))
        
        return letter + number

