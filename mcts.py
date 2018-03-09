import numpy as np

class Node(object):
    """ A class that represents a node in MC search tree. 
    A should include the following attributes:
        N: the number of times this node has been selected from its parents
        Q: the mean value of this staet
        P: the prior probability of selecting this node (ie. taking this action)
    """
    def __init__(self, parent, P, cur_player):
        self.N = 0
        self.Q = 0.0
        self.P = P
        self.child = {}
        self.action_type = None # will be assigned upon expanding
        self.cur_player = cur_player
        self.parent = parent

    def update(self, predicted_v):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.        
        """
        self.Q = (self.Q * self.N + predicted_v) / (self.N + 1)
        self.N += 1

    def recurse_update(self, predicted_v):
        """Call by MCTS to recursively propagate predicted_v up to ancestor
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            if self.cur_player == self.parent.cur_player:
                self.parent.recurse_update(predicted_v)
            else:
                self.parent.recurse_update(-predicted_v)
        self.update(predicted_v)

    def expand(self, action_type, predicted_p, player_change):
        """Expand the search tree by attaching child nodes to current state
        Args:
            player_change: an int to represent either the child nodes results in player change
                            1 if player remains the same and -1 if player changes
        """
        assert abs(np.sum(predicted_p) - 1) < .0001
        self.action_type = action_type
        predicted_p = predicted_p.squeeze()
        z, y, x = predicted_p.shape
        for i in xrange(z):
            for j in xrange(y):
                for k in xrange(x):
                    action = (i, j, k)
                    prob = predicted_p[action]
                    if action not in self.child and prob > 0:
                        self.child[action] = Node(self, prob, player_change*self.cur_player)

    def get_action(self, c_puct):
        """Gets best action based on current estimate of Q and U
        """
        max_u = float('-inf')
        best_a = None
        next_node = None
        for action, node in self.child.items():
            u = node.Q + c_puct * node.P * np.sqrt(self.N) / (1 + node.N)
            if u > max_u:
                max_u = u
                best_a = action
                next_node = node

        # if self.child[best_a].P == 0:
        #     import pdb; pdb.set_trace()
        
        assert(self.action_type is not None)
        return self.action_type, best_a, next_node

    def is_leaf(self):
        return self.child == {}


class MCTS(object):
    def __init__(self, game, nnet, c_puct, num_sim):
        """
        Arguments:
            policy_fn: is a function to that returns (p, v) tuple given a state. 
                Called when reached a leaf node in tree. In the case of AlphaZero, it is a NN.
            num_sim: number of simulations to run before selecting move
        """
        self.game = game
        self.nnet = nnet
        self.c_puct = c_puct
        self.num_sim = num_sim
        self.root = Node(None, 1.0, 1)
        #self.node_dict = {tuple(self.game.board.state.flatten()): self.root}

    def reset(self):
        self.root = Node(None, 1.0, 1)

    def simulate(self, board_state):
        """
        Perform one simulation of MCTS. Recursively called until a leaf is found.
        Then uses policy_fn to make prediction of (p,v). This value is propogated up the 
        path.
        Args:
            state is a tuple of (board_state, player)
        """

        #node = self.node_dict[tuple(board_state.flatten())]
        #self.root = Node(None, 1.0, 1)
        node = self.root
        player_change = 1
        while True:
            if node.is_leaf():
                # print('break')
                break
            action_type, best_a, node = node.get_action(self.c_puct)
            # print(action_type, best_a, board_state[-1, 0, 0])
            # print(np.sum(board_state[:4], axis=0))
            #import pdb; pdb.set_trace()
            next_board_state, _ = self.game.get_next_state(best_a, action_type, board_state)
            #self.node_dict[tuple(next_board_state.flatten())] = node
            player_change = 1 if next_board_state[-1, 0, 0] == board_state[-1, 0, 0] else -1
            board_state = next_board_state
        
        valid_placement, valid_capture = self.game.get_valid_actions(board_state)
        if np.any(valid_placement == True):
            action_filter = 1
        else:
            action_filter = 0
        p_placement, p_capture, v = self.nnet.predict(board_state, action_filter)

        winner = self.game.get_game_ended(board_state)
        if ((np.sum(valid_placement * p_placement) == 0) and
                (np.sum(valid_capture * p_capture) == 0) and winner == 0):
            import pdb; pdb.set_trace()
        if winner == 0:
            # No player has won
            valid_placement, valid_capture = self.game.get_valid_actions(board_state)
            valid_placement = np.squeeze(valid_placement)
            valid_capture = np.squeeze(valid_capture)
            if np.any(valid_placement == True):
                p_placement = np.multiply(p_placement, valid_placement)
                if np.sum(p_placement) == 0:
                    p_placement = valid_placement.astype(np.float32)
                p_placement /= np.sum(p_placement)
                node.expand('PUT', p_placement, player_change)
            else:
                p_capture = np.multiply(p_capture, valid_capture)
                if np.sum(p_capture) == 0:
                    p_capture = valid_capture.astype(np.float32)
                p_capture /= np.sum(p_capture)
                # if p_capture[0, 4,2,2] != 0:
                #     import pdb; pdb.set_trace()
                node.expand('CAP', p_capture, player_change)

        else:
            v = winner

        node.recurse_update(-v)


    def get_action_prob(self, state, temp):

        for _ in range(self.num_sim):
            state_copy = np.copy(state)
            self.simulate(state_copy)

        action_type = self.root.action_type
        Nas = [(action, node.N) for action, node in self.root.child.items()]
        actions, count = zip(*Nas)
        if temp==0:
            probs = np.zeros_like(count)
            probs[np.argmax(count)] = 1
            return self.restore_action_matrix(action_type, actions, probs)

        counts = [x**(1./temp) for x in count]
        probs = [x/float(sum(counts)) for x in counts]

        '''
        if action_type == 'PUT':
            probs = np.array(probs).reshape(self.game.get_placement_action_shape())
        else:
            probs = np.array(probs).reshape(self.game.get_capture_action_shape())
                    '''
         
        return self.restore_action_matrix(action_type, actions, probs)

    def restore_action_matrix(self, action_type, actions, probs):
        if action_type == 'PUT':
            probs_full = np.zeros(self.game.get_placement_action_shape())
        else:
            probs_full = np.zeros(self.game.get_capture_action_shape())

        for ind, p in zip(actions, probs):
            probs_full[ind] = p

        x, y, z = probs_full.shape
        actions_full = np.array([(i, j, k) for i in range(x) for j in range(y) for k in range(z)])
        probs_full = list(probs_full.flatten())

        assert(np.sum(probs_full) == 1)

        return action_type, actions_full, probs_full


