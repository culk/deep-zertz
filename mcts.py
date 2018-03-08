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
        self.Q = 0
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
        self.Q = (self._Q * self.N +predicted_v)/ (self.N +1)
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
        self.action_type = action_type
        for action, prob in predicted_p:
            if action not in self.child:
                self.child[action] = Node(self, prob, player_change*self.cur_player)

    def get_action(self, c_puct):
        """Gets best action based on current estimate of Q and U
        """
        max_u = float('-inf')
        best_a = None
        for action, node in self.child.items():
            u = node.Q + c_puct * node.P * np.sqrt(self.N) / (1 + node.N)
            if u > max_u:
                max_u = u
                best_a = action

        assert(self.action_type is not None)
        return self.action_type, best_a

    def is_leaf(self):
        return self.child == {}


class MCTS(object):
    def __init__(self, game, policy_fn, c_puct, num_sim):
        """
        Arguments:
            policy_fn: is a function to that returns (p, v) tuple given a state. 
                Called when reached a leaf node in tree. In the case of AlphaZero, it is a NN.
            num_sim: number of simulations to run before selecting move
        """
        self.game = game
        self.policy_fn = policy_fn
        self.c_puct = c_puct
        self.num_sim = num_sim
        self.root = Node(None, 1.0, 1)

    def simulate(self, state):
        """
        Perform one simulation of MCTS. Recursively called until a leaf is found.
        Then uses policy_fn to make prediction of (p,v). This value is propogated up the 
        path.
        Args:
            state is a tuple of (board_state, player)
        NOTE: board should be a deep copy of original board
        """

        node = self.root
        while True:
            if node.is_leaf():
                break
            action_type, best_a = node.get_action(self.c_puct)
            board_state, _ = self.game.get_next_state(self, best_a, action_type, state)
            player_change = 1 if board_state[-1,0,0] == state[-1,0,0] else -1

        p_placement, p_capture, v = self.policy_fn(board_state)

        winner = self.game.get_game_ended(board_state)
        if winner == 0:
            # No player has won
            valid_placement, valid_capture = self.game.get_valid_actions(board_state)
            if valid_placement is not None:
                p_placement = np.multiply(p_placement, valid_placement)
                p_placement /= np.sum(p_placement)
                node.expand('PUT', p_placement, player_change)
            else:
                p_capture = np.multiply(p_capture, valid_capture)
                p_capture /= np.sum(p_capture)
                node.expand('CAP', p_capture, player_change)
        else:
            v = winner

        node.recurse_update(-v)


    def get_action_prob(self, state, temp):

        for _ in range(self.num_sim):
            state_copy = np.deep_copy(state)
            self.simulate(state_copy)

        action_type = self.root.action_type
        Nas = [(action, node.N) for action, node in self.root.child.items()]
        actions, count = zip(*Nas)
        if temp==0:
            probs = np.zeros_like(count)
            probs[np.argmax(count)] = 1
            return actions, probs

        counts = [x**(1./temp) for x in count]
        probs = [x/float(sum(counts)) for x in counts]
         
        return action_type, actions, probs