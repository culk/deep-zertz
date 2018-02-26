

class Node(object):
    """ A class that represents a node in MC search tree. 
    A should include the following attributes:
        N: the number of times this node has been selected from its parents
        Q: the mean value of this staet
        P: the prior probability of selecting this node (ie. taking this action)
    """
    def __init__(self, parent, P):
        self.N = 0
        self.Q = 0
        self.P = P
        self.child = {}
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
            self.parent.recurse_update(-predicted_v)
        self.update(predicted_v)

    def expand(self, predicted_p):
        """Expand the search tree by attaching child nodes to current state
        NOTE: assuming predicted_p is a list of tuple with action and coresponding prob
        """
        for action, prob in predicted_p:
            if action not in self.child:
                self.child[action] = Node(self, prob)

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

        return best_a

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
        self.root = Node(None, 1.0)

    # should be able to get rid of player argument if we have canonical board
    def simulate(self, board, player):
        """
        Perform one simulation of MCTS. Recursively called until a leaf is found.
        Then uses policy_fn to make prediction of (p,v). This value is propogated up the 
        path.

        NOTE: board should be a deep copy of original board
        """

        node = self.root
        while True:
            if node.is_leaf():
                break
            action = node.get_action(self.c_puct)
            board.take_action(action, __)
            player = -1*player

        # TODO: in game.py, we should implement a function that can spit out game state at specified board and player
        state = self.game.get_current_state(board)
        p, v = self._policy(state)

        # TODO: similarly, we should be able to query if any board state is ended
        end, winner = self.game.is_end(board)
        if not end:
            node.expand(p)
        else:
            if player == winer:
                v = 1
            else:
                v = -1

        node.recurse_update(-v)


    def get_action_prob(self, board, player, temp):

        for _ in range(self.num_sim):
            board_copy = np.deep_copy(board)
            simulate(board, player)

        Nas = [(action, node.N) for action, node in self.root.child.items()]
        actions, count = zip(*Nas)
        if temp==0:
            probs = np.zeros_like(count)
            probs[np.argmax(count)] = 1
            return actions, probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
         
        return actions, probs