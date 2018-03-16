import numpy as np

class Node(object):
    """ A class that represents a node in MC search tree. 
    A should include the following attributes:
        N: the number of times this node has been selected from its parents
        Q: the mean value of this state
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
        """
        Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.        
        """
        # Update mean value of the state
        self.Q = (self.Q * self.N + predicted_v) / (self.N + 1.)
        # Increment how many times the state has been visited
        self.N += 1

    def recurse_update(self, predicted_v):
        """
        Call by MCTS to recursively propagate predicted_v up to ancestor
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.recurse_update(predicted_v)
        self.update(predicted_v)

    def expand(self, action_type, predicted_p):
        """
        Expand the search tree by attaching child nodes to current state
        Args:
            action_type - 'PUT' or 'CAP' depending on the action
            predicted_p - predicted probability from the neural network
        """
        assert abs(np.sum(predicted_p) - 1) < .0001

        self.action_type = action_type
        for action in zip(*np.where(predicted_p > 0)):
            prob = predicted_p[action]
            self.child[action] = Node(self, prob, 0)

    def get_action(self, c_puct):
        """
        Gets best action based on current estimate of Q and U
        """
        max_u = float('-inf')
        best_a = None
        next_node = None

        for action, node in self.child.items():
            # TODO: (feature add) scale the prior probabilities for the root node based on the size
            #       of the typical action state space (page 14 AlphaZero paper).
            U = self.cur_player * node.Q + c_puct * node.P * np.sqrt(self.N) / (1. + node.N)
            if U > max_u:
                max_u = U
                best_a = action
                next_node = node

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

    def reset(self, cur_player):
        # Reset the tree and create a root node for the current player
        self.root = Node(None, 1.0, cur_player)

    def move_root(self, action, cur_player):
        # Move the root to the child node corresponding to the action.
        # Requires cur_player in case the action isn't already a child.
        if action in self.root.child and self.root.child[action].cur_player != 0:
            self.root = self.root.child[action]
            self.root.parent = None
        else:
            self.reset(cur_player)

    def simulate(self, board_state):
        """
        Perform one simulation of MCTS. Recursively called until a leaf is found.
        Then uses policy_fn to make prediction of (p,v). This value is propogated up the 
        path.
        Args:
            board_state is a 3-D array representing game state
        """
        node = self.root

        while True:
            if node.is_leaf():
                break
            action_type, best_a, node = node.get_action(self.c_puct)
            next_board_state, player_value = self.game.get_next_state(best_a, action_type, board_state)
            if node.cur_player == 0:
                node.cur_player = player_value
            board_state = next_board_state
        
        # Check if the leaf node is a game over state
        game_value = self.game.get_game_ended(board_state)

        if game_value == 0:
            # No player has won, predict the policy distribution and state value to add nodes
            # TODO: remove the below code because action_filter isn't needed to predict anymore
            # Get which type of action is valid from the leaf node board state
            valid_placement, valid_capture = self.game.get_valid_actions(board_state)
            if np.any(valid_placement == True):
                action_filter = 1
            else:
                action_filter = 0

            # Get a symmetrical board_state and call predict to get the policy and value
            symmetries = self.game.get_symmetries(board_state)
            selected_symmetry = symmetries[np.random.choice(np.arange(len(symmetries)))]
            symmetry_id, symmetrical_state = selected_symmetry

            # TODO: (feature add) split the policy into placement and capture and reshape them
            p_placement, p_capture, v = self.nnet.predict(symmetrical_state, action_filter)
            p_placement = np.squeeze(p_placement)
            p_capture = np.squeeze(p_capture)
            v = np.squeeze(v)

            # Translate the actions back for the current board_state
            p_placement = self.game.translate_action_symmetry('PUT', symmetry_id, p_placement)
            p_capture = self.game.translate_action_symmetry('CAP', symmetry_id, p_capture)
            # For planned changes to opponent symmetry generation
            if symmetry_id > 3:
                v = -v

            if np.any(valid_placement):
                p_placement = np.multiply(p_placement, valid_placement)
                if np.sum(p_placement) == 0:
                    p_placement = valid_placement.astype(np.float32)
                p_placement /= np.sum(p_placement)
                node.expand('PUT', p_placement)
            else:
                p_capture = np.multiply(p_capture, valid_capture)
                if np.sum(p_capture) == 0:
                    p_capture = valid_capture.astype(np.float32)
                p_capture /= np.sum(p_capture)
                node.expand('CAP', p_capture)

        else:
            # If game is over we know the true value of the game
            v = game_value

        # Use the true or predicted value of the game to update the nodes
        node.recurse_update(v)

    def get_action_prob(self, state, temp):
        # Return the actions and corresponding probabilities for the current state.
        #   temp is the temperature to control exploration/eploitation
        for _ in xrange(self.num_sim):
            state_copy = np.copy(state)
            self.simulate(state_copy)

        # Get list of actions from tree root and number of times each child has been visited
        action_type = self.root.action_type
        action_visits = [(action, node.N) for action, node in self.root.child.items()]
        actions, visits = zip(*action_visits)

        if temp == 0:
            # Exploitation, recommend the action that has the highest visit count
            # TODO: (debugging) actions seem to be clustered to only a few
            #explored = list(np.where(np.array(visits) > 0)[0])
            #print(explored)
            #for a in explored:
                #n = self.root.child[actions[a]]
                #print(actions[a], n.P, n.N, n.Q)
            # end debug code
            probs = np.zeros(len(visits), dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
            actions, probs = self.restore_action_matrix(actions, probs)
        else:
            # Exploration, assign some probability to less visited child nodes
            probs = np.array(visits, dtype=np.float32)**(1. / temp)
            actions, probs = self.restore_action_matrix(actions, probs)

        return action_type, actions, probs

    def restore_action_matrix(self, actions, probs):
        # Returns lists of actions and their corresponding probabilities for all 
        # actions of the current action type. Invalid actions will have 0 probability.
        if self.root.action_type == 'PUT':
            probs_full = np.zeros(self.game.get_placement_action_shape())
        else:
            probs_full = np.zeros(self.game.get_capture_action_shape())

        for index, p in zip(actions, probs):
            probs_full[index] = p

        z, y, x = probs_full.shape
        actions_full = [(i, j, k) for i in xrange(z) for j in xrange(y) for k in xrange(x)]
        probs_full = probs_full.flatten()
        probs_full /= np.sum(probs_full)

        assert abs(np.sum(probs_full) - 1) < .0001

        return actions_full, probs_full

