from copy import deepcopy
import numpy as np

from mcts import MCTS
from config import Config

class SelfPlay(object):
    def __init__(self, game, nnet):
        self.game = deepcopy(game)
        self.nnet = nnet
        self.mcts = MCTS(self.game, self.nnet, Config.c_puct, Config.num_sims)
        self.temp_threshold = Config.temp_threshold

    def generate_play_data(self):
        examples = []
        self.game.reset_board()     
        episode_step = 0
        null_cap_pi = np.zeros(self.game.get_capture_action_size())
        null_put_pi = np.zeros(self.game.get_placement_action_size())
        board_state, player_value = self.game.get_current_state()

        while True:
            episode_step += 1
            #self.mcts.reset()

            # Set tempurature to 1 if current turn is less than the threshold
            # TODO: (debugging) Check if temp should be higher for ealier turns and then scale down
            temp = int(episode_step < self.temp_threshold)

            action_type, actions, probs = self.mcts.get_action_prob(board_state, temp=temp)
            examples.append([board_state, action_type, probs, player_value])

            action = actions[np.random.choice(np.arange(len(actions)), p=probs)]
            self.mcts.move_root(action)

            board_state, player_value = self.game.get_next_state(action, action_type)

            winner = self.game.get_game_ended(board_state)

            if winner != 0 or episode_step > 200:
                # Once winner is known, update each example with value based on the current player
                # If the game reaches turn 200 with no winner then it is a draw and value is 0
                new_examples = []
                for e in examples:
                    # TODO: (feature add) how to incorporate state symmetries without creating 
                    #       multiple MCTS trees. Don't have an easy way of converting 
                    #       probs/actions in a symmetrical way.
                    state = e[0]
                    v = winner * e[3]
                    if e[1] == 'PUT':
                        p_placement = e[2]
                        p_capture = null_cap_pi
                        action_type = 1
                    else:
                        p_placement = null_put_pi
                        p_capture = e[2]
                        action_type = 0
                    new_examples.append((state, p_placement, p_capture, v, action_type))
                return new_examples   

class Arena(object):
    def __init__(self, game, player_agent1, player_agent2):
        """
        player_agent1 and playeragent2 are two MCTS instances which have the newest and previous nnets policy_fn.
        """
        self.player1 = player_agent1
        self.player2 = player_agent2
        self.game = game

    def match(self, logging=False):
        """
        Returns 1 if player1 won, -1 if player2 won.
        """
        self.game.reset_board()

        while self.game.get_game_ended() == 0:
            # Reset the MCTS to start at the new board state
            self.player1.reset()
            self.player2.reset()
            state, player_value = self.game.get_current_state()

            # Obtain the policy from the player's agent
            if player_value == 1: # if cur_player is player1
                action_type, actions, probs = self.player1.get_action_prob(state, temp=1)
            else: # plaver_value == -1 and cur_player is player2
                action_type, actions, probs = self.player2.get_action_prob(state, temp=1)

            # Choose the action greedily
            action = actions[np.argmax(probs)]
            if logging:
                print(np.sum(state[:2] + state[2], axis=0))
                print(player_value, action_type, action)
            self.game.get_next_state(action, action_type)

        return self.game.get_game_ended()

    def play_matches(self, num_games):
        player1_win, player2_win, draw = 0, 0, 0
        # Player1 is new model
        for t in xrange(num_games/2):
            if t == 0:
                winner = self.match(logging=False)
            else:
                winner = self.match()
            if winner == 1:
                player1_win += 1
            elif winner == -1:
                player2_win += 1
            else:
                draw += 1

        # Switch who goes first, player2 is new model
        self.player1, self.player2 = self.player2, self.player1
        for _ in xrange(num_games/2):
            winner = self.match()
            if winner == 1:
                player2_win += 1
            elif winner == -1:
                player1_win += 1
            else:
                draw += 1

        return player1_win, player2_win, draw

