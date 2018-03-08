from mcts import MCTS
from config import Config
from copy import deepcopy
import numpy as np

class SelfPlay(object):
    def __init__(self, game, nnet):
        self.game = deepcopy(game)
        self.nnet = nnet
        self.mcts = MCTS(self.game, nnet, Config.c_puct, Config.num_sims)
        self.temp_threshold = Config.temp_threshold

    def generate_play_data(self):
        examples = []
        self.game.reset_board()     
        episode_step = 0
        null_cap_pi = np.zeros_like(self.game.get_capture_action_shape())
        null_put_pi = np.zeros_like(self.game.get_placement_action_shape())

        while True:
            episode_step += 1
            board_state, player_value = self.game.get_current_state()
            temp = int(episode_step < self.temp_threshold)

            action_type, actions, probs = self.mcts.get_action_prob(board_state, temp=temp)
            # TODO: make sure exmples format compatible with training input format
            examples.append([board_state, action_type, probs, player_value])


            act = np.random.choice(actions, p=probs)
            board_state, player_value = self.game.get_next_state(act, action_type, board_state)

            winner = self.game.get_game_ended(board_state)

            if winner != 0:
                new_examples = []
                for e in examples:
                    if e[1] == 'PUT':
                        new_examples.append((e[0],e[2], null_cap_pi, 1 if winner==player_value else -1, 1))
                    else:
                        new_examples.append((e[0], null_put_pi, e[2], 1 if winner==player_value else -1, 0))
                np_board = np.array([ne[0] for ne in new_examples])
                np_pi_put = np.array([ne[1] for ne in new_examples])
                np_pi_cap = np.array([ne[2] for ne in new_examples])
                np_v = np.array([ne[3] for ne in new_examples])
                np_mask = np.array([ne[4] for ne in new_examples])
                return (np_board, np_pi_put, np_pi_cap, np_v, np_mask)

            if episode_step > 2000 and winner == 0:
                return self.generate_play_data()


class Arena(object):
    def __init__(self, game, player_agent1, player_agent2):
        """
        player_agent1 and playeragent2 are two MCTS instances which have the newest and previous nnets policy_fn.
        """
        self.player1 = player_agent1
        self.player2 = player_agent2
        self.game = game

    def match(self):
        """
        Returns 1 if player1 won, -1 if player2 won.
        """
        self.game.reset_board()

        while self.game.get_game_ended() == 0:
            state, player_value = self.game.get_current_state()
            if player_value == 1: # if cur_player is player1
                action_type, actions, probs = self.player1.get_action_prob(state, temp=0)
            else: # plaver_value == -1
                action_type, actions, probs = self.player2.get_action_prob(state, temp=0)

            act = action[np.argmax(probs)]
            self.game.get_next_state(action, action_type)

        return self.game.get_game_ended()

    def play_matches(self, num_games):
        player1_win, player2_win, draw = 0, 0, 0
        for _ in range(num_games/2):
            winner = self.match()
            if winner == 1:
                player1_win += 1
            elif winner == -1:
                player2_win += 1
            else:
                draw += 1

        self.player1, self.player2 = self.player2, self.player1
        for _ in range(num_games/2):
            winner = self.match()
            if winner == 1:
                player1_win += 1
            elif winner == -1:
                player2_win += 1
            else:
                draw += 1

        return player1_win, player2_win, draw

            
