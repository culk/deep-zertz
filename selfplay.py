from mtcs import MCTS
from config import Config
from copy import deepcopy

class SelfPlay(object):
    def __init__(self, game, nnet):
        self.game = deepcopy(game)
        self.nnet = nnet
        # Either let MCTS takes entire nnet object or just the function
        self.mcts = MCTS(game, nnet.get_policy_fn(), Config.c_puct, Config.num_sim)
        self.temp_threshold = config.temp_threshold

    def generate_play_data(self):
        examples = []
        self.game.reset_board()     
        episode_step = 0

        while True:
            episode_step += 1
            board_state, cur_player = self.game.get_current_state()
            temp = int(episode_step < self.temp_threshold)

            action_type, actions, probs = self.mcts.get_action_prob((board_state, cur_player), temp=temp)
            # TODO: make sure exmples format compatible with training input format
            examples.append([board_state, action_type, probs, cur_player])


            act = np.random.choice(actions, p=probs)
            board_state, cur_player = self.game.get_next_state(act, action_type, (board_state, cur_player))

            winner = self.game.get_game_ended((board_state, self.cur_player))

            if winner!=0:
                return [(e[0],e[1],winner*((-1)**(e[2]!=curPlayer))) for e in examples]



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

            
