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

            actions, probs = self.mcts.get_action_prob((board_state, cur_player), temp=temp)
            # TODO: make sure exmples format compatible with training input format
            examples.append([board_state, probs, cur_player])


            act = np.random.choice(actions, p=probs)
            # TODO: fix ACTION_TYPE
            board_state, cur_player = self.game.get_next_state(act, ACTION_TYPE, (board_state, cur_player))

            winner = self.game.get_game_ended((board_state, self.cur_player))

            if winner!=0:
                return [(e[0],e[1],winner*((-1)**(e[2]!=curPlayer))) for e in examples]



class Arena(object):
	def __init__(self, game, player_agent1, player_agent2):
		self.player1 = player_agent1
        self.player2 = player_agent2
        self.game = game

    def match(self):
    	pass