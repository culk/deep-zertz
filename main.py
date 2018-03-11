from retrain import Coach, Individual
from selfplay import HumanPlay
from mcts import MCTS
from zertz.ZertzGame import ZertzGame as Game
from model import NNetWrapper as NN
from config import Config

if __name__ == '__main__':
    rings = 19
    marbles = {'w': 10, 'g': 10, 'b': 10}
    win_con = [{'w': 2}, {'g': 2}, {'b': 2}, {'w': 1, 'g': 1, 'b': 1}]
    t = 3

    game = Game(rings, marbles, win_con, t)
    config = Config()
    nnet = NN(game, config)
    trainer = Individual(game, nnet, config)

    # load model weights?

    # learn
    trainer.learn()

    # play against AI
    #ai_agent = MCTS(game, nnet, config.c_puct, config.num_sims)
    #hp = HumanPlay(game, ai_agent)
    #hp.play()

