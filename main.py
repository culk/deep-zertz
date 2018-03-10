from retrain import Coach
from zertz.ZertzGame import ZertzGame as Game
from model import NNetWrapper as NN
from config import Config

if __name__ == '__main__':
    rings = 19
    marbles = {'w': 10, 'g': 0, 'b': 0}
    win_con = [{'w': 2}]
    t = 2

    game = Game(rings, marbles, win_con, t)
    config = Config()
    nnet = NN(game, config)
    coach = Coach(game, nnet, config)

    # load model weights?

    # learn
    coach.learn()

