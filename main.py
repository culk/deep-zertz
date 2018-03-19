from retrain import Coach, Individual
from selfplay import Arena, HumanPlay
from mcts import MCTS
from zertz.ZertzGame import ZertzGame as Game
from model import NNetWrapper as NN
from config import Config, Config1, Config2

if __name__ == '__main__':
    # Game settings
    rings = 19
    marbles = {'w': 10, 'g': 10, 'b': 10}
    win_con = [{'w': 2}, {'g': 2}, {'b': 2}, {'w': 1, 'g': 1, 'b': 1}]
    t = 5

    # Setup
    game = Game(rings, marbles, win_con, t)
    config = Config()
    nnet = NN(game, config)

    # Option #1: Learn
    trainer = Individual(game, nnet, config)
    trainer.learn()

    # Option #2: Human vs AI
    #nnet.load_checkpoint(filename='checkpoint_32_10_0001_29.pth.tar')
    #ai_agent = MCTS(game, nnet, config.c_puct, config.num_sims)
    #hp = HumanPlay(game, ai_agent)
    #hp.play()

    # Option #3: AI vs AI
    #config1 = Config1()
    #config2 = Config2()
    #nnet1 = NN(game, config1)
    #nnet2 = NN(game, config2)

    #nnet1.load_checkpoint(filename='checkpoint_64_10_29.pth.tar')
    #nnet2.load_checkpoint(filename='checkpoint_16_15_29.pth.tar')

    #ai_agent1 = MCTS(game, nnet1, config1.c_puct, config1.num_sims)
    #ai_agent2 = MCTS(game, nnet2, config2.c_puct, config2.num_sims)

    #arena = Arena(game, ai_agent1, ai_agent2)
    #ai1_win, ai2_win, draw = arena.play_matches(10)
    #print(ai1_win, ai2_win, draw)

