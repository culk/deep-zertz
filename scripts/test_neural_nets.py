from model import NNetWrapper
import numpy as np
from config import Config

class Game(object):
    def __init__(self):
        print 'New Game built!'

    def getBoardSize(self):
        return (4, 4)

    def getActionSize(self):
        putActionSize = (16, 17, 3)
        captureActionSize = (16, 6)
        return (putActionSize, captureActionSize)

    def getStateDepth(self):
        return 11


def generate_states(game, num_states=10):
    board_x, board_y = game.getBoardSize()
    state_depth = game.getStateDepth()
    if Config.model == 'linear' or Config.model == 'dense':
        states = np.random.randint(2, size=(num_states, board_x * board_y * state_depth))
    else:
        states = np.random.randint(2, size=(num_states, board_x, board_y, state_depth))
    return states

def generate_train_examples(game, num_examples=1000):
    # Generate states
    board_x, board_y = game.getBoardSize()
    state_depth = game.getStateDepth()
    put_pi_size, capture_pi_size = game.getActionSize()


    if Config.model == 'linear' or Config.model == 'dense':
        states = np.random.randint(2, size=(num_examples, board_x * board_y * state_depth))
    else:
        states = np.random.randint(2, size=(num_examples, board_x, board_y, state_depth))

    put_pi = np.random.random((num_examples, put_pi_size[0] * put_pi_size[1] * put_pi_size[2]))
    capture_pi = np.random.random((num_examples, capture_pi_size[0] * capture_pi_size[1]))
    v = np.random.random((num_examples, 1))

    examples = (states, put_pi, capture_pi, v)
    return examples


def test_model():
    game = Game()
    model = NNetWrapper(game, config=Config)
    examples = generate_train_examples(game)
    eval_states = generate_states(game)

    model.train(examples)
    put_pi, capture_pi, v = model.predict(eval_states)

    print 'Predicting...'
    print 'put_pi = ', put_pi.shape
    print 'capture_pi = ', capture_pi.shape
    print 'v = ', v

if __name__ == '__main__':
    print 'Start testing...'
    test_model()

