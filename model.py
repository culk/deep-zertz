'''
This file is a wrapper of the models including functions for training and evaluation
'''
from neural_nets import LinearModel, DenseModel, ConvModel
import numpy as np
import os

class NNetWrapper(object):
    def __init__(self, game, config):
        '''
        Game, an object, needs to have the following attributes:
        game.getBoardSize() -> a tuple like (4, 4)

        game.getActionSize() -> a tuple of (putActionSize, captureActionSize), where each of the actionSize is a tuple
            - putActionSize = (16, 17, 3) * three dimensional
            - captureActionSize = (16, 6) * two dimensional

        game.getStateDepth() -> int. How deep is each state, such as 11

        :param game: A game object
        :param config: A config object. It's by default the config.py
        '''
        self.config = config
        self.game = game

        if self.config.model == 'linear':
            self.nnet = LinearModel(game, config)
        elif self.config.model == 'dense':
            self.nnet = DenseModel(game, config)
        elif self.config.model == 'conv':
            self.nnet = ConvModel(game, config)
        else:
            raise ValueError('The model ' + self.config.model + ' has not been implemented!')

        self.state_depth, self.board_x, self.board_y = game.board.state.shape
        self.put_action_size = game.get_placement_action_size()
        self.capture_action_size = game.get_capture_action_size()


    def train(self, examples):
        '''
        :param examples: (state, pi_put, pi_capture, v) a tuple
                state size=(num_examples, board_x, board_y, state_depth)
                pi_put size = (num_examples, put_pi_size[0] * put_pi_size[1] * put_pi_size[2])
                pi_capture size = (num_examples, capture_pi_size[0] * capture_pi_size[1])
                v size = (num_examples, 1)
        :return:
        '''
        input_states, target_put_pis, target_capture_pis, target_vs = examples

        input_states = np.asarray(input_states)
        target_put_pis = np.asarray(target_put_pis)
        target_capture_pis = np.asarray(target_capture_pis)
        target_vs = np.asarray(target_vs)

        self.nnet.model.fit(x=input_states, y=[target_put_pis, target_capture_pis, target_vs],
                            batch_size=self.config.batch_size, epochs=self.config.epochs, verbose=1)

    def predict(self, states):
        put_pi, capture_pi, v = self.nnet.model.predict(states)

        put_pi_size = self.game.get_placement_action_size()
        capture_pi_size = self.game.get_capture_action_size()

        put_pi = np.reshape(put_pi, (-1, put_pi_size[0], put_pi_size[1], put_pi_size[2]))
        capture_pi = np.reshape(capture_pi, (-1, capture_pi_size[0], capture_pi_size[1]))

        return put_pi, capture_pi, v


    def save_checkpoint(self, filename='checkpoint.pth.tar'):
        folder = self.config.checkpoint_folder
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        folder = self.config.checkpoint_folder
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)