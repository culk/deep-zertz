from neural_nets import LinearModel
import numpy as np
import os

class NNetWrapper(object):
    def __init__(self, game, config):
        self.config = config
        self.game = game

        if self.config.model == 'linear':
            self.nnet = LinearModel(game, config)
        else:
            raise ValueError('The model ' + self.config.model + ' has not been implemented!')

        self.board_x, self.board_y = game.getBoardSize()
        self.state_depth = game.getStateDepth()
        self.put_action_size, self.capture_action_size = game.getActionSize()


    def train(self, examples):
        '''
        :param examples: (state, pi_put, pi_capture, v). State has shape (num_examples, x * y * d)
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

        put_pi_size, capture_pi_size = self.game.getActionSize()
        put_pi = np.reshape(put_pi[0], (-1, put_pi_size[0], put_pi_size[1], put_pi_size[2]))
        capture_pi = np.reshape(capture_pi[0], (-1, capture_pi_size[0], capture_pi_size[1]))

        return put_pi, capture_pi, v[0]


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