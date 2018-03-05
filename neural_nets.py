from keras.layers import Input, Reshape, Dense
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class LinearModel():
    def __init__(self, game, config):
        self.board_x, self.board_y = game.getBoardSize()
        self.state_depth = game.getStateDepth()
        self.put_action_size, self.capture_action_size = game.getActionSize()
        self.config = config

        input_dim = self.board_x * self.board_y * self.state_depth
        put_action_dim = np.prod(self.put_action_size)
        capture_action_dim = np.prod(self.capture_action_size)

        inputs = Input(shape=(input_dim, ))
        # Input() is used to instantiate a Keras tensor. Shape does not including the batch size
        # TODO: make sure that the state is in shape (x, y, depth)

        hidden = Dense(self.config.hidden_size, input_shape=(input_dim,))(inputs)
        self.pi_put = Dense(put_action_dim, activation='softmax', name='pi_put')(hidden)
        self.pi_capture = Dense(capture_action_dim, activation='softmax', name='pi_capture')(hidden)
        self.v = Dense(1, activation='tanh', name='v')(hidden)

        self.model = Model(inputs=inputs, outputs=[self.pi_put, self.pi_capture, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.config.lr))