from keras.layers import Input, Reshape, Dense, Conv2D, BatchNormalization, Activation, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class LinearModel(object):
    def __init__(self, game, config):
        self.board_x, self.board_y = game.getBoardSize()
        self.state_depth = game.getStateDepth()
        self.put_action_size, self.capture_action_size = game.getActionSize()
        self.config = config

        input_dim = self.board_x * self.board_y * self.state_depth
        put_action_dim = np.prod(self.put_action_size)
        capture_action_dim = np.prod(self.capture_action_size)

        inputs = Input(shape=(input_dim, ))

        hidden = Dense(self.config.hidden_size, input_shape=(input_dim,))(inputs)
        self.pi_put = Dense(put_action_dim, activation='softmax', name='pi_put')(hidden)
        self.pi_capture = Dense(capture_action_dim, activation='softmax', name='pi_capture')(hidden)
        self.v = Dense(1, activation='tanh', name='v')(hidden)

        self.model = Model(inputs=inputs, outputs=[self.pi_put, self.pi_capture, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.config.lr))

class DenseModel(object):
    def __init__(self, game, config):
        self.board_x, self.board_y = game.getBoardSize()
        self.state_depth = game.getStateDepth()
        self.put_action_size, self.capture_action_size = game.getActionSize()
        self.config = config

        input_dim = self.board_x * self.board_y * self.state_depth
        put_action_dim = np.prod(self.put_action_size)
        capture_action_dim = np.prod(self.capture_action_size)

        inputs = Input(shape=(input_dim, ))
        hidden = inputs

        for i in range(self.config.num_layers):
            hidden = Dense(self.config.hidden_size, input_shape=(input_dim,))(inputs)

        self.pi_put = Dense(put_action_dim, activation='softmax', name='pi_put')(hidden)
        self.pi_capture = Dense(capture_action_dim, activation='softmax', name='pi_capture')(hidden)
        self.v = Dense(1, activation='tanh', name='v')(hidden)

        self.model = Model(inputs=inputs, outputs=[self.pi_put, self.pi_capture, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.config.lr))


class ConvModel(object):
    def __init__(self, game, config):
        self.board_x, self.board_y = game.getBoardSize()
        self.state_depth = game.getStateDepth()
        self.put_action_size, self.capture_action_size = game.getActionSize()
        self.config = config

        put_action_dim = np.prod(self.put_action_size)
        capture_action_dim = np.prod(self.capture_action_size)

        inputs = Input(shape=(self.board_x, self.board_y, self.state_depth))
        hidden = inputs

        for i in range(self.config.num_layers):
            hidden = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.config.num_filers,
                                                                    self.config.kernel_size,
                                                                    padding='same')(hidden)))
        hidden = Flatten()(hidden)
        hidden = Dropout(self.config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(hidden))))
        hidden = Dropout(self.config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(hidden))))

        self.pi_put = Dense(put_action_dim, activation='softmax', name='pi_put')(hidden)
        self.pi_capture = Dense(capture_action_dim, activation='softmax', name='pi_capture')(hidden)
        self.v = Dense(1, activation='tanh', name='v')(hidden)

        self.model = Model(inputs=inputs, outputs=[self.pi_put, self.pi_capture, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.config.lr))