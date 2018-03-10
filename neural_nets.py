'''
neural_nets.py contains various network structure, including linear model, dense model, conv model and residual net
model
'''
from keras.layers import Input, Reshape, Dense, Conv2D, BatchNormalization, Activation, Flatten, Dropout, Lambda, Multiply
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

'''
# Possible alternate loss function
def combined_loss(args):
    is_put, pi_put, pi_cap, v = args
    not_is_put = Lambda(lambda x: (x - 1) * -1)
    put_loss = categorical_crossentropy(
    cap_loss = 
    v_loss = 
    return is_put * put_loss + not_is_put * cap_loss + v_loss

# This part would go in each model
custom_loss = Lambda(combined_loss, output_shape=(1,), name='combined_loss')([inputs, self.pi_put, self.pi_capture, self.v])
'''

class LinearModel(object):
    '''
    A linear model takes in a state and estimates the corresponding pi_put, pi_capture and v
    '''
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
        self.state_depth, self.board_x, self.board_y = game.board.state.shape
        self.put_action_size = game.get_placement_action_size()
        self.capture_action_size = game.get_capture_action_size()
        self.config = config

        inputs = Input(shape=(self.state_depth, self.board_x, self.board_y), name="inputs")

        hidden = Flatten()(inputs)
        hidden = Dense(self.config.hidden_size, activation='linear')(hidden)
        self.pi = Dense(self.put_action_size + self.capture_action_size, activation='softmax', name='pi')(hidden)
        self.v = Dense(1, activation='tanh', name='v')(hidden)

        self.model = Model(inputs=[inputs], outputs=[self.pi, self.v])

        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.config.lr))

class DenseModel(object):
    '''
    Fully connected neural networks. Number of layers is decided by config.num_layers
    '''
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

        self.state_depth, self.board_x, self.board_y = game.board.state.shape
        self.put_action_size = game.get_placement_action_size()
        self.capture_action_size = game.get_capture_action_size()
        self.config = config

        inputs = Input(shape=(self.state_depth, self.board_x, self.board_y), name="inputs")

        hidden = Flatten()(inputs)

        for i in range(self.config.num_layers):
            hidden = Dense(self.config.hidden_size, activation='relu')(hidden)

        self.pi = Dense(self.put_action_size + self.capture_action_size, activation='softmax', name='pi')(hidden)
        self.v = Dense(1, activation='tanh', name='v')(hidden)


        self.model = Model(inputs=[inputs], outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(self.config.lr))


class ConvModel(object):
    '''
    A convolution NN. filters are decided by config.num_filters and config.kernel_size
    '''
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
        self.state_depth, self.board_x, self.board_y = game.board.state.shape
        self.put_action_size = game.get_placement_action_size()
        self.capture_action_size = game.get_capture_action_size()
        self.config = config

        inputs = Input(shape=(self.state_depth, self.board_x, self.board_y), name="inputs")

        hidden = inputs

        for i in range(self.config.num_layers):
            hidden = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.config.num_filters,
                                                                    self.config.kernel_size,
                                                                    padding='same',
                                                                    data_format='channels_first')(hidden)))
        hidden = Flatten()(hidden)
        hidden = Dropout(self.config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(hidden))))
        hidden = Dropout(self.config.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(hidden))))

        self.pi = Dense(self.put_action_size + self.capture_action_size, activation='softmax', name='pi')(hidden)
        self.v = Dense(1, activation='tanh', name='v')(hidden)

        self.model = Model(inputs=[inputs], outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.config.lr))
