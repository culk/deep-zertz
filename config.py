class Config(object):
    """
    Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Training settings
    num_iters = 30
    batch_size = 100
    epochs = 50
    lr = 0.01

    # Neural network settings
    models = ['linear', 'dense', 'conv']
    model = models[2]
    checkpoint_folder = 'checkpoints'
    num_layers = 10
    hidden_size = 64 # linear and dense only
    num_filters = 64 # conv only
    kernel_size = 3 # conv only
    dropout = 0.1 # conv only
    regularizer = 0.0001
    num_residual_blocks = 3 # resnet only

    # MCTS settings
    num_episodes = 25
    num_sims = 10
    c_puct = 1
    # Should be set based on game length to encourage exploration in early moves
    temp_threshold = 6

    use_dirichlet = True
    dir_alpha = 0.05263

    # Unused:
    #arena_games = 40
    #arena_threshold = 0.55

# For AI vs AI play
class Config1(Config):
    model = Config.models[2]
    checkpoint_folder = 'checkpoints_old'

    num_layers = 10 # both
    num_filters = 64 # conv only
    hidden_size = 64 # linear and dense only

class Config2(Config):
    model = Config.models[2]
    checkpoint_folder = 'checkpoints_old'

    num_layers = 15 # both
    num_filters = 16 # conv only
    hidden_size = 64 # linear and dense only

