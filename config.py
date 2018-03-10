class Config(object):
    """
    Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    models = ['linear', 'dense', 'conv']

    hidden_size = 10
    # TODO: (feature add) scale the learning rate over time?
    lr = 0.01

    batch_size = 100
    epochs = 50

    num_layers = 5
    num_filters = 16
    kernel_size = 3
    dropout = 0.1

    model = models[2]
    checkpoint_folder = 'checkpoints'
    arena_games = 40
    arena_threshold = 0.55
    # Should be set in a way that encourages exploration in early moves and then 
    # selects optimal moves later in the game
    temp_threshold = 6
    num_iters = 30

    num_episodes = 100
    c_puct = 1
    num_sims = 25

    regularizer = 0.0001
    num_residual_blocks = 3
