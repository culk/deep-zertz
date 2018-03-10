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
    lr = 0.001

    batch_size = 20
    epochs = 10

    num_layers = 5
    num_filters = 32
    kernel_size = 2
    dropout = 0.1

    model = models[2]
    checkpoint_folder = 'checkpoints'
    arena_games = 40
    arena_threshold = 0.55
    # Should be set in a way that encourages exploration in early moves and then 
    # selects optimal moves later in the game
    temp_threshold = 6
    num_iters = 10

    num_episodes = 25
    c_puct = 1
    num_sims = 10
