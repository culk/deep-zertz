class Config(object):
    """
    Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    models = ['linear', 'dense', 'conv']

    hidden_size = 200
    lr = 0.001

    batch_size = 20
    epochs = 10

    num_layers = 5
    num_filters = 10
    kernel_size = 2
    dropout = 0.1

    model = models[0]
    checkpoint_folder = 'checkpoints'
    arena_games = 40
    arena_threshold = 0.55
    temp_threshold = 15
    num_iters = 10

    custom_loss = False

    c_puct = 1
    num_sims = 5
