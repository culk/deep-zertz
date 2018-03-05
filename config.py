class Config(object):
    """
    Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    models = ['linear', 'dense']

    hidden_size = 200
    lr = 0.001

    batch_size = 20
    epochs = 10

    num_layers = 5

    model = models[1]
    checkpoint_folder = 'checkpoint'