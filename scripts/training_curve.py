import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def generate_loss_arrray():
    file_head = 'results/log_64_10/log_64_10_'
    save_path = 'results/log10_64' + '.npy'

    results = []
    for i in range(30):
        f = file_head + str(i) + '.csv'
        df = pd.read_csv(f).as_matrix()
        results.append(df[-1, 1])

    np.save(save_path, np.array(results))
    print np.array(results)

def concat_loss():
    folder = 'results/'
    file_list = ['dense.npy', 'log5_16.npy', 'log5_32.npy', 'log5_64.npy',
                 'log10_16.npy', 'log10_32.npy', 'log10_64.npy',
                 'log15_16.npy', 'log15_32.npy']
    loss = np.array(range(30)).reshape((-1, 1))

    for f in file_list:
        loss_array = np.load(folder + f).reshape((-1, 1))
        loss = np.concatenate((loss, loss_array), axis=1)

    loss = pd.DataFrame(loss, columns=['step', 'dense', '5_16', '5_32', '5_64', '10_16', '10_32', '10_64', '15_16', '15_32'])
    loss.to_csv('results/training_curve.csv', index=False)
    print loss.head()

def plot_curve():
    path = 'results/training_curve.csv'
    array = pd.read_csv(path).as_matrix()

    # plt.plot(array[:, 0], array[:, 1], label='Dense NN')
    for i in range(2, 10):
        num_layers = (i - 2)//3 * 5 + 5
        hidden_size = 2**((i - 2) % 3) * 16
        plt.plot(array[:, 0], array[:, i], label='CNN w/ %i layers, %i filters' %(num_layers, hidden_size))

    plt.xlabel('# Iterations')
    plt.ylabel('Loss')
    plt.title('Training Curves of Different Models')

    fontP = FontProperties()
    fontP.set_size('small')

    leg = plt.legend(prop = fontP, fancybox=True)
    leg.get_frame().set_alpha(0.2)
    plt.show()

plot_curve()

