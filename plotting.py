import matplotlib.pyplot as plt
import matplotlib
import numpy as np

ATTRIBUTE_NAMES = {
    0: "fixed acidity",
    1: "volatile acidity",
    2: "citric acid",
    3: "residual sugar",
    4: "chlorides",
    5: "free sulfur dioxide",
    6: "total sulfur dioxide",
    7: "density",
    8: "pH",
    9: "sulphates",
    10: "alcohol"
}

LABEL_0 = 'low quality'
LABEL_1 = 'high quality'

COLOR_0 = 'blue'
COLOR_1 = 'red'


def plot_hist(data: np.ndarray, labels: np.ndarray):

    for idx in range(len(ATTRIBUTE_NAMES)):

        plt.figure()

        plt.xlabel(ATTRIBUTE_NAMES[idx])
        plt.hist(data[labels == 0, idx], bins='auto', alpha=0.4, label='low quality')
        plt.hist(data[labels == 1, idx], bins='auto', alpha=0.4, label='high quality')

        plt.legend()

    plt.show()


def plot_dims(data: np.ndarray, labels: np.ndarray):

    n_dims = data.shape[1]

    fig, axs = plt.subplots(n_dims, n_dims)

    for plot_dim_1 in range(n_dims):
        for plot_dim_2 in range(n_dims):

            if plot_dim_1 == plot_dim_2:
                axs[plot_dim_1, plot_dim_2].hist(data[labels == 0, plot_dim_1], bins='auto', alpha=0.4)
                axs[plot_dim_1, plot_dim_2].hist(data[labels == 1, plot_dim_1], bins='auto', alpha=0.4)

            else:
                axs[plot_dim_2, plot_dim_1].scatter(data[labels == 0, plot_dim_1], data[labels == 0, plot_dim_2], marker='x')
                axs[plot_dim_2, plot_dim_1].scatter(data[labels == 1, plot_dim_1], data[labels == 1, plot_dim_2], marker='x')

    plt.show()
