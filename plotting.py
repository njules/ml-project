import matplotlib.pyplot as plt
import numpy as np

ATTRIBUTE_NAMES = {
    0: "fixed acidity (mg/dm³)",
    1: "volatile acidity (mg/dm³)",
    2: "citric acid (mg/dm³)",
    3: "residual sugar (mg/dm³)",
    4: "chlorides (g/dm³)",
    5: "free sulfur dioxide (mg/dm³)",
    6: "total sulfur dioxide (mg/dm³)",
    7: "density (g/cm³)",
    8: "acidity (pH)",
    9: "sulphates (g/dm³)",
    10: "alcohol (%)",
}

LABEL_NAMES = {
    0: 'low quality',
    1: 'high quality',
}

COLOR_0 = 'blue'
COLOR_1 = 'red'


def split_data_classes(data: np.ndarray, labels: np.ndarray):
    classes = np.unique(labels)
    split_data = [data[labels == c, :] for c in classes]
    return split_data, classes


def plot_hist(data: np.ndarray, labels: np.ndarray):

    D_c, c = split_data_classes(data, labels)

    for idx_attribute in range(len(ATTRIBUTE_NAMES)):
        plt.figure()
        plt.xlabel(ATTRIBUTE_NAMES[idx_attribute])

        for idx_class in range(len(c)):
            plt.hist(D_c[idx_class][:, idx_attribute], bins='auto', alpha=0.4, label=LABEL_NAMES[c[idx_class]])

        plt.legend()
    plt.show()


def plot_dims(data: np.ndarray, labels: np.ndarray):

    D_c, c = split_data_classes(data, labels)
    n_dims = data.shape[1]
    fig, axs = plt.subplots(n_dims, n_dims, sharex='col')

    for plot_dim_1 in range(n_dims):
        axs[plot_dim_1, 0].set_ylabel(f'dimension {plot_dim_1+1}')
        axs[n_dims-1, plot_dim_1].set_xlabel(f'dimension {plot_dim_1+1}')
        for plot_dim_2 in range(n_dims):

            if plot_dim_1 == plot_dim_2:
                for idx_class in range(len(c)):
                    axs[plot_dim_1, plot_dim_2].hist(D_c[idx_class][:, plot_dim_1], bins='auto', alpha=0.4)
            else:
                for idx_class in range(len(c)):
                    axs[plot_dim_2, plot_dim_1].scatter(D_c[idx_class][:, plot_dim_1],
                                                        D_c[idx_class][:, plot_dim_2], marker='x')
    plt.show()


def plot_scree(eig_values):
    plt.figure()
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalues')
    plt.plot(range(1, len(eig_values) + 1), eig_values, marker='o')
    plt.show()


def confusion_matrix(labels, predictions):
    c_matrix = np.zeros((np.max(predictions)+1, np.max(labels)+1), np.int32)
    for idx in range(labels.shape[0]):
        c_matrix[predictions[idx], labels[idx]] += 1
    return c_matrix


def accuracy(labels, predictions):
    total = predictions.shape[0]
    correct = np.sum(labels == predictions)
    return correct / total


def error_rate(labels, predictions):
    total = predictions.shape[0]
    miss = np.sum(labels != predictions)
    return miss / total
