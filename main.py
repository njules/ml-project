import csv
import numpy as np
from sklearn.decomposition import PCA

from plotting import plot_hist, plot_dims


def read_data(path: str):
    data = []
    labels = []

    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for csvline in csvreader:
            data.append(csvline[:11])
            labels.append(csvline[-1])

    return np.array(data), np.array(labels, dtype=np.int32)


def pca(data: np.ndarray, n_components: int):

    mean = data.mean(0)

    normalized_data = (data - mean)

    cov = np.dot(normalized_data.T, normalized_data) / normalized_data.shape[0]

    eig_values, eig_vectors = np.linalg.eigh(cov)

    principal_components = eig_vectors[:, ::-1][:, :n_components]

    projected_data = np.dot(data, principal_components)

    testPCA = PCA(n_components=n_components)
    test_red = testPCA.fit_transform(data)
    print("\ndata variance")
    print(data.std(0))
    print(sum(data.std(0)))
    print("\nscikit variance")
    print(test_red.std(0))
    print(sum(test_red.std(0)))
    print("\nmy PCA variance")
    print(projected_data.std(0))
    print(sum(projected_data.std(0)))
    print("\nmy PCA % explained variance")
    print(sum(projected_data.std(0) / sum(data.std(0))))

    return projected_data, principal_components, eig_values[::-1][:n_components]


if __name__ == '__main__':

    data_train, labels_train = read_data('data/Train.txt')

    # plot histogram for original attributes
    # plot_hist(data_train, labels_train)

    pca_data, _, _ = pca(data_train, 11)

    # plot_dims(data_train, labels_train)
