import csv
from abc import ABC
from typing import List, Any, Union, Tuple, Optional

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA as cheatPCA
import test

from plotting import plot_scree, plot_hist, confusion_matrix, plot_dims, accuracy, error_rate
from classifiers import MVG


class DrModel(ABC):
    def fit(self, data: np.ndarray, labels: np.ndarray):
        pass

    def transform(self, data: np.ndarray):
        pass


# convenience class for cross validation function
class NoDr(DrModel):
    @staticmethod
    def fit(data: np.ndarray, labels=None):
        return

    @staticmethod
    def transform(data: np.ndarray):
        return data


class PCA(DrModel):
    eig_values: np.ndarray
    principal_components: np.ndarray

    explained_variance: float

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, data: np.ndarray, labels=None):
        mean = data.mean(0)
        normalized_data = data - mean

        cov = np.cov(normalized_data.T, ddof=0)
        eig_values, eig_vectors = np.linalg.eigh(cov)

        self.eig_values = eig_values[::-1][:self.n_components]
        self.principal_components = eig_vectors[:, ::-1][:, :self.n_components]

        self.explained_variance = sum(self.eig_values) / sum(eig_values)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.dot(data, self.principal_components)


class LDA(DrModel):
    eig_values: np.ndarray
    eig_vectors: np.ndarray

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, data: np.ndarray, labels: np.ndarray):
        classes = np.unique(labels)
        num_classes = len(classes)

        mu = data.mean(0)
        N = data.shape[0]

        D_c = [data[labels == c, :] for c in classes]
        mu_c = [D.mean(0) for D in D_c]
        n_c = [D.shape[0] for D in D_c]

        S_B = sum([n_c[c_idx] * np.outer(mu_c[c_idx] - mu, mu_c[c_idx] - mu) for c_idx in range(num_classes)]) / N
        S_W = sum([np.dot((D_c[c_idx] - mu_c[c_idx]).T, D_c[c_idx] - mu_c[c_idx]) for c_idx in range(num_classes)]) / N

        s, U = eigh(S_B, S_W)
        sort_idx = np.argsort(-s)

        self.eig_values = s[sort_idx][:self.n_components]
        self.eig_vectors = np.real(U)[:, sort_idx][:, :self.n_components]

    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.dot(data, self.eig_vectors)


class SVM:
    _duality_gap: float
    support_vectors: np.ndarray
    alpha_z_sv: np.ndarray

    def __init__(self, C: float, kernel=None):
        self.C = C
        if kernel is None:
            self.kernel = lambda X_1, X_2: SVM.linear_kernel(X_1, X_2, 1)

    def fit(self, data: np.ndarray, labels: np.ndarray):

        (N, D) = data.shape
        z = labels * 2 - 1
        self.H = self.kernel(data, data) * np.outer(z, z)

        alpha, t, _ = fmin_l_bfgs_b(self.obj_func, x0=np.zeros((N,)), bounds=[(0, self.C)] * N, factr=1)

        eps = 1e-9
        self.support_vectors = data[alpha > eps, :]
        self.alpha_z_sv = alpha[alpha > eps] * labels[alpha > eps]


        K=1
        x_hat = np.append(data, values=np.ones((data.shape[0], 1)) * K, axis=1)
        self.H_hat = self.kernel(x_hat, x_hat) * np.outer(z, z)
        alpha_hat, t, _ = fmin_l_bfgs_b(self.obj_func, x0=np.zeros((N,)), bounds=[(0, self.C)] * N, factr=1)
        w_hat = (alpha * z).T @ x_hat
        self.w_hat = w_hat[:D]
        self.b_hat = w_hat[D]
        print(self.b_hat)

        primal = 1 / 2 * w_hat.T @ w_hat + self.C * np.sum(
            [np.max([0, 1 - z[i] * w_hat.T @ x_hat[i, :]]) for i in range(N)])
        dual = -self.obj_func(alpha)[0]
        self._duality_gap = primal - dual

    def obj_func(self, alpha):
        loss = 1 / 2 * alpha.T @ self.H @ alpha - sum(alpha)
        delta_loss = self.H @ alpha - 1
        return loss, delta_loss

    def obj_func_hat(self, alpha):
        loss = 1 / 2 * alpha.T @ self.H_hat @ alpha - sum(alpha)
        delta_loss = self.H @ alpha - 1
        return loss, delta_loss

    def predict(self, data):
        predictions = np.zeros((data.shape[0],), dtype=np.int32)
        predictions[self.kernel(data, self.support_vectors) @ self.alpha_z_sv + self.b_hat > 0] = 1
        return predictions

    def predict_hat(self, data):
        predictions = np.zeros((data.shape[0],), dtype=np.int32)
        predictions[data @ self.w_hat + self.b_hat > 0] = 1
        return predictions

    @staticmethod
    def linear_kernel(X_1: np.ndarray, X_2: np.ndarray, K: float) -> np.ndarray:
        return X_1 @ X_2.T + K**2

    @staticmethod
    def polynomial_kernel(X:np.ndarray, c: float, d: float) -> np.ndarray:
        return ((X @ X.T) + c) ** d

    @staticmethod
    def polynomial_kernel(X:np.ndarray, g: float) -> np.ndarray:
        distance_squared = X
        return np.exp(-g * 0)


def read_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []

    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for csvline in csvreader:
            data.append(csvline[:11])
            labels.append(csvline[-1])

    return np.array(data), np.array(labels, dtype=np.int32)


def cross_val_split(X: np.ndarray, Y: np.ndarray, k: int) -> List[
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    classes = np.unique(Y)

    c_idxs = [Y == c for c in classes]
    X_c_split = [np.array_split(X[I, :], k) for I in c_idxs]
    Y_c_split = [np.array_split(Y[I], k) for I in c_idxs]

    X_split = [np.concatenate([X_i_split[split_idx] for X_i_split in X_c_split], axis=0) for split_idx in range(k)]
    Y_split = [np.concatenate([Y_i_split[split_idx] for Y_i_split in Y_c_split], axis=0) for split_idx in range(k)]

    datasets = []
    for split_idx in range(k):
        X_train = np.concatenate(X_split[:split_idx] + X_split[split_idx + 1:])
        X_val = X_split[split_idx]
        Y_train = np.concatenate(Y_split[:split_idx] + Y_split[split_idx + 1:])
        Y_val = Y_split[split_idx]
        datasets.append(((X_train, Y_train), (X_val, Y_val)))

    return datasets


def cross_validation(model, data: np.ndarray, labels: np.ndarray, dr_model: DrModel = NoDr(), k: int = 5) -> \
        Tuple[float, float]:
    # [((X_train, Y_train), (X_val, Y_val))]
    datasets = cross_val_split(data, labels, k)

    accuracies = np.zeros((k,))
    errors = np.zeros((k,))
    for dataset_idx in range(len(datasets)):
        dataset = datasets[dataset_idx]
        dr_model.fit(data=dataset[0][0], labels=dataset[0][1])
        data_train = dr_model.transform(dataset[0][0])
        data_val = dr_model.transform(dataset[1][0])

        model.fit(data=data_train, labels=dataset[0][1])
        Y_predicted = model.predict(data=data_val)

        accuracies[dataset_idx] = accuracy(dataset[1][1], Y_predicted)
        errors[dataset_idx] = error_rate(dataset[1][1], Y_predicted)

    return accuracies.mean(), errors.mean()


def run_configurations(data: np.ndarray, labels: np.ndarray, pca_dims: List[int], lda_dims: List[int], model,
                       cross_k: int):
    accuracies = []
    accuracies.append(cross_validation(model=model,
                                       data=data,
                                       labels=labels,
                                       k=cross_k)[0])
    for n_dims in pca_dims:
        accuracies.append(cross_validation(model=model,
                                           data=data,
                                           labels=labels,
                                           dr_model=PCA(n_dims),
                                           k=cross_k)[0])
    for n_dims in lda_dims:
        accuracies.append(cross_validation(model=model,
                                           data=data,
                                           labels=labels,
                                           dr_model=LDA(n_dims),
                                           k=cross_k)[0])
    print_table_format = f''
    for idx, acc in enumerate(accuracies):
        if idx != 0:
            print_table_format += ' & '
        print_table_format += f'{acc:.3f}'
    print(print_table_format)


if __name__ == '__main__':
    np.random.seed(420)
    data_train, labels_train = read_data('data/Train.txt')
    data_test, labels_test = read_data('data/Test.txt')

    mean = data_train.mean(0)
    stddev = data_train.std(0)
    data_train = (data_train - mean) / stddev
    data_test = (data_test - mean) / stddev

    # load Iris dataset for sanity checks
    # iris = datasets.load_iris()
    # data_train = iris.data
    # labels_train = iris.target
    # # reduce iris dataset to easy binary classification task
    # # idcs = labels_train != 2
    # idcs = labels_train != 0  # labels - 1!
    # data_train = data_train[idcs, :]
    # labels_train = labels_train[idcs]
    # labels_train -= 1

    # plot histogram for original attributes
    # plot_hist(data_train, labels_train)
    # plot_dims(data_train, labels_train)

    # get lower dimensional representation using principal component analysis
    # pca_components = 5
    # pca_model = PCA(pca_components)
    # pca_model.fit(data_train)
    # pca_data = pca_model.transform(data_train)

    # plot_scree(pca_model.eig_values)
    # print(f'PCA explained variance: {pca_model.explained_variance:.2f}')
    # plot_dims(pca_data, labels_train)

    # get lower dimensional representation using linear discriminant analysis
    # lda_components = 3
    # lda_model = LDA(lda_components)
    # lda_model.fit(data_train, labels_train)
    # lda_data_train = lda_model.transform(data_train)
    # lda_data_test = lda_model.transform(data_test)

    # plot_dims(lda_data, labels_train)

    # predict classes using multivariate gaussian classifier
    # mvg_model = MVG()
    # mvg_model.fit(lda_data_train, labels_train)
    # predicted = mvg_model.predict(lda_data_test)
    #
    # plot_dims(lda_data_test, predicted)
    # print(f'MVG confusion matrix:\n{confusion_matrix(labels_test, predicted)}')
    # print(f'MVG accuracy: {accuracy(labels_test, predicted):.3f}')
    # print(f'error rate:\n{error_rate(labels_test, predicted):.3f}')

    # classify data using logistic regression
    # lr_model = LogisticRegression(1e-6)
    # lr_model.fit(lda_data_train, labels_train)
    # predicted = lr_model.predict(lda_data_test)
    #
    # plot_dims(lda_data_test, predicted)
    # print(f'Logistic Regression confusion matrix:\n{confusion_matrix(labels_test, predicted)}')
    # print(f'Logistic Regression accuracy: {accuracy(labels_test, predicted):.3f}')
    # print(f'Logistic Regression error rate:{error_rate(labels_test, predicted):.3f}')

    # classify data using SVM
    svm_model = SVM(1)
    svm_data = svm_model.fit(data_train, labels_train)
    predicted = svm_model.predict_hat(data_train)

    # plot_dims(data_train, predicted)
    print(f'SVM confusion matrix:\n{confusion_matrix(labels_train, predicted)}')
    print(f'SVM accuracy: {accuracy(labels_train, predicted):.3f}')
    print(f'SVM error rate:{error_rate(labels_train, predicted):.3f}')

    # k = 5
    # cross_accuracy, cross_error = cross_validation(model=SVM(10, 10),
    #                                                data=data_train,
    #                                                labels=labels_train,
    #                                                dr_model=PCA(10),
    #                                                k=5)
    # print(f'{k}-fold cross validation resulting average {cross_accuracy:.2f} accuracy and {cross_error:.2f} error')

    # eval_dims = [3, 7]
    # run_configurations(data_train, labels_train, eval_dims, eval_dims, SVM(1), 5)
