import numpy as np
from scipy.special import logsumexp
from scipy.optimize import fmin_l_bfgs_b


def loglikelihood(mu: np.ndarray, sigma: np.ndarray, X: np.ndarray) -> np.ndarray:
    _, logdet = np.linalg.slogdet(sigma)
    exp = ((X - mu) @ np.linalg.inv(sigma) * (X - mu)).sum(1)
    return - 1 / 2 * (mu.size * np.log(2 * np.pi) + logdet + exp)


class MVG:
    classes: np.ndarray
    num_classes: int

    p_c: list[float]
    mu_c: list[np.ndarray]
    sig_c: list[np.ndarray]

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)

        D_c = [data[labels == c, :] for c in self.classes]
        self.p_c = [D_c[c_idx].shape[0] / data.shape[0] for c_idx in range(self.num_classes)]
        self.mu_c = [D.mean(0) for D in D_c]
        self.sig_c = [np.cov((D_c[c_idx] - self.mu_c[c_idx]).T, ddof=0) for c_idx in range(self.num_classes)]

    def predict(self, data: np.ndarray) -> np.ndarray:
        joint = np.zeros([data.shape[0], self.num_classes])
        for c_idx in range(self.num_classes):
            joint[:, c_idx] = loglikelihood(self.mu_c[c_idx], self.sig_c[c_idx], data) + np.log(self.p_c[c_idx])
        probabilities = joint - np.reshape(logsumexp(joint, 1), (joint.shape[0], 1))

        return self.classes[probabilities.argmax(1)]


class LogisticRegression:
    _w: np.ndarray
    _b: float

    def __init__(self, L: float):
        self.L = L

    def fit(self, data: np.ndarray, labels: np.ndarray):
        D = data.shape[1]
        wb, _, _ = fmin_l_bfgs_b(lambda wb: self.obj_func(wb=wb, X=data, Y=labels),
                                 x0=np.zeros((D + 1, 1)),
                                 approx_grad=True)
        self._w = wb[:D]
        self._b = wb[D]

    def obj_func(self, wb: np.ndarray, X: np.ndarray, Y: np.ndarray):
        [N, D] = X.shape
        w = wb[:D]
        b = wb[D]

        z = 2 * Y - 1
        exponent = -z * (X @ w + b)
        sum_factor = np.sum(np.log1p(np.exp(exponent))) / N
        regularization = self.L / 2 * w.T @ w

        loss = regularization + sum_factor
        return loss

    def predict(self, data: np.ndarray) -> np.ndarray:
        predictions = np.zeros((data.shape[0],), dtype=np.int32)
        predictions[data @ self._w + self._b > 0] = 1
        return predictions
