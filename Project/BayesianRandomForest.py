from NaiveBayesClassifier import NaiveBayes
import numpy as np


class RandomForest:
    def __init__(self, n_estimators) -> None:
        self.clf = NaiveBayes()
        self.n_estimators = n_estimators

    @staticmethod
    def feature_bag(self, X):
        n_rows, n_cols = X.shape
        samples_cols = np.random.choice(
            a=n_cols, size=int(np.sqrt(n_cols)), replace=True)
        return X[:, samples_cols]

    @staticmethod
    def bag(self, X, y):
        n_rows, n_cols = X.shape
        samples_rows = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples_rows], y[samples_rows]

    def fit(self, X, y):
        for i in range(self.n_estimators):
            _X, _y = self.bag(X, y)
