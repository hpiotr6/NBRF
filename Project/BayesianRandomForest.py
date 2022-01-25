from .NaiveBayesClassifier import NaiveBayes
import numpy as np


class RandomForest:
    def __init__(self, n_estimators, feature_bagging=False) -> None:
        self.clf = NaiveBayes
        self.n_estimators = n_estimators
        self.feature_bagging = feature_bagging
        self._clfs = []

    @staticmethod
    def feature_bag(X, y):
        n_rows, n_cols = X.shape
        samples_cols = np.random.choice(
            a=n_cols, size=int(np.sqrt(n_cols)), replace=True)
        return X[:, samples_cols], y

    @staticmethod
    def bag(X, y):
        n_rows, n_cols = X.shape
        samples_rows = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples_rows], y[samples_rows]

    def fit(self, X, y):
        clfs = []
        for _ in range(self.n_estimators):
            clf = self.clf()
            _X, _y = self.bag(X, y)
            clf.fit(_X, _y)
            clfs.append(clf)
        self.clfs = clfs
        return self.clfs

    def predict(self, X):
        for clf in self.clfs:
            prediction = clf.predict(X)
