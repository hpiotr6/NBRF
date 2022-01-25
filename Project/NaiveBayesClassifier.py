import numpy as np
from collections import defaultdict


class NaiveBayes:
    def __init__(self) -> None:
        self.prior = None
        self.likelihood = None

    def get_label_indices(self, y):
        label_indices = defaultdict(list)
        for idx, label in enumerate(y):
            label_indices[label].append(idx)
        return label_indices

    def get_prior(self, label_indices):
        prior = {label: len(indices)
                 for label, indices in label_indices.items()}
        total = sum(prior.values())
        for label in prior:
            prior[label] /= total
        return prior

    def get_likelihood(self, X, y, smoothing=1):
        label_indices = self.get_label_indices(y)
        likelihood = {}
        for label, indices in label_indices.items():
            likelihood[label] = X[indices, :].sum(axis=0)
            + smoothing
            total_count = len(indices)
            likelihood[label] = likelihood[label] / \
                (total_count + 2 * smoothing)
        return likelihood

    def get_posterior(self, X, prior, likelihood):
        posteriors = []
        for x in X:
            posterior = prior.copy()
            for label, likelihood_label in likelihood.items():
                for index, bool_value in enumerate(x):
                    posterior[label] *= likelihood_label[index] if bool_value else (
                        1 - likelihood_label[index])
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
            posteriors.append(posterior.copy())
        return posteriors

    def fit(self, X, y):
        label_indices = self.get_label_indices(y)
        self.prior = self.get_prior(label_indices)
        self.likelihood = self.get_likelihood(X, y)

    def predict(self, X):
        if self.prior is None or self.likelihood is None:
            raise ValueError("Fit classifier first to use predict")
        posterior = self.get_posterior(X, self.prior, self.likelihood)
        result = [max(prediction, key=prediction.get)
                  for prediction in posterior]
        return np.array(result)

    # def predict_proba(self, X):
    #     if self.prior is None or self.likelihood is None:
    #         raise ValueError("Fit classifier first to use predict")
    #     posterior = self.get_posterior(X, self.prior, self.likelihood)
    #     return posterior
