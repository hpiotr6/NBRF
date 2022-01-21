# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
from collections import defaultdict
# import math
# import os

# def get_label_indices(y):
#     label_indices = defaultdict(list)
#     for idx, label in enumerate(y):
#         label_indices[label].append(idx)
#     return label_indices

# def get_prior(label_indices):
#     prior = {label: len(indices) for label, indices in label_indices.items()}
#     total = sum(prior.values())
#     for label in prior:
#         prior[label] /= total
#     return prior

# def get_likelihood(x, gausian_params, label_indices):
#     likelihoods = []
#     for x in X_test.to_numpy():
#         likelihood = {}
#         for label, indices in label_indices.items():
#             likelihood[label] = 0
#             # gausian_params[label] = [f1,f2,f3]
#             for feature, (mean, std) in zip(list(x), list(gausian_params[label])):
#                 likelihood[label] += math.log(
#                     gaussian_formula(std, mean, feature) + math.pow(10, -10)
#                 )
#         likelihoods.append(likelihood)
#     return likelihoods

# def get_posterior(prior, likelihood):
#     post = {}
#     for label in prior.keys():
#         post[label] = math.log(prior[label]) + likelihood[label]
#     return post

# def predict(posterior: dict):
#     return max(posterior, key=posterior.get)


class NaiveBayes:

    def __init__(self) -> None:
        pass

    def get_label_indices(self, y):
        label_indices = defaultdict(list)
        for idx, label in enumerate(y):
            label_indices[label].append(idx)
        return label_indices

    def get_prior(self, label_indices):
        prior = {
            label: len(indices)
            for label, indices in label_indices.items()
        }
        total = sum(prior.values())
        for label in prior:
            prior[label] /= total
        return prior
