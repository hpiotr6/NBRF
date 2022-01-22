from collections import defaultdict
from Project.NaiveBayesClassifier import NaiveBayes
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def data():
    data = pd.read_csv('data/agaricus-lepiota.data', header=None).to_numpy()
    # print(data[0:5])
    return data[:5]


@pytest.fixture
def X(data):
    _, X = np.split(data, [1], axis=1)
    return X


@pytest.fixture
def y(data):
    y, _ = np.split(data, [1], axis=1)
    return y


@pytest.fixture
def nbClassifier():
    return NaiveBayes()


def test_get_label_indices(y, nbClassifier):
    y_list = y.flatten().tolist()
    label_indices = nbClassifier.get_label_indices(y_list)
    assert len(label_indices) == 2
    assert type(label_indices) == defaultdict


def test_get_prior(y, nbClassifier):
    y_list = y.flatten().tolist()
    label_indices = nbClassifier.get_label_indices(y_list)
    prior = nbClassifier.get_prior(label_indices)
    print(prior)
    assert len(prior) == 2


def test_get_likelihood(y, X, nbClassifier):
    y_list = y.flatten().tolist()
    nbClassifier.get_likelihood(X, y_list)
