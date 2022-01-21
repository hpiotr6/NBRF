import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def data():
    data = pd.read_csv('data/agaricus-lepiota.data', header=None).to_numpy()
    return data


@pytest.fixture
def X_y(data):
    y, X = np.split(data, [1], axis=1)
    return X, y


def test_init():
    assert True
