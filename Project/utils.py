import pandas as pd
from pathlib import Path
import numpy as np

data = pd.read_csv('data/agaricus-lepiota.data', header=None).to_numpy()

y, X = np.split(data, [1], axis=1)
print(y.shape)
print(X.shape)
