# Naive Bayes Random Forest Classifier

The project is a Random Forest from-scratch implementation, which relies on a Naive Bayes classifiers instead of decision trees.

Project covers:
* Random Forest implementation
* Naive Bayes implemenatation
* Comparison between `sklearn` Random Forest and own implementation

More information was reported in the polish language [here](https://github.com/hpiotr6/Naive-Bayes-Random-Forest/blob/main/21ZUMA_sprawozdanie_koncowe_Hondra_Groszyk.pdf).



## Installation

1. Install `poetry`: https://python-poetry.org/docs/#installation
2. Create an environment with `poetry install`
3. Run `poetry shell`
4. To run unit tests for your service use `poetry run pytest` or simply `pytest` within `poetry shell`.
 

## Usage

```python
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
```
Import dependencies
```python
%load_ext autoreload
%autoreload 2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from Project.NaiveBayesClassifier import NaiveBayes
from Project.BayesianRandomForest import RandomForest
from sklearn.model_selection import train_test_split
```
Read data
```python
data = pd.read_csv('../data/agaricus-lepiota.data', header=None).to_numpy()
y, X = np.split(data, [1], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y)
```
Preprocess
```python
y_list = y_train.reshape(-1)
enc = OneHotEncoder()
X_train_enc = enc.fit_transform(X_train)
X_train_enc = X_train_enc.toarray()
X_test_enc = enc.transform(X_test)
X_test_enc = X_test_enc.toarray()
```
Fit and predict
```python
rf_classifier = RandomForest(100, feature_bagging=False)
rf_classifier.fit(X_train_enc, y_list)
result = rf_classifier.predict(X_test_enc)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)



