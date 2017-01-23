from flask import jsonify
import requests, operator, math, json, csv, os

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score as CVS

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np

from collections import Counter

def hello():
    return 'hello machines'

# Read data from the csv
# Place it into a dataframe
def prepareData():
    filename = 'austen-sense-400-1N1S1L2U2L(2017-01-07 23:41:04.780)'
    data = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename + '.csv')
    X = pd.DataFrame(data['sample_doc'])
    y = pd.DataFrame(data[data.columns[1:400]])

    le = LabelEncoder()
    enc = OneHotEncoder()

    X_array = np.ravel(X)
    counter = Counter()
    for sample in X_array:
        # TODO: Find unique words, use them as encoding labels
        # counter.update(sample)
        print sample

    # TODO: Use the encoded labels to one-hot encode each sample
    encoded = le.fit_transform(X_array)
    encoded = np.reshape(encoded,(-1,1))
    onehotlabels = enc.fit_transform(encoded).toarray()
    print onehotlabels.shape

    # TODO: Return one-hot encoded data and labels
    return 'Data prepared'

# Do a shufflesplit or other cross-validation
# Train a classifier on the data and labels
def trainModel():
    # data = prepareData()
    # data = data[:400]
    split = ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)
    # Perform one-hot encoding on string data
    groups = split.split(X, y)


    reg = DTR()
    scorer = CVS(reg, onehotlabels, y, cv=split)
    for score in scorer:
        print score

    return 'training model'

# Measure model performance with CV
def validateModel():
    return 'validating model'
