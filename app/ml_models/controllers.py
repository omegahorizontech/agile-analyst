from flask import jsonify
import requests, operator, math, json, csv, os

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score as CVS

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import DictVectorizer

import pandas as pd
import numpy as np

from collections import Counter

def hello():
    return 'hello machines'

enc_samples = []
y = []
# Read data from the csv
# Place it into a dataframe
def prepare_data(as_generator=False):
    filename = 'austen-sense-400-1N1S1L2U2L(2017-01-07 23:41:04.780)'
    data = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename + '.csv')
    X = pd.DataFrame(data['sample_doc'])
    y = pd.DataFrame(data[data.columns[1:400]])

    le = LabelEncoder()
    enc = OneHotEncoder()

    X_array = np.ravel(X)

    # Initialize DictVectorizer and an empty sparse_matrix to store vectors
    vectorizer = DictVectorizer(sparse=False)
    sparse_matrix = []
    # Find unique words, use them as encoding labels
    for sample in X_array[17:20]:
        words = sample.split(' ')
        counted_sample = Counter()
        counted_sample.update(words)
        sparse_matrix.append(dict(counted_sample))

    # X is the list of 'fit_transformed' vectors
    X = vectorizer.fit_transform(sparse_matrix)

    # TODO: Return encoded data and labels
    if not as_generator:
        return 'data prepared'
    else:
        return X, y


# Do a shufflesplit or other cross-validation
# Train a classifier on the data and labels
def train_model():
    enc_samples = prepare_data(True)
    # enc_samples = np.reshape(enc_samples,(-1,1))
    print enc_samples

    # data = data[:400]
    split = ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)


    reg = DTR()
    scorer = CVS(reg, enc_samples, y, cv=split)
    for score in scorer:
        print score

    return 'training model'

# Measure model performance with CV
def validate_model():
    return 'validating model'
