from flask import jsonify
import requests, operator, math, json, csv, os

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score as CVS
import pandas as pd
import numpy as np

def hello():
    return 'hello machines'

# Read data from the csv
# Place it into a dataframe
def prepareData():
    filename = 'austen-sense-400-1N1S1L2U2L(2017-01-07 23:41:04.780)'
    data = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename + '.csv')
    return data

# Do a shufflesplit or other cross-validation
# Train a classifier on the data and labels
def trainModel():
    data = prepareData()
    split = ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)
    # Perform one-hot encoding on string data 
    X = pd.DataFrame(data['sample_doc'])
    y = pd.DataFrame(data[data.columns[1:400]])
    groups = split.split(X, y)

    reg = DTR()
    scorer = CVS(reg, X, y, cv=split)
    print scorer

    return 'training model'

# Measure model performance with CV
def validateModel():
    return 'validating model'
