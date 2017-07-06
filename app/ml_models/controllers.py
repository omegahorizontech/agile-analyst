from flask import jsonify
import requests, operator, math, json, csv, os
import matplotlib.pyplot as plt
import affect_ai

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import learning_curve

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer

from sklearn.externals import joblib

import pandas as pd
import numpy as np

from collections import Counter
import time

def hello():
    return 'hello machines'

enc_samples = []
y = []
# Read data from the csv
# Place it into a dataframe
def prepare_data(as_generator=False, text_encoder=False):
    filename = 'austen-sense-400-1N1S1L2U2L(2017-01-07 23:41:04.780)'
    data = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename + '.csv')
    X = pd.DataFrame(data['sample_doc'])
    y = pd.DataFrame(data[data.columns[1:400]])
    # print 'before: ',X.shape

    filename2 = 'belles_lettres-brown-400-1N1S1L2U2L(2017-01-17 17:30:39.071)'
    data2 = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename2 + '.csv')
    A = pd.DataFrame(data2['sample_doc'])
    b = pd.DataFrame(data2[data2.columns[1:400]])
    # print A, b

    filename3 = 'learned-brown-400-1N1S1L2U2L(2017-01-15 20:45:24.877)'
    data3 = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename3 + '.csv')
    C = pd.DataFrame(data3['sample_doc'])
    d = pd.DataFrame(data3[data3.columns[1:400]])

    filename4 = 'news-brown-400-1N1S1L2U2L(2017-01-14 17:57:42.849)'
    data4 = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename4 + '.csv')
    E = pd.DataFrame(data4['sample_doc'])
    f = pd.DataFrame(data4[data4.columns[1:400]])

    filenames = []
    # filenames.append('10-19-20s_706posts-400-1N1S1L2U2L(2017-01-11 20:56:06.286)')
    # filenames.append('10-19-30s_705posts-400-1N1S1L2U2L(2017-01-11 20:56:06.286)')
    # filenames.append('10-19-40s_686posts-400-1N1S1L2U2L(2017-01-11 20:56:06.286)')
    filenames.append('10-19-adults_706posts-400-1N1S1L2U2L(2017-01-11 20:56:06.286)')
    filenames.append('editorial-brown-400-1N1S1L2U2L(2017-01-17 17:30:39.071)')
    # filenames.append('milton-paradise-400-1N1S1L2U2L(2017-01-09 16:08:02.377)')
    filenames.append('shakespeare-macbeth-400-1N1S1L2U2L(2017-01-09 16:08:02.377)')

    for filename in filenames:
        data = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename + '.csv')
        Input = pd.DataFrame(data['sample_doc'])
        Output = pd.DataFrame(data[data.columns[1:400]])
        X = X.append(Input, ignore_index=True)
        y = y.append(Output, ignore_index=True)



    X = X.append(A, ignore_index=True)
    y = y.append(b, ignore_index=True)
    X = X.append(C, ignore_index=True)
    y = y.append(d, ignore_index=True)
    X = X.append(E, ignore_index=True)
    y = y.append(f, ignore_index=True)

    # print 'after: ',X.shape
    scaler = StandardScaler(with_mean=False)
    output_scaler = StandardScaler(with_mean=False)
    le = LabelEncoder()
    enc = OneHotEncoder()

    X_array = np.ravel(X)

    # Initialize DictVectorizer and an empty sparse_matrix to store vectors
    vectorizer = DictVectorizer(sparse=False)
    sparse_matrix = []
    # Find unique words, use them as encoding labels
    for sample in X_array:
        words = sample.split(' ')
        counted_sample = Counter()
        counted_sample.update(words)
        sparse_matrix.append(dict(counted_sample))

    # X is the list of 'fit_transformed' vectors
    # print sparse_matrix
    X = vectorizer.fit_transform(sparse_matrix)
    X = scaler.fit_transform(X)
    y = output_scaler.fit_transform(y)
    y = pd.DataFrame(y)
    # for datapoint in X:
    #     print max(datapoint)
    # print X
    # TODO: Return encoded data and labels
    if not as_generator:
        return 'data prepared'
    else:
        if text_encoder:
            return X, y, vectorizer, scaler, output_scaler
        return X, y

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    print test_scores_mean
    plt.legend(loc="best")
    return plt

# Do a shufflesplit or other cross-validation
# Train a classifier on the data and labels
def train_model():

    # TODO: Use real corpus data to train the affect_ai.
    affect_vocab = {}
    affect_weights = {}
    title = 'affect_ai.pkl'

    aff_ai = affect_ai.affect_AI()
    aff_ai.train(affect_vocab, affect_weights)
    joblib.dump(estimator, title)

# Measure model performance with CV
def validate_model():
    # # The vectorizer we retrieve here must be the same as that used to train the model we're validating.
    # _,_,vectorizer,scaler,output_scaler = prepare_data(True, True)
    #
    # # Retrieve a model from a .pkl file with joblib.load()
    # # title = '(DTR(9 depth, MSE, 0.24 features, random splits, min_samples_split 0.027, min_samples_leaf .018, presort)+MOR, 24.5k samples, 3 columns).pkl'
    # estimator = joblib.load(title)
    #
    # # Use an unseen dataset to score it
    # filename = 'science_fiction-brown-400-1N1S1L2U2L(2017-01-17 17:30:39.071)'
    # data = pd.read_csv(os.path.dirname(__file__) + '/../../data/' + filename + '.csv')
    # X = pd.DataFrame(data['sample_doc'])
    # y = pd.DataFrame(data[data.columns[1:400]])
    # X_array = np.ravel(X)
    # sparse_matrix = []
    #
    # for sample in X_array:
    #     words = sample.split(' ')
    #     counted_sample = Counter()
    #     counted_sample.update(words)
    #     sparse_matrix.append(dict(counted_sample))
    #
    # X = vectorizer.transform(sparse_matrix)
    # X = scaler.transform(X)
    # y = output_scaler.transform(y)
    # y = pd.DataFrame(y)
    # y = y[y.columns[30:33]]
    # # Measure amount of time for predictions
    # t_1 = time.clock()
    # score = estimator.score(X,y)
    # t_2 = time.clock()
    # print 'model score: ', score, 'time required for',y.shape,'predictions: ',t_2-t_1
    # # predictions = estimator.predict(X)
    # # for prediction in range(1,len(predictions)):
    # #     print X[prediction]
    # #     print 'predicted', predictions[prediction], 'actual', y.iloc[prediction]

    # Use data scored by ample affect to gauge accuracy of data scored by affect_ai.
    return 'validating model'
