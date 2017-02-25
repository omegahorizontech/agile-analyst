from flask import jsonify
import requests, operator, math, json, csv, os
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.linear_model import ElasticNet as EN

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import learning_curve

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import DictVectorizer

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
    for sample in X_array:
        words = sample.split(' ')
        counted_sample = Counter()
        counted_sample.update(words)
        sparse_matrix.append(dict(counted_sample))

    # X is the list of 'fit_transformed' vectors
    X = vectorizer.fit_transform(sparse_matrix)
    # print X
    # TODO: Return encoded data and labels
    if not as_generator:
        return 'data prepared'
    else:
        return X, y


# Do a shufflesplit or other cross-validation
# Train a classifier on the data and labels
def train_model():
    X,y = prepare_data(True)
    y = y[y.columns[3]]
    print 'this should be y', y
    # print X[:1000]
    split = ShuffleSplit(n_splits=1, test_size=0.03, random_state=42)
    t_1 = time.clock()
    estimator = DTR()
    estimator2 = RFR()
    estimator3 = RFR(n_estimators=40, n_jobs=-1, random_state=21)

    estimator4 = ETR(n_estimators=100, max_features=0.66, random_state=12, n_jobs=-1, bootstrap=True)

    estimator5 = SVR(kernel='poly')

    estimator6 = SGDR(learning_rate='optimal', n_iter=12)

    estimator7 = EN()

    # scorer = CVS(reg, X[:1000], y[:1000], cv=split, pre_dispatch=3)
    # for score in scorer:
        # print score


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

        plt.legend(loc="best")
        return plt

    # title = "Learning Curves (RFR)"
    # plot_learning_curve(estimator2, title, X[:2000], y[:2000], (0.1, 1.01), split, n_jobs=1)
    # plt.show()

    title = "Learning Curves (Stock SVR, 2k samples)"
    plot_learning_curve(estimator6, title, X[:2000], y[:2000], (0.1, 1.01), split, n_jobs=-1)
    plt.show()

    t_2 = time.clock()
    print "Total time: ", t_2-t_1
    return 'training model'

# Measure model performance with CV
def validate_model():
    return 'validating model'
