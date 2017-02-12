# -*- coding: utf-8 -*-

import scipy
import numpy
import sklearn
import random

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.metrics
import sklearn.utils


reviews_train_info = sklearn.datasets.load_files('./data/reviewdata/train')
reviews_dev_info = sklearn.datasets.load_files('./data/reviewdata/dev')

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf=True)
X_train = vectorizer.fit_transform(reviews_train_info.data)
y_train = reviews_train_info.target
X_dev = vectorizer.transform(reviews_dev_info.data)
y_dev = reviews_dev_info.target
index_to_word = vectorizer.get_feature_names()

ur, sigmar, wrt = scipy.sparse.linalg.svds(X_train, k=1425)
X_train_pca =  X_train.dot(numpy.transpose(wrt))
X_dev_pca = X_dev.dot(numpy.transpose(wrt))

_ = classifier.fit(X_train_pca, y_train)
pred = classifier.predict(X_dev_pca)
print(sklearn.metrics.accuracy_score(y_dev, pred))
print(sklearn.metrics.confusion_matrix(y_dev, pred))
print("Precision:", sklearn.metrics.precision_score(y_dev, pred))
print("Recall: ", sklearn.metrics.recall_score(y_dev, pred))
print("F1:", sklearn.metrics.f1_score(y_dev, pred))