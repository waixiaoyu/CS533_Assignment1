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

#A = lil_matrix((1000, 500))
#A[0, :100] = rand(100)
#A[1, 100:200] = A[0, :100]
#A.setdiag(rand(1000))
#
#ur, sigmar, wrt = scipy.sparse.linalg.svds(A,k=6)
#t = A.dot(numpy.transpose(wrt))


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

reviews_train_info = sklearn.datasets.load_files('./data/reviewdata/train')
reviews_dev_info = sklearn.datasets.load_files('./data/reviewdata/dev')
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf=True)
#vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_df=0.5,stop_words='english')
x_train = vectorizer.fit_transform(reviews_train_info.data)
y_train = reviews_train_info.target
x_dev = vectorizer.transform(reviews_dev_info.data)
y_dev = reviews_dev_info.target
index_to_word = vectorizer.get_feature_names()
ur, sigmar, wrt = scipy.sparse.linalg.svds(x_train,k=1450)

t_train=x_train.dot(numpy.transpose(wrt))
t_dev=x_dev.dot(numpy.transpose(wrt))

classifier = sklearn.linear_model.SGDClassifier(loss="log",penalty="elasticnet",n_iter=5)
_ = classifier.fit(t_train, y_train)

pred = classifier.predict(t_dev)

print "Accuracy:",sklearn.metrics.accuracy_score(y_dev, pred)
print sklearn.metrics.confusion_matrix(y_dev, pred)
print "Precision:", sklearn.metrics.precision_score(y_dev, pred)
print "Recall: ", sklearn.metrics.recall_score(y_dev, pred)
print "F1:", sklearn.metrics.f1_score(y_dev, pred)