import sklearn
import numpy
import random

import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.metrics
import sklearn.utils

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

reviews_train_info = sklearn.datasets.load_files('D:/pythonworkspace/CS533/reviewdata/train')
reviews_dev_info = sklearn.datasets.load_files('D:/pythonworkspace/CS533/reviewdata/dev')

#vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')

vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_df=0.5,stop_words='english')


X_train = vectorizer.fit_transform(reviews_train_info.data)
y_train = reviews_train_info.target
X_dev = vectorizer.transform(reviews_dev_info.data)
y_dev = reviews_dev_info.target
index_to_word = vectorizer.get_feature_names()

#classifier = sklearn.linear_model.SGDClassifier(loss="log",penalty="elasticnet",n_iter=5)

classifier=sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)

_ = classifier.fit(X_train, y_train)

pred = classifier.predict(X_dev)

print sklearn.metrics.accuracy_score(y_dev, pred)

print sklearn.metrics.confusion_matrix(y_dev, pred)

print "Precision:", sklearn.metrics.precision_score(y_dev, pred)
print "Recall: ", sklearn.metrics.recall_score(y_dev, pred)
print "F1:", sklearn.metrics.f1_score(y_dev, pred)

ppred = classifier.predict_proba(X_dev)[:,1]
fpr, tpr, _ = sklearn.metrics.roc_curve(y_dev, ppred)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label='ROC curve (area = %0.2f)' % sklearn.metrics.auc(fpr, tpr))
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")


#classifier2 = sklearn.naive_bayes.BernoulliNB()
#
#_ = classifier2.fit(X_train, y_train)
#pred2 = classifier2.predict(X_dev)
#print sklearn.metrics.accuracy_score(y_dev, pred2)