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

reviews_train_info = sklearn.datasets.load_files('./data/reviewdata/train')
reviews_dev_info = sklearn.datasets.load_files('./data/reviewdata/dev')

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')

#vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_df=0.5,stop_words='english')


X_train = vectorizer.fit_transform(reviews_train_info.data)
y_train = reviews_train_info.target
X_dev = vectorizer.transform(reviews_dev_info.data)
y_dev = reviews_dev_info.target
index_to_word = vectorizer.get_feature_names()

classifier = sklearn.linear_model.SGDClassifier(loss="log",penalty="elasticnet",n_iter=5)



_ = classifier.fit(X_train, y_train)

pred = classifier.predict(X_dev)

print "Accuracy:",sklearn.metrics.accuracy_score(y_dev, pred)

print sklearn.metrics.confusion_matrix(y_dev, pred)

print "Precision:", sklearn.metrics.precision_score(y_dev, pred)
print "Recall: ", sklearn.metrics.recall_score(y_dev, pred)
print "F1:", sklearn.metrics.f1_score(y_dev, pred)

#ppred = classifier.predict_proba(X_dev)[:,1]
#fpr, tpr, _ = sklearn.metrics.roc_curve(y_dev, ppred)
#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, 
#         label='ROC curve (area = %0.2f)' % sklearn.metrics.auc(fpr, tpr))
#plt.xlim([-0.05, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend(loc="lower right")





#classifier2 = sklearn.naive_bayes.BernoulliNB()

#classifier2=sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
#_ = classifier2.fit(X_train, y_train)
#pred2 = classifier2.predict(X_dev)
#print sklearn.metrics.accuracy_score(y_dev, pred2)
#
#
#
#
#def sim_expt(test, p1, p2):
#    bits = [random.getrandbits(1) for i in range(0, len(p1))]
#    return test([p1[i] if bits[i] else p2[i] for i in range(0,len(p1))],
#                [p2[i] if bits[i] else p1[i] for i in range(0,len(p1))])
#
#def eval_diff(data, p1, p2) :
#    diff = abs(sklearn.metrics.accuracy_score(data, p1) -
#               sklearn.metrics.accuracy_score(data, p2))
#    def test_diff(n, m) :
#        return (abs(sklearn.metrics.accuracy_score(data, n) - 
#                    sklearn.metrics.accuracy_score(data, m)) >= 
#                diff)
#    return test_diff
#
#def mcmcp_diff(data, p1, p2, k) :
#    success = 0
#    test = eval_diff(data, p1, p2)
#    for i in range(0,k) :
#        success += sim_expt(test, p1, p2)
#    return success / float(k)
#
#print mcmcp_diff(y_dev, pred, pred2, 10000)
