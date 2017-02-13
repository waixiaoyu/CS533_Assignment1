import os
import nltk
import re
import itertools
import vocabulary
import numpy as np
import scipy
import sklearn

#how many words missing from GloVe
words = vocabulary.Vocabulary.from_iterable(w.lower() for w in nltk.corpus.brown.words())
words.stop_growth()

def build_embedding(vocab) :
    remaining_vocab = vocab.keyset()
    embeddings = np.zeros((len(remaining_vocab), 50))
    
    with open("./glove6B50/glove.6B.50d.txt") as glovefile :
        fileiter = glovefile.readlines()
        count = 0
        
        for line in fileiter :
            line = line.replace("\n","").split(" ")
            try:
                word, nums = line[0], [float(x.strip()) for x in line[1:]]
                if word in remaining_vocab:
                    embeddings[vocab[word]]  = np.array(nums)
                    remaining_vocab.remove(word)
            except Exception as e:
                print("{} broke. exception: {}. line: {}.".format(word, e, x))
            count+=1

        print("{} words were not in glove".format(len(remaining_vocab)))
        return embeddings

e = build_embedding(words)



def tagged_lc_fivegrams(seq) :
    n2b, n1b = "START", "START"
    (w, t) = next(seq)
    w = w.lower()
    n1a = "END"    
    try:
        (n1a, t1a) = next(seq)
        n1a = n1a.lower()
    except StopIteration:#this sentence might have only one word
        yield ((n2b, n1b, w, "END", "END"), t)
        return
    for (item, nt) in seq :
        item = item.lower()
        yield ((n2b, n1b, w, n1a, item), t)
        (n2b, n1b, w, t, n1a, t1a) = (n1b, w, n1a, t1a, item, nt)
    yield ((n2b, n1b, w, n1a, "END"), t)
    yield ((n1b, w, n1a, "END", "END"), t1a)

def lc_fivegrams(seq) :
    n2b, n1b = "START", "START"
    w = next(seq)
    w = w.lower()
    n1a = "END"    
    try:
        n1a = next(seq)
        n1a = n1a.lower()
    except StopIteration:
        yield (n2b, n1b, w, "END", "END")
        return
    for item in seq :
        item = item.lower()
        yield (n2b, n1b, w, n1a, item)
        (n2b, n1b, w, n1a) = (n1b, w, n1a, item)
    yield (n2b, n1b, w, n1a, "END")
    yield (n1b, w, n1a, "END", "END")
    
#represent each window using appropriate features
features = vocabulary.Vocabulary()

def mkf(features, name, fl) :
    r = features.add(name)
    if r :
        fl.append(r)
    
def word_feature_columns(features, code, item) :
    f = []
    for i in range(0,50) :
        mkf(features, "{}:e{}".format(code, i), f)
    mkf(features, "{}:w_{}".format(code, item), f)
    for i in range(1,4) :
        mkf(features, "{}:{}_{}".format(code, i, item[-i:]), f)
    for i in range(1,4) :
        mkf(features, "{}:{}{}_".format(code, i, item[0:i]), f)
    return f

def word_feature_values(embeddings, vocab, item, f) :
    values = np.zeros(len(f))
    r = vocab.add(item) 
    if r: 
        values[:50] = embeddings[r]
    values[50:] = np.ones(len(f)-50)
    return values

def fivegram_features(features, embeddings, vocab, cxt) :
    (n2b, n1b, t, n1a, n2a) = cxt
    f2b = word_feature_columns(features, "w_t", n2b)
    v2b = word_feature_values(embeddings, vocab, n2b, f2b)
    f1b = word_feature_columns(features, "wt", n1b)
    v1b = word_feature_values(embeddings, vocab, n1b, f1b)
    ft = word_feature_columns(features, "t", t)
    vt = word_feature_values(embeddings, vocab, t, ft)
    f1a = word_feature_columns(features, "tw", n1a)
    v1a = word_feature_values(embeddings, vocab, n1a, f1a)
    f2a = word_feature_columns(features, "t_w", n2a)
    v2a = word_feature_values(embeddings, vocab, n2a, f2a)
    return np.concatenate([f2b, f1b, ft, f1a, f2a]), np.concatenate([v2b, v1b, vt, v1a, v2a])



dev_items = 123
test_items = 500
train_items = 4000

#The data is loaded incrementally into a coordinate format matrix and 
#then converted into column sparse row format for training.
def mk_tagging_data(features, embeddings, vocab, start, end) :
    row = 0
    rows = []
    columns = []
    values = []
    tags = []
#islice('ABCDEFG', 2, None) --> C D E F G, jump over the first 123+500
#s is a sentence
    for s in itertools.islice(nltk.corpus.brown.tagged_sents(categories='news',tagset='universal'), start, end) :
        for (cxt, tag) in tagged_lc_fivegrams(iter(s)) :
            #cxt are five words, (n2b, n1b, w, n1a, n2a, tag) is the tag of w.
            f, v = fivegram_features(features, embeddings, vocab, cxt)
            columns.append(f)
            values.append(v)
            rows.append(np.full(v.shape, row))
            tags.append(tag)
            row = row + 1
    return scipy.sparse.coo_matrix((np.concatenate(values), 
                                    (np.concatenate(rows), 
                                     np.concatenate(columns))),  
                                   shape=(row,len(features))).tocsr(), np.array(tags)
    
    
#Build the training data and fix the features we're going to be using.
Xtr, ytr = mk_tagging_data(features, e, words, dev_items + test_items, None)
features.stop_growth()

Xtr.shape

#We build a classifier as always

classifier = sklearn.linear_model.SGDClassifier(loss="log",
                                   penalty="elasticnet",
                                   n_iter=5)

classifier.fit(Xtr, ytr)


def test_tagger(c, features, embeddings, vocab, s) :
    row = 0
    rows = []
    columns = []
    values = []
    words = nltk.word_tokenize(s)
    for cxt in lc_fivegrams(iter(words)) :
        f, v = fivegram_features(features, embeddings, vocab, cxt)
        columns.append(f)
        values.append(v)
        rows.append(np.full(v.shape, row))
        row = row + 1
    data = scipy.sparse.coo_matrix((np.concatenate(values), 
                                    (np.concatenate(rows), 
                                     np.concatenate(columns))),  
                                   shape=(row,len(features))).tocsr()
    pd = c.predict(data)
    return zip(words, pd)


#==============================================================================
# e:building embedding
#==============================================================================
Xd, yd = mk_tagging_data(features, e, words, 0, dev_items)
pd = classifier.predict(Xd)
print sklearn.metrics.accuracy_score(yd, pd)

test_tagger(classifier, features, e, words, "they refuse to permit us to obtain the refuse permit")
