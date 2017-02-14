# -*- coding: utf-8 -*-
#question 1
import nltk
from nltk.corpus import brown

def brown_sentence_items() :
    for sent in brown.tagged_sents(tagset='universal') :
        yield ('START', 'START')
        for (word, tag) in sent :
            yield (tag, word)
        yield ('END', 'END')
        

class Experiment(object) :
    pass

expt1 = Experiment()
expt1.cfd_tagwords = nltk.ConditionalFreqDist(brown_sentence_items())
expt1.cpd_tagwords = nltk.ConditionalProbDist(expt1.cfd_tagwords, nltk.MLEProbDist)

#print "The probability of an adjective (NOUN) being 'huge' is", expt1.cpd_tagwords["NOUN"].prob("huge")
expt1.cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams((tag for (tag, word) in brown_sentence_items())))
expt1.cpd_tags = nltk.ConditionalProbDist(expt1.cfd_tags, nltk.MLEProbDist)
expt1.tagset = set((tag for (tag, word) in brown_sentence_items()))

#print "If we have just seen 'DET', the probability of 'X' is", expt1.cpd_tags["DET"].prob("X")

prob_tagsequence = expt1.cpd_tags["START"].prob("PRON") * expt1.cpd_tagwords["PRON"].prob("I") * \
    expt1.cpd_tags["PRON"].prob("VERB") * expt1.cpd_tagwords["VERB"].prob("want") * \
    expt1.cpd_tags["VERB"].prob("PRT") * expt1.cpd_tagwords["PRT"].prob("to") * \
    expt1.cpd_tags["PRT"].prob("VERB") * expt1.cpd_tagwords["VERB"].prob("race") * \
    expt1.cpd_tags["VERB"].prob("END")

#print "The probability of the tag sequence 'START PRON VERB PRT VERB END' for 'I want to race' is:", prob_tagsequence


def tsp(expt, ts):
    prob=1
    last_tag=''
    try:
        while True:
            t_w=next(ts)
            if last_tag=='':
                prob=prob*expt.cpd_tagwords[t_w[0]].prob(t_w[1])
            else:
                prob=prob * \
                expt.cpd_tags[last_tag].prob(t_w[0]) * \
                expt.cpd_tagwords[t_w[0]].prob(t_w[1])
            last_tag=t_w[0]
    except StopIteration:
        return prob

sentence=[('PRON', 'I'), ('VERB', 'want'),('PRT', 'to'), ('VERB', 'race')]
prob=tsp(expt1,iter(sentence))
print prob