# -*- coding: utf-8 -*-
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






#viterbi algorithm
def viterbi_init(expt, word) :
    prob = {}
    back = {}
    for tag in expt.tagset :
        if tag == 'START' :
            continue
        prob[ tag ] = expt.cpd_tags['START'].prob(tag) * expt.cpd_tagwords[tag].prob( word )
        back[ tag ] = 'START'
    return (prob, back)

def viterbi_extend(expt, prev_prob, word) :
    print prev_prob
    prob = {}
    back = {}
    for tag in expt.tagset :
        if tag == 'START' :
            continue
#        we want to find the max(P(t-1)*P(t|t-1))
        best_previous = max(prev_prob.keys(), key = lambda prevtag:prev_prob[prevtag] * expt.cpd_tags[prevtag].prob(tag))
                          
        prob[tag] = prev_prob[best_previous] * expt.cpd_tags[best_previous].prob(tag) * expt.cpd_tagwords[tag].prob(word)
        back[tag] = best_previous
        print tag,best_previous
    return (prob, back)

def viterbi_run(expt, sentence) :
    (prob, back) = viterbi_init(expt, sentence[0])
    history = [back]
    for i in range(1, len(sentence)) :
        (prob, back) = viterbi_extend(expt, prob, sentence[i])
        history.append(back)
    return (prob, history)

#backtracking
def viterbi_decode(expt, prob, history, sentence) :
        best_previous = max(prob.keys(), 
                            key = lambda prevtag: \
                            prob[prevtag] * expt.cpd_tags[prevtag].prob('END'))
        p = prob[best_previous] * expt.cpd_tags[best_previous].prob('END')
        tags = [ 'END', best_previous ]
        history.reverse()
        current_best_tag = best_previous
        for bp in history:
            tags.append(bp[current_best_tag])
            current_best_tag = bp[current_best_tag]
        tags.reverse()
        return (p, zip(tags, ['START'] + sentence + ['END']))
    
def viterbi(expt, sentence) :
    prob, history = viterbi_run(expt, sentence)
    return viterbi_decode(expt, prob, history, sentence)


print viterbi(expt1, ['I', 'want', 'to', 'race'])