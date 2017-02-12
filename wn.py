import nltk
from nltk.corpus import wordnet as wn

def describe(w, **kwargs) :
    for n in wn.synsets(w, **kwargs) :
        print n.name(), n.definition() 
        if len(n.examples()) > 0 :
            print "     (", n.examples()[0], ")"
            
describe('newspaper')

describe('rise', pos=wn.VERB)

def synonyms(s, **kwargs) :
    ws = wn.synset(s, **kwargs)
    print "Synonyms in", ws.name()
    for l in ws.lemmas() :
        print "    ", str(l.name())
        
synonyms('bag.n.03')
synonyms('survive.v.01')




dog = wn.synset('dog.n.01')
hyper = lambda s: s.hypernyms()
list(dog.closure(hyper, depth=1)) == dog.hypernyms()
list(dog.closure(hyper))

hypon= lambda s: s.hyponyms()
list(dog.closure(hypon))
