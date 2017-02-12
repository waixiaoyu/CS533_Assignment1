import math
import nltk
import powerlaw
import collections
import numpy
import matplotlib.pyplot as plt
import scipy.stats as ss
import csv
_  = numpy.seterr(divide='ignore', invalid='ignore')

class Zipf(object):
    """
    Zipf's law data structure 
    Visualizes the rank-frequency distribution of discrete data
    """
    def __init__(self, name, rates):
        self.name = name
        self.rates = rates
        lowest_to_highest = ss.rankdata([c for (i,c) in rates.iteritems()])
        self.size = len(lowest_to_highest)
        self.ranks = [self.size - r for r in lowest_to_highest]
        self.rate_data = [c for (i,c) in rates.iteritems()]
        self.log_data = [math.log(c) for c in self.rate_data]
        self.corr = numpy.corrcoef(self.ranks, self.log_data)
        
    def graph(self):
        plt.figure()
        plt.plot(self.ranks, self.rate_data, 'ro')
        plt.xscale('log')
        plt.xlabel('Frequency rank of item')
        plt.yscale('log')
        plt.ylabel('Number of tokens of item')
        plt.title('Rank-frequency distribution for {}'.format(self.name))
    
    def singletons(self):
        for (i, c) in self.rates.iteritems():
            if c == 1 :
                yield i
    
    def n_items_with_rate(self, rate):
        return sum(1 for (i, c) in self.rates.iteritems() if c == rate)
    
    def describe_singletons(self):
        print 'In {}, {} of {} tokens are singletons.'.format(self.name,self.n_items_with_rate(1),self.size)

def power_law_graph(plf):
    f = plf.plot_pdf(color='b', linewidth=2)
    plf.power_law.plot_pdf(color='b', linestyle='--', ax=f)
    f.axes.set_xlabel('Magnitude of item')
    f.axes.set_ylabel('Probability of item')
    

brown_word_counts = collections.Counter(w.lower() for w in nltk.corpus.brown.words())
brown_word_zipf = Zipf("Brown corpus word frequencies", brown_word_counts)
brown_word_data = numpy.array(brown_word_counts.values())
brown_word_fit = powerlaw.Fit(brown_word_data, discrete=True)

brown_word_zipf.graph()

power_law_graph(brown_word_fit)

print brown_word_fit.power_law.alpha

print brown_word_fit.lognormal_positive.mu
print brown_word_fit.lognormal_positive.sigma

print "Mean: ", math.exp(brown_word_fit.lognormal_positive.mu + 0.5*brown_word_fit.lognormal_positive.sigma*brown_word_fit.lognormal_positive.sigma)
print "Median: ", math.exp(brown_word_fit.lognormal_positive.mu)
print "Mode: ", math.exp(brown_word_fit.lognormal_positive.mu - brown_word_fit.lognormal_positive.sigma*brown_word_fit.lognormal_positive.sigma)

print brown_word_fit.distribution_compare('power_law', 'lognormal_positive')

###example 2

#tagged_corpus = nltk.corpus.brown.tagged_words(tagset='universal')
#brown_wt_counts = collections.Counter((w.lower(), t) for (w, t) in tagged_corpus)
#brown_wt_zipf = Zipf("Brown corpus word-tag frequencies", brown_wt_counts)
#brown_wt_data = numpy.array(brown_wt_counts.values())
#brown_wt_fit = powerlaw.Fit(brown_wt_data, discrete=True)

###example 3

populations = dict()
with open('./Top5000Population.csv', 'r') as csvfile :
    cityreader = csv.reader(csvfile)
    for row in cityreader:
        populations[row[0]+", "+row[1]] = int(row[2].replace(",", ""))

cities_zipf = Zipf("Populations of US cities", populations)
cities_data = numpy.array(populations.values())
cities_fit = powerlaw.Fit(cities_data, discrete=True)

cities_zipf.graph()
power_law_graph(cities_fit)

print cities_fit.power_law.alpha
print cities_fit.distribution_compare('power_law', 'lognormal_positive')

###exercise
words = dict()
file = open("./words.txt")
index = 0
while 1:
    line = file.readline()
    if not line:
        break
    words[index]=int(line.replace("\n", ""))
    index=index+1

words_zipf = Zipf("The frequency of occurrence of unique words", words)
words_data = numpy.array(words.values())
words_fit = powerlaw.Fit(words_data, discrete=True)

words_zipf.graph()
power_law_graph(words_fit)

print words_fit.power_law.alpha
print words_fit.distribution_compare('power_law', 'lognormal_positive')
