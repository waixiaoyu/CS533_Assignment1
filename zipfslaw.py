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

brown_word_fit.distribution_compare('power_law', 'lognormal_positive')

