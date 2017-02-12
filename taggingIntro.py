import nltk
from nltk.corpus import brown

text = nltk.word_tokenize("they permit")
nltk.pos_tag(text)

text = nltk.word_tokenize("the permit")
nltk.pos_tag(text)

# document for each tag
nltk.help.upenn_tagset('RB')

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(text)

brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
brown_news_tagged[:30]

tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.most_common()

word_tag_fd = nltk.FreqDist(brown_news_tagged)
def common_items(target_tag) :
    return [ word for ((word,tag), count) in
            word_tag_fd.most_common()[:1000]
            if tag == target_tag ]

print common_items('ADP')

def words_that_follow(token) :
    brown_news_tagged_bigrams = nltk.bigrams(brown_news_tagged)
    dist = nltk.FreqDist([two for ((one, _), (two, _)) in brown_news_tagged_bigrams 
                          if one == token ])
    return dist.most_common(20)
    
words_that_follow('other')
def pos_that_follow(token):
    brown_news_tagged_bigrams = nltk.bigrams(brown_news_tagged)
    dist = nltk.FreqDist([tag 
                          for ((one, _), (_, tag)) in brown_news_tagged_bigrams 
                          if one == token ])
    return dist

pos_that_follow('they').tabulate()




#generator function to get sentence : V T0 V
def brown_pattern(testp):
    for tagged_sentence in brown.tagged_sents() :
        for trigram in nltk.trigrams(tagged_sentence) :
            if testp(trigram) :
                yield trigram

def v_to_v_pattern( ((w1, t1), (w2, t2), (w3, t3)) ) :
    return t1.startswith('V') and t2 == 'TO' and t3.startswith('V')

matches = brown_pattern(v_to_v_pattern)
[ next(matches) for _ in range(10)]

#highly ambiguous
dictionary = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_news_tagged)
def very_ambiguous():
    for item in sorted(dictionary.conditions()) :
        if len(dictionary[item]) >= 3:
            tags = [tag for (tag, _) in dictionary[item].most_common()]
            yield (item, tags)
matches = very_ambiguous()
for i in range(20):
    (word, tags) = next(matches)
    print(word, ' '.join(tags))