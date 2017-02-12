import nltk
from nltk.corpus import brown

text = nltk.word_tokenize("they permit")
print nltk.pos_tag(text)

text = nltk.word_tokenize("the permit")
print nltk.pos_tag(text)

# document for each tag
nltk.help.upenn_tagset('RB')

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
print nltk.pos_tag(text)

print
brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
print brown_news_tagged[:30]

tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
print tag_fd.most_common()