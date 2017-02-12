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

print common_items('ADJ')