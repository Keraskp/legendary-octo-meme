from numpy import float32, zeros
from nltk import download, word_tokenize
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
download('punkt')

def tokenize(sentence):
    return word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = zeros(len(words), dtype=float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag