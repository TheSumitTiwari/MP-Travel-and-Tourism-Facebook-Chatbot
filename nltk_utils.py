from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
# nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentance):
    """
    Tokenize text using NLTK
    """
    return nltk.word_tokenize(sentance)


def stem(word):
    """
    Stem word using PorterStemmer
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentance, all_words):
    """
    Create bag of words
    """
    tokenized_sentance = [stem(w) for w in tokenized_sentance]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for i, w in enumerate(all_words):
        if w in tokenized_sentance:
            bag[i] = 1.0
    return bag

