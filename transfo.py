#Tous les packages

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer

import nltk
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
#from sklearn.pipeline import FeatureUnionclass

class Extracteur_Mots(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.regexp = RegexpTokenizer("[a-z][a-z']{2,}")

    def fit(self, comments, y = None):
        return self

    def transform(self, comments, y = None):
        mots = []
        for c in comments:
            mots.append(self.regexp.tokenize(c.lower()))
        return mots



def stop_words_filtering(words):
    res = []
    sw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
      "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
      'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
      "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
      'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
      'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
      'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
      'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
      'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
      'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
      'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
      'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
      'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
      "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
      "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
      'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
      "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
      "wouldn't", "an'", "at'", "dn'", "en'", "he'", "it'", "ld'", "on'", "ou'", "sn'", "tn'", "i'd"]
    sw = set(sw)

    for w in words:
        if w not in sw:
            res.append(w)
    return res


class Filtre_Mots(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, mots, y = None):
        return self

    def transform(self, mots, y = None):
        for i in range(len(mots)):
            mots[i] = stop_words_filtering(mots[i])
        return mots


class Stemmer_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        return

    def fit(self, mots, y = None):
        return self

    def transform(self, mots, y = None):
        for i in range(len(mots)):
            for mot in mots[i]:
                self.stemmer.stem(mot)
        return mots

class Transformer_join(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, mots, y = None):
        return self

    def transform(self, mots, y = None):
        comments2 = []
        for m in mots:
            comments2.append(" ".join(m))
        return comments2
