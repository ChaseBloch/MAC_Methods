# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:37:34 2022

@author: chase
"""
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = nltk.stem.PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])