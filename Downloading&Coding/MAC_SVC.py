# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df = pd.read_csv(r"C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\test_coding.csv")
df_OS_test = pd.read_csv(r"C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\OS_test.csv")

stemmer = nltk.stem.PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

vectorizer = StemmedTfidfVectorizer(stop_words='english', ngram_range = (1,2), max_df = .8, min_df = 3, max_features=75000)
#vectorizer = CountVectorizer(stop_words='english', ngram_range = (1,2), max_df = .8, min_df = .001, max_features=75000)

X = vectorizer.fit_transform(df['paragraphs'])
feature_names = vectorizer.get_feature_names_out()

dense = X.todense()
denselist = dense.tolist()
tdf = pd.DataFrame(denselist, columns=feature_names)
