# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import os

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\MachineLearning')

import pandas as pd
from sklearn.model_selection import train_test_split



df = pd.read_csv(r"C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\test_coding3.csv")
df_OS_test = pd.read_csv(r"C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\OS_test.csv")

df = df[df['code'].notna()]
df_OS_test = df_OS_test[~df_OS_test.par_number.isin(df.par_number)]

from NLTK_Stemmer import StemmedTfidfVectorizer
vectorizer = StemmedTfidfVectorizer(norm='l2', encoding='utf-8', 
                                    stop_words='english', ngram_range = (1,2),
                                    max_df = .8, min_df = 3, max_features=10000,
                                    strip_accents = 'ascii')

features = vectorizer.fit_transform(df.paragraphs).toarray()
labels = df.code

X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state = 0,test_size=0.3)

#######Tuning Hyper-Parameters for SVC
###https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py

scores = ['precision', 'recall']

from SVC_GridSearch import SVC_gridsearch
SVC_BestParams = SVC_gridsearch(scores, X_train, y_train, X_test, y_test)
