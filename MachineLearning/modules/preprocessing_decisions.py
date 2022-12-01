# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:04:03 2022

@author: chase
"""
import os
outpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation/MACoding/'
           r'MAC_Methods/MachineLearning/Downloading&Coding/Exported/')
inpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation\MACoding/'
          r'MAC_Methods/MachineLearning/Downloading&Coding/Downloaded/')

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation'
         r'\MACoding\MAC_Methods\MachineLearning')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (f1_score, precision_score)
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from modules.nltk_stemmer import StemmedTfidfVectorizer, StemmedCountVectorizer
    
vectorizer = [
# Baseline
StemmedTfidfVectorizer(
    norm='l2', encoding='utf-8', 
    stop_words='english', ngram_range = (1,2),
    max_df = .8, min_df = 3, max_features=60000,
    strip_accents = 'ascii', lowercase=True
    ),

StemmedCountVectorizer(
    encoding='utf-8', stop_words='english', 
    ngram_range = (1,2), max_df = .8, 
    min_df = 3, max_features=60000,
    strip_accents = 'ascii', lowercase=True
    ),

# Keep stop words
StemmedTfidfVectorizer(
    norm='l2', encoding='utf-8', 
    ngram_range = (1,2), max_df = .8, 
    min_df = 3, max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    ),

StemmedCountVectorizer(
    encoding='utf-8', ngram_range = (1,2), 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    ),

# Unigrams Only
StemmedTfidfVectorizer(
    norm='l2', encoding='utf-8', 
    ngram_range = (1,1), stop_words = 'english', 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    ),

StemmedCountVectorizer(
    encoding='utf-8', ngram_range = (1,1), 
    stop_words = 'english', 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    ),
# No Stemming
TfidfVectorizer(
    norm='l2', encoding='utf-8', 
    ngram_range = (1,2), stop_words = 'english', 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    ),

CountVectorizer(
    encoding='utf-8', ngram_range = (1,2), 
    stop_words = 'english', 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    )
]

# Specify the list of pre-processing decisions for output.
vec_name = [
'TF-IDF Baseline', 'BoW Baseline',
'TF-IDF Keep Stop Words', 'BoW Keep Stop Words',
"TF-IDF Unigrams Only", "BoW Unigrams Only",
"TF-IDF No Stemming", "BoW No Stemming"
]

def sensitivity_analysis(df, scores, model, gridsearch):    
    cv_mean = []
    cv_std = []
    Out_score = []
    score_name = []
    vec_names = []
    i = 0
    for score in scores:
        for vec in vectorizer:
            print(vec_name[i])
            
            # Extract features and labels, and split data for training.
            features = vec.fit_transform(df.paragraphs).toarray()
            labels = df.code
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, random_state = 1234,test_size=0.3
                )
           
            # Run the sensitivity analysis and store the best parameters.
            BestParams = gridsearch(score, X_train, y_train)
            
            #Run cross validation using the best parameters.
            clf = model(**BestParams, class_weight = {0:.1, 1:.9}).fit(X_train, y_train)
            cv_scores = cross_val_score(
                clf, X_train, y_train, cv = 5, scoring = score
                )
            cv_mean.append(cv_scores.mean())
            cv_std.append(cv_scores.std())
            score_name.append(score)
            pred1 = clf.predict(X_test)
            
            # Specify the output for different scores.
            if score == 'precision':
                Out_score.append(
                    precision_score(y_test, pred1, average="macro")
                    )
                print('Calculating Out of Sample Precision')
                print()
            elif score == 'f1':
                Out_score.append(f1_score(y_test, pred1, average="macro"))
                print('Calculating Out of Sample F1 Score')
                print()
            else:
                sys.exit(score + ' not yet added to pre-process function')    
            i = i+1
        vec_names = vec_names + vec_name
    return([vec_names, cv_mean,cv_std, Out_score, score_name])


def preprocess_plots(preprocessing_scores, scores):
    temp_df = pd.DataFrame(preprocessing_scores).transpose()
    fig = []
    for score in scores: 
        for_plot = temp_df[temp_df[4] == score]
        fig.append(plt.figure())
        # Specify mean and error bar range.
        x = list(range(1, len(for_plot)+1))
        y = for_plot[1]
        e = for_plot[2]
        
        plt.errorbar(y, x, xerr = e, linestyle='None', marker='o')
        plt.plot(for_plot[3],x , linestyle = 'None', marker = 'o')
        plt.gca().invert_yaxis()
        plt.yticks(ticks = x, labels = for_plot[0])
        plt.xticks(np.arange(.4, 1, .05))
        plt.title('Average ' + score + ' Score from Sensitivity Analysis')
        plt.show()
    return fig
