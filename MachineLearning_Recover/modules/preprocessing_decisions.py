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
from sklearn import metrics
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

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
            if(model == XGBClassifier):
                BestParams = gridsearch(X_train, y_train, X_test, y_test)
            else:
                BestParams = gridsearch(score, X_train, y_train)
            
            #Run cross validation using the best parameters.
            if(model == MultinomialNB):
               clf = model(**BestParams).fit(X_train, y_train)
            elif(model == XGBClassifier):
                clf = model(**BestParams).fit(X_train, y_train)
            else:
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
    temp_df = pd.DataFrame(preprocessing_scores).transpose().rename(
        columns = {
            0:'vec_names', 1:'cv_mean',2:'cv_std', 3:'out_score', 4:'score_name'
            })
    fig = []
    for score in scores: 
        for_plot = temp_df[temp_df['score_name'] == score]
        fig.append(plt.figure())
        # Specify mean and error bar range.
        x = list(range(1, len(for_plot)+1))
        y = for_plot['cv_mean']
        e = for_plot['cv_std']
        
        plt.errorbar(y, x, xerr = e, linestyle='None', marker='o')
        plt.plot(for_plot['out_score'],x , linestyle = 'None', marker = 'o')
        plt.gca().invert_yaxis()
        plt.yticks(ticks = x, labels = for_plot['vec_names'])
        plt.xticks(np.arange(.4, 1, .05))
        plt.title('Average ' + score + ' Score from Sensitivity Analysis')
        plt.show()
    return fig, temp_df




def confidence_measures(predicted_prob, X_test, y_test, pred1):
    
    counter = np.arange(0.5,0.96,0.001)
    lcount = len(counter)
    f1=[]
    prec=[]
    rec=[]
    obs=[]
    obs_perc=[]
    acc=[]

    for i in range(lcount):
    
        indexNames_1 = np.where(predicted_prob>counter[i])
        indexNames_col_1 = np.unique(indexNames_1[0])
        y_test_new = np.take(y_test,indexNames_col_1)
        y_pred_new = np.take(pred1,indexNames_col_1)
    
        obs.append(len(indexNames_col_1))
        obs_perc.append(len(indexNames_col_1)/len(X_test))
        acc.append(metrics.accuracy_score(y_test_new,y_pred_new))
        f1.append(metrics.f1_score(y_test_new,y_pred_new,average='macro'))
        prec.append(metrics.precision_score(y_test_new,y_pred_new,average='macro'))
        rec.append(metrics.recall_score(y_test_new,y_pred_new,average='macro'))
        
    metr_df = pd.DataFrame(list(zip(counter, obs, obs_perc, acc, f1, prec, rec)), columns = ('counter','obs', 'obs_perc', 'acc', 'f1', 'prec', 'rec'))
    return metr_df 

def extract_forhand(df, df_test, rf, threshold, vec_rf):
    features_test = vec_rf.transform(df_test.paragraphs).toarray()
    pred_test = rf.predict(features_test)
    predicted_prob_test = rf.predict_proba(features_test)
    df_test['code'] = pred_test
    df_test['pp_1'] = predicted_prob_test[:,1]
    coded_index = np.where(predicted_prob_test > threshold)[0]
    coded = pd.concat([df_test.iloc[coded_index], df])
    for_hand = df_test.iloc[~df_test.index.isin(coded_index)]
    return for_hand, coded
