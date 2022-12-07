# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, precision_score, 
                             recall_score, confusion_matrix, accuracy_score, 
                             ConfusionMatrixDisplay)
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation'
         r'\MACoding\MAC_Methods\MachineLearning')

# Local modules
from modules.gridsearches import svc_gridsearch, rf_gridsearch_sens, nb_gridsearch_sens, lr_gridsearch_sens, xgb_gridsearch, svc_gridsearch, rf_gridsearch
from modules.nltk_stemmer import StemmedCountVectorizer, ProperNounExtractor
from modules.preprocessing_decisions import sensitivity_analysis, preprocess_plots, confidence_measures, extract_forhand

df = pd.read_csv(r'Downloading&Coding/Exported/df_train_2.csv')
df_test = pd.read_csv(r'Downloading&Coding/Exported/df_test_2.csv')

scores = ['f1_macro']
labels = df.code

# Run Full SVC Model
vec_svc = StemmedCountVectorizer(
    #norm='l2',
    encoding='utf-8', 
    stop_words='english',
    ngram_range = (1,1),
    max_df = .8, min_df = 3, max_features=60000,
    strip_accents = 'ascii', lowercase=True
    )

features_svc = vec_svc.fit_transform(df.paragraphs).toarray()

X_train_svc, X_test_svc, y_train_svc, y_test_svc = train_test_split(
    features_svc, labels, random_state = 1234,test_size=0.3
    )

SVC_BestParams = svc_gridsearch(scores, X_train_svc, y_train_svc)
svc = SVC(**SVC_BestParams, class_weight = {0:.1, 1:.9}, probability = True).fit(X_train_svc, y_train_svc)
#pickle.dump(svc, open('Saves/svc.pkl', 'wb'))

# Run full Random Forest model
vec_rf = TfidfVectorizer(
    norm='l2', encoding='utf-8', 
    ngram_range = (1,2), stop_words = 'english', 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    )

features_rf = vec_rf.fit_transform(df.paragraphs).toarray()
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    features_rf, labels, random_state = 1234,test_size=0.3
    )

RF_BestParams = rf_gridsearch(scores, X_train_rf, y_train_rf)
rf = RandomForestClassifier(**RF_BestParams, class_weight = {0:.24, 1:.76}, n_jobs = -1).fit(X_train_rf, y_train_rf)
#pickle.dump(rf, open('Saves/rf_2.pkl', 'wb'))

# Run full XGBoost model
vec_xgb = StemmedCountVectorizer(
    encoding='utf-8', ngram_range = (1,1), 
    stop_words = 'english', 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    )

features_xgb = vec_xgb.fit_transform(df.paragraphs).toarray()
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    features_xgb, labels, random_state = 1234,test_size=0.3
    )

XGB_BestParams = xgb_gridsearch(X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb)
xgb = xgb.XGBClassifier(**XGB_BestParams).fit(X_train_xgb, y_train_xgb)
#pickle.dump(xgb, open('Saves/xgb.pkl', 'wb'))

# Reload saved models
svc = pickle.load(open('Saves/svc.pkl', 'rb'))
rf = pickle.load(open('Saves/rf_2.pkl', 'rb'))
xgb = pickle.load(open('Saves/xgb.pkl', 'rb'))

svc_pred = svc.predict(X_test_svc)
svc_predicted_prob = svc.predict_proba(X_test_svc)
svc_confidence = confidence_measures(svc_predicted_prob, X_test_svc, y_test_svc, svc_pred)

rf_pred = rf.predict(X_test_rf)
rf_predicted_prob = rf.predict_proba(X_test_rf)
rf_confidence = confidence_measures(rf_predicted_prob, X_test_rf, y_test_rf, rf_pred)

rf_cm = confusion_matrix(y_test_rf, rf_pred)
rf_display = ConfusionMatrixDisplay(confusion_matrix = rf_cm, display_labels = [False, True])
rf_display.plot()
plt.show()


xgb_pred = xgb.predict(X_test_xgb)
xgb_predicted_prob = xgb.predict_proba(X_test_xgb)
xgb_confidence = confidence_measures(xgb_predicted_prob, X_test_xgb, y_test_xgb, xgb_pred)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Percent of Observations Coded')
ax1.invert_xaxis()
plt.xticks(np.arange(0, 1.1, .1))
plt.yticks(np.arange(.9, 1.01, .01))
plt.axhline(y = 0.95, color = 'black', linestyle = 'dashed')
ax1.set_ylabel('Accuracy')
#line1, = ax1.plot(svc_confidence['obs_perc'], svc_confidence['acc'], color='tab:red', label = 'SVC')
line2, = ax1.plot(rf_confidence['obs_perc'], rf_confidence['acc'], color='tab:blue', label = 'Random Forest')
#line3, = ax1.plot(xgb_confidence['obs_perc'], xgb_confidence['acc'], color='tab:green', label ='XGBoost')
ax1.legend(handles=[line2], loc = 4)

#plt.savefig('MAC_Performance.pdf',bbox_inches='tight')
plt.show()

# Test on full set
for_hand_svc, coded_svc = extract_forhand(df, df_test, svc, .783 ,vec_svc)
for_hand_rf, coded_rf = extract_forhand(df, df_test, rf, .641, vec_rf)
for_hand_xgb, coded_xgb = extract_forhand(df, df_test, xgb, .564 ,vec_xgb)

# Draw another sample for training labelling
temp = []
window = 0
while len(temp) < 500:
    temp = list(np.where((for_hand_rf['pp_1'] > .5 - window) & (for_hand_rf['pp_1'] < .5 + window))[0])
    window = window + .000001

df_unconf = for_hand_rf.iloc[temp]
df_unconf.to_csv(r'Downloading&Coding/Exported/training_2.csv', index = False)

# Create file for final hand-coding
df_coded = coded_rf[coded_rf['code'] == 1]

temp = df_coded.groupby(['article_index', 'Title','Date','Source.Name'])['paragraphs'].agg('\n'.join).reset_index()
temp['year'] = [int(x[0:4]) for x in temp['Date']]
                
prop_nouns = [] 
for paragraph in temp['paragraphs']:
    prop_nouns.append(list(set(ProperNounExtractor(paragraph))))
    
temp['prop_nouns'] = prop_nouns
df_prop = temp.explode('prop_nouns').reset_index(drop=True)
df_prop['paragraphs'] = [x[0:32000] for x in df_prop['paragraphs']]

removals = df_prop['prop_nouns'].value_counts().reset_index()
#removals.to_csv('Downloading&Coding/Exported/removals.csv')

df_removals = pd.read_csv('Downloading&Coding/Exported/removals.csv')
df_removals = df_removals[df_removals['country'] == 1]

df_propmerge = df_removals.merge(df_prop, left_on = 'index', right_on = 'prop_nouns')
df_propmerge = df_propmerge.drop_duplicates(subset=['country_name','article_index'])
df_propmerge['paragraphs'] = df_propmerge[['article_index', 'paragraphs']].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1)

df_final = df_propmerge.groupby(['year','country_name'])['paragraphs'].agg('\n--------------------------------------------------\n'.join).reset_index()
df_final['output'] = df_final[['year', 'country_name']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df_final['output'] = df_final[['output','paragraphs']].apply(lambda row: ':\n'.join(row.values.astype(str)), axis=1)

df_final['output'].to_csv('Downloading&Coding/Exported/final_articles.txt', sep =' ', index = False)
df_final[['year','country_name']].to_csv('Downloading&Coding/Exported/final_articles.csv')