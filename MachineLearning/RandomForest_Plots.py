# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:11:57 2023

@author: chase
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation'
         r'\MACoding\MAC_Methods\MachineLearning')

from modules.preprocessing_decisions import confidence_measures

df_1 = pd.read_csv(r'Downloading&Coding/Exported/df_train_1.csv')
df_test_1 = pd.read_csv(r'Downloading&Coding/Exported/df_test_1.csv')

df_2 = pd.read_csv(r'Downloading&Coding/Exported/df_train_2.csv')
df_test_2 = pd.read_csv(r'Downloading&Coding/Exported/df_test_2.csv')

df_3 = pd.read_csv(r'Downloading&Coding/Exported/df_train_3.csv')
df_test_3 = pd.read_csv(r'Downloading&Coding/Exported/df_test_3.csv')

labels_1 = df_1.code
labels_2 = df_2.code
labels_3 = df_3.code

vec_rf = TfidfVectorizer(
    norm='l2', encoding='utf-8', 
    ngram_range = (1,2), stop_words = 'english', 
    max_df = .8, min_df = 3, 
    max_features=60000, 
    strip_accents = 'ascii', lowercase=True
    )

features_rf_1_array = vec_rf.fit_transform(df_1.paragraphs).toarray()
features_rf_1 = pd.DataFrame(features_rf_1_array)
X_train_rf_1, X_test_rf_1, y_train_rf_1, y_test_rf_1 = train_test_split(
    features_rf_1, labels_1, random_state = 1234,test_size=0.3
    )

vec_rf.fit_transform(df_2.paragraphs)
features_rf_2_array = vec_rf.transform(df_1.paragraphs).toarray()
features_rf_2 = pd.DataFrame(features_rf_2_array)
X_train_rf_2, X_test_rf_2, y_train_rf_2, y_test_rf_2 = train_test_split(
    features_rf_2, labels_1, random_state = 1234,test_size=0.3
    )

features_temp_array = vec_rf.fit_transform(df_3.paragraphs).toarray()
features_temp = pd.DataFrame(features_temp_array)
X_train_rf_temp, X_test_rf_temp, y_train_rf_temp, y_test_rf_temp = train_test_split(
    features_temp, labels_3, random_state = 1234,test_size=0.3
    )

features_rf_3_array = vec_rf.transform(df_1.paragraphs).toarray()
features_rf_3 = pd.DataFrame(features_rf_3_array)
X_train_rf_3, X_test_rf_3, y_train_rf_3, y_test_rf_3 = train_test_split(
    features_rf_3, labels_1, random_state = 1234,test_size=0.3
    )

temp = (X_train_rf_temp==X_test_rf_3[:,None]).all(-1)
temp
temp_2 = (X_test_rf_temp==X_test_rf_3[:,None]).all(-1).sum()
temp_2
temp_3 = (X_train_rf_temp==X_test_rf_temp[:,None]).all(-1).sum()
temp_3
temp_4 = (X_train_rf_3==X_test_rf_3[:,None]).all(-1).sum()
temp_4

rf_1 = pickle.load(open('Saves/rf_1.pkl', 'rb'))
rf_2 = pickle.load(open('Saves/rf_2.pkl', 'rb'))
rf_3 = pickle.load(open('Saves/rf_3.pkl', 'rb'))

rf_pred_1 = rf_1.predict(X_test_rf_1)
rf_predicted_prob_1 = rf_1.predict_proba(X_test_rf_1)
rf_confidence_1 = confidence_measures(rf_predicted_prob_1, X_test_rf_1, y_test_rf_1, rf_pred_1)

rf_pred_2 = rf_2.predict(X_test_rf_2)
rf_predicted_prob_2 = rf_2.predict_proba(X_test_rf_2)
rf_confidence_2 = confidence_measures(rf_predicted_prob_2, X_test_rf_2, y_test_rf_2, rf_pred_2)

rf_pred_3 = rf_3.predict(X_test_rf_3)
rf_predicted_prob_3 = rf_3.predict_proba(X_test_rf_3)
rf_confidence_3 = confidence_measures(rf_predicted_prob_3, X_test_rf_3, y_test_rf_3, rf_pred_3)

# Plot of accuracy by percent coded
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Percent of Observations Coded')
ax1.invert_xaxis()
plt.xticks(np.arange(0, 1.1, .1))
plt.yticks(np.arange(.9, 1.01, .01))
plt.axhline(y = 0.95, color = 'black', linestyle = 'dashed')
ax1.set_ylabel('Accuracy')
line1, = ax1.plot(rf_confidence_1['obs_perc'], rf_confidence_1['acc'], color='tab:green', label = 'Original', ls = 'dashdot')
line2, = ax1.plot(rf_confidence_2['obs_perc'], rf_confidence_2['acc'], color='tab:blue', label = 'Second Round', ls = 'solid')
line3, = ax1.plot(rf_confidence_3['obs_perc'], rf_confidence_3['acc'], color='tab:red', label = 'Third Round', ls = 'dashed')
ax1.legend(handles=[line1, line2, line3], loc = 4)

#plt.savefig('Plots&Tables/MAC_Performance.png',bbox_inches='tight', dpi = 600)
plt.show()
