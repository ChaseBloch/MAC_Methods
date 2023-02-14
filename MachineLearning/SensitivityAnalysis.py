# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:18:53 2022

@author: chase
"""
import os

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods'
         r'\MachineLearning')

import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from modules.nltk_stemmer import StemmedCountVectorizer
from sklearn.model_selection import train_test_split


from modules.NLTK_Stemmer import ProperNounExtractor
from modules.preprocessing_decisions import sensitivity_analysis, preprocess_plots, confidence_measures, extract_forhand
from modules.gridsearches import svc_gridsearch_sens, rf_gridsearch_sens, nb_gridsearch_sens, lr_gridsearch_sens, xgb_gridsearch, svc_gridsearch, rf_gridsearch

df = pd.read_csv(r'Downloading&Coding/Exported/df_train.csv')
df_test = pd.read_csv(r'Downloading&Coding\Exported/df_test.csv')

# Run sensitivity analysis

scores = ['f1']

# Run SVC sensitivity analysis
svc_preprocessing_scores = sensitivity_analysis(df, scores, SVC, svc_gridsearch_sens)
svc_plots, df_svc_pre = preprocess_plots(svc_preprocessing_scores, scores)

df_svc_pre.to_csv(r'Downloading&Coding/Exported/df_svc_pre.csv', index = False)

svc_plots[0].savefig('Plots/SV_f1_sensitivity.png', bbox_inches = "tight")

# Run Random Forest sensitivity analysis
rf_preprocessing_scores = sensitivity_analysis(df, scores, RandomForestClassifier, rf_gridsearch_sens)
rf_plots, df_rf_pre = preprocess_plots(rf_preprocessing_scores, scores)

df_rf_pre.to_csv(r'Downloading&Coding/Exported/df_rf_pre.csv', index = False)

rf_plots[0].savefig('Plots/RF_f1_sensitivity.png', bbox_inches = "tight")

# Run Multinomial Naive Bayes sensitivity analysis
nb_preprocessing_scores = sensitivity_analysis(df, scores, MultinomialNB, nb_gridsearch_sens)
nb_plots, df_nb_pre = preprocess_plots(nb_preprocessing_scores, scores)

df_nb_pre.to_csv(r'Downloading&Coding/Exported/df_nb_pre.csv', index = False)

nb_plots[0].savefig('Plots/NB_f1_sensitivity.png', bbox_inches = "tight")

# Run Logistic Regression sensitivity analysis
lr_preprocessing_scores = sensitivity_analysis(df, scores, LogisticRegression, lr_gridsearch_sens)
lr_plots, df_lr_pre = preprocess_plots(lr_preprocessing_scores, scores)

df_lr_pre.to_csv(r'Downloading&Coding/Exported/df_lr_pre.csv', index = False)

lr_plots[0].savefig('Plots/LR_f1_sensitivity.png', bbox_inches = "tight")

# Run XGBoost tuning
xgb_preprocessing_scores = sensitivity_analysis(df, scores, XGBClassifier, xgb_gridsearch)
xgb_plots, df_xgb_pre = preprocess_plots(xgb_preprocessing_scores, scores)

df_xgb_pre.to_csv(r'Downloading&Coding/Exported/df_xgb_pre.csv', index = False)

xgb_plots[0].savefig('Plots/XGB_f1_sensitivity.png', bbox_inches = "tight")


# Creating table of best performing models
df_svc_pre = pd.read_csv(r'Plots&Tables/df_svc_pre.csv')
df_rf_pre = pd.read_csv(r'Plots&Tables/df_rf_pre.csv')
df_nb_pre = pd.read_csv(r'Plots&Tables/df_nb_pre.csv')
df_lr_pre = pd.read_csv(r'Plots&Tables/df_lr_pre.csv')
df_xgb_pre = pd.read_csv(r'Plots&Tables/df_xgb_pre.csv')

svc_max = df_svc_pre.loc[df_svc_pre['out_score'].idxmax()]
rf_max = df_rf_pre.loc[df_rf_pre['out_score'].idxmax()]
nb_max = df_nb_pre.loc[df_nb_pre['out_score'].idxmax()]
lr_max = df_lr_pre.loc[df_lr_pre['out_score'].idxmax()]
xgb_max = df_xgb_pre.loc[df_xgb_pre['out_score'].idxmax()]

df_models = pd.concat([svc_max, rf_max, nb_max, lr_max, xgb_max],axis = 1).transpose()
df_models['names'] = ['SVC','Random Forest','Naive Bayes','Logistic Regression','XGBoost']
df_models['labels'] = df_models[['names', 'vec_names']].apply(lambda row: ':\n'.join(row.values.astype(str)), axis=1)

x = list(range(1, len(df_models)+1))
y = df_models['cv_mean']
e = df_models['cv_std']

plt.errorbar(y, x, xerr = e, linestyle='None', marker='o', label  = 'Cross-Validated')
plt.plot(df_models['out_score'],x , linestyle = 'None', marker = 'o', label = 'Out-Of-Sample')
plt.gca().invert_yaxis()
plt.yticks(ticks = x, labels = df_models['labels'])
plt.xticks(np.arange(.5, 1, .05))
plt.legend(loc = 2)
plt.savefig('Plots&Tables/AllModels_f1_sensitivity.png', bbox_inches = "tight", dpi = 600)
plt.show()

# SVC pre-analysis plot
x = list(range(1, len(df_svc_pre)+1))
y = df_svc_pre['cv_mean']
e = df_svc_pre['cv_std']

plt.errorbar(y, x, xerr = e, linestyle='None', marker='o', label = 'Cross-Validated')
plt.plot(df_svc_pre['out_score'],x , linestyle = 'None', marker = 'o', label = 'Out-Of-Sample')
plt.gca().invert_yaxis()
plt.yticks(ticks = x, labels = df_svc_pre['vec_names'])
plt.xticks(np.arange(.50, 1, .05))
plt.legend(loc = 2)
plt.savefig('Plots&Tables/SV_f1_sensitivity.png', bbox_inches = "tight", dpi = 600)
plt.show()

# RF pre-analysis plot
x = list(range(1, len(df_rf_pre)+1))
y = df_rf_pre['cv_mean']
e = df_rf_pre['cv_std']

plt.errorbar(y, x, xerr = e, linestyle='None', marker='o', label = 'Cross-Validated')
plt.plot(df_rf_pre['out_score'],x , linestyle = 'None', marker = 'o', label = 'Out-Of-Sample')
plt.gca().invert_yaxis()
plt.yticks(ticks = x, labels = df_rf_pre['vec_names'])
plt.xticks(np.arange(.50, 1, .05))
plt.legend(loc = 2)
plt.savefig('Plots&Tables/RF_f1_sensitivity.png', bbox_inches = "tight", dpi = 600)
plt.show()

# XGB pre-analysis plot
x = list(range(1, len(df_xgb_pre)+1))
y = df_xgb_pre['cv_mean']
e = df_xgb_pre['cv_std']

plt.errorbar(y, x, xerr = e, linestyle='None', marker='o', label = 'Cross-Validated')
plt.plot(df_xgb_pre['out_score'],x , linestyle = 'None', marker = 'o', label = 'Out-Of-Sample')
plt.gca().invert_yaxis()
plt.yticks(ticks = x, labels = df_xgb_pre['vec_names'])
plt.xticks(np.arange(.50, 1, .05))
plt.legend(loc = 2)
plt.savefig('Plots&Tables/XGB_f1_sensitivity.png', bbox_inches = "tight", dpi = 600)
plt.show()



