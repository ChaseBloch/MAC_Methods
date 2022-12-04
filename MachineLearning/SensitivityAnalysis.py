# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:18:53 2022

@author: chase
"""
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods'
         r'\MachineLearning/')

from modules.preprocessing_decisions import sensitivity_analysis, preprocess_plots
from modules.gridsearches import svc_gridsearch_sens, rf_gridsearch_sens, nb_gridsearch_sens, lr_gridsearch_sens, xgb_gridsearch

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
