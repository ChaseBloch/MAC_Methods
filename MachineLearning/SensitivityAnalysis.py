# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:18:53 2022

@author: chase
"""
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from modules.preprocessing_decisions import sensitivity_analysis, preprocess_plots
from modules.gridsearches import svc_gridsearch_sens, rf_gridsearch_sens, nb_gridsearch_sens
from sklearn.naive_bayes import MultinomialNB


os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods'
         r'\MachineLearning\Downloading&Coding\Exported')

df = pd.read_csv(r'df_train.csv')
df_test = pd.read_csv(r'df_test.csv')

# Run sensitivity analysis
os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods'
         r'\MachineLearning')

scores = ['f1']

# Run SVC sensitivity analysis
svc_preprocessing_scores = sensitivity_analysis(df, scores, SVC, svc_gridsearch_sens)
svc_plots = preprocess_plots(svc_preprocessing_scores, scores)

svc_plots[0].savefig('Plots/SV_f1_sensitivity.png', bbox_inches = "tight")

# Run Random Forest sensitivity analysis
rf_preprocessing_scores = sensitivity_analysis(df, scores, RandomForestClassifier, rf_gridsearch_sens)
rf_plots = preprocess_plots(rf_preprocessing_scores, scores)

rf_plots[0].savefig('Plots/RF_f1_sensitivity.png', bbox_inches = "tight")

# Run Multinomial Naive Bayes sensitivity analysis
nb_preprocessing_scores = sensitivity_analysis(df, scores, MultinomialNB, nb_gridsearch_sens)
nb_plots = preprocess_plots(nb_preprocessing_scores, scores)

nb_plots[0].savefig('Plots/NB_f1_sensitivity.png', bbox_inches = "tight")