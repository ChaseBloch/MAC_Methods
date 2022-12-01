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
from modules.gridsearches import svc_gridsearch_sens, rf_gridsearch_sens


os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods'
         r'\MachineLearning\Downloading&Coding\Exported')

df = pd.read_csv(r'Downloading&Coding/Exported/df_train.csv')
df_test = pd.read_csv(r'Downloading&Coding/Exported/df_test.csv')

#Run sensitivity analysis
scores = ['f1']
svc_preprocessing_scores = sensitivity_analysis(df, scores, SVC, svc_gridsearch_sens)
rf_preprocessing_scores = sensitivity_analysis(df, scores, RandomForestClassifier, rf_gridsearch_sens)
rf_plots = preprocess_plots(rf_preprocessing_scores, scores)

rf_plots[0].savefig('Plots/f1_sensitivity.png', bbox_inches = "tight")
rf_plots[1].savefig('Plots/precision_sensitivity.png', bbox_inches = "tight")
