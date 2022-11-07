# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import os

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\MachineLearning')

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#User modules
from Modules.PreProcessingDecisions import SVC_sensitivity, pre_process_plots

#Importing and preparing datasets

df = pd.read_csv(r"C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\test_coding3.csv")
df_OS_test = pd.read_csv(r"C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\OS_test.csv")

df = df[df['code'].notna()]
df_OS_test = df_OS_test[~df_OS_test.par_number.isin(df.par_number)]

#Running sensitivity analysis

scores = ['f1', 'precision']


preprocessing_scores = SVC_sensitivity(df, scores)


pre_process_plots(preprocessing_scores, scores)

