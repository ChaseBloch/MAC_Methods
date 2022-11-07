# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import os
import re

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\MachineLearning')

outpath = 'C:/Users/chase/GDrive/GD_Work/Dissertation/MACoding/MAC_Methods/MachineLearning/Downloading&Coding/Exported/'
inpath = 'C:/Users/chase/GDrive/GD_Work/Dissertation\MACoding/MAC_Methods/MachineLearning/Downloading&Coding/Downloaded/'


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#User modules
from Modules.PreProcessingDecisions import SVC_sensitivity, pre_process_plots
from Modules.Cleaning import clean_multi

#Importing and preparing datasets

clean_multi(inpath, outpath)

res = [f for f in os.listdir(outpath) if re.search(r'ForCode_\d_.*.csv', f)]
li = []
for filename in res:
    df = pd.read_csv(outpath + filename, index_col=None, header=0)
    li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    
df_OS_test = pd.read_csv(r"C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\OS_test.csv")

df = df[df['code'].notna()]
df_OS_test = df_OS_test[~df_OS_test.par_number.isin(df.par_number)]

#Running sensitivity analysis

scores = ['f1', 'precision']


preprocessing_scores = SVC_sensitivity(df, scores)


pre_process_plots(preprocessing_scores, scores)

