# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import os
import re
import pandas as pd
import warnings

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation'
         r'\MACoding\MAC_Methods\MachineLearning')

# Local modules
from modules.preprocessing_decisions import svc_sensitivity, preprocess_plots
from modules.cleaning import clean_multi, nodup_sample



outpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation/MACoding/'
           r'MAC_Methods/MachineLearning/Downloading&Coding/Exported/')
inpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation\MACoding/'
          r'MAC_Methods/MachineLearning/Downloading&Coding/Downloaded/')

# Silence warnings caused by models without good training data.
warnings.filterwarnings('ignore') 

# Importing and preparing datasets
clean_multi(inpath, outpath)
nodup_sample(inpath, outpath, 1500)

# Re-import files and merge them after hand-coding
res = [f for f in os.listdir(outpath) if re.search(r'ForCode_\d_.*.csv', f)]
li = []
for filename in res:
    df = pd.read_csv(outpath + filename, index_col=None, header=0)
    li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    
res = [f for f in os.listdir(outpath) if re.search(r'Full_.*.csv', f)]
li = []
for filename in res:
    df_test = pd.read_csv(outpath + filename, index_col=None, header=0)
    li.append(df_test)
    df_test = pd.concat(li, axis=0, ignore_index=True)

# Drop duplicates and remove segments in training set from test set.
df = df[df['code'].notna()]
df_test = df_test[~df_test.par_number.isin(df.par_number)]

#Run sensitivity analysis
scores = ['f1']
preprocessing_scores = svc_sensitivity(df, scores)
preprocess_plots(preprocessing_scores, scores)

