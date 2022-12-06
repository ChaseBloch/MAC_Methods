# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:11:49 2022

@author: chase
"""
import os
import re
import pandas as pd

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods'
         r'\MachineLearning\Downloading&Coding\Exported')

outpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation/MACoding/'
           r'MAC_Methods/MachineLearning/Downloading&Coding/Exported/')
inpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation\MACoding/'
          r'MAC_Methods/MachineLearning/Downloading&Coding/Downloaded/')

# Importing and preparing datasets
#clean_multi(inpath, outpath)
#nodup_sample(inpath, outpath, 1500)

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
df = df[df['code'].notna()].reset_index()
df_test = df_test[~df_test.par_number.isin(df.par_number)].reset_index()

df.to_csv('df_train_2.csv')
df_test.to_csv('df_test_2.csv')