# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:08:52 2023

@author: chase
"""
import re
import os
import pandas as pd

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\MachineLearning')

from modules.Cleaning import nodup_sample

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\MachineLearning\Downloading&Coding\Exported')

outpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation/MACoding/'
           r'MAC_Methods/MachineLearning/Downloading&Coding/Exported/')
inpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation\MACoding/'
          r'MAC_Methods/MachineLearning/Downloading&Coding/Downloaded/')

res = [f for f in os.listdir(outpath) if re.search(r'Full_.*.csv', f)]
li = []
for filename in res:
    df_test = pd.read_csv(outpath + filename, index_col=None, header=0)
    li.append(df_test)
    df_test = pd.concat(li, axis=0, ignore_index=True)
    
df_test = df_test.drop_duplicates(subset = ['paragraphs'])

train1 = pd.read_csv('df_train.csv')[['paragraphs','code']]
train1_final = train1.merge(df_test, how = 'inner', left_on=('paragraphs'), right_on=('paragraphs'))
train1_nodups = train1_final.drop_duplicates(subset = ['paragraphs'])
train1_nodups.drop('code_y', inplace = True, axis = 1)
train1_nodups.rename(columns = {'code_x':'code'}, inplace = True)

nodup_sample(inpath, outpath, 2)

res = [f for f in os.listdir(outpath) if re.search(r'ForCode_\d_.*.csv', f)]
li = []
for filename in res:
    df = pd.read_csv(outpath + filename, index_col=None, header=0)
    li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    
df = pd.concat([train1_nodups, df], ignore_index = True)

# Drop duplicates and remove segments in training set from test set.
df = df.drop_duplicates(subset = ['paragraphs']) 
df_test = df_test[~df_test.par_number.isin(df.par_number)].reset_index()

df.to_csv('df_train_1.csv')
df_test.to_csv('df_test_1.csv')




