# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:08:52 2023

@author: chase
"""
import re
import os
import pandas as pd

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\MachineLearning\Downloading&Coding\Exported')

outpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation/MACoding/'
           r'MAC_Methods/MachineLearning/Downloading&Coding/Exported/')
inpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation\MACoding/'
          r'MAC_Methods/MachineLearning/Downloading&Coding/Downloaded/')

res = [f for f in os.listdir(outpath) if re.search(r'Full_MAC_.*.csv', f)]
li = []
for filename in res:
    df = pd.read_csv(outpath + filename, index_col=None, header=0)
    li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)

train1 = pd.read_csv('df_train.csv')


