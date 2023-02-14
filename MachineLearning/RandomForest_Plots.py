# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:11:57 2023

@author: chase
"""
import os
import pandas as pd

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation'
         r'\MACoding\MAC_Methods\MachineLearning')


df_1 = pd.read_csv(r'Downloading&Coding/Exported/df_train_1.csv')
df_test_1 = pd.read_csv(r'Downloading&Coding/Exported/df_test_1.csv')
