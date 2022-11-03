# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:40:45 2021

@author: chase
"""

import pandas as pd
import os
import glob
from bs4 import BeautifulSoup
import nltk 
import random

random.seed(1234)

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\DownloadingArticles')





####################
###Merge Datasets###
####################

print('Merging Datasets')

path = r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\DownloadingArticles\Downloaded'  
all_files = glob.glob(os.path.join(path , "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

#ResultId, Link, Source.Name, Document.Content
df_full = df[['ResultId','Title','Date','Document.Content','Source.Name']]
df_full_nona = df_full[df_full['Document.Content'].notna()]

article_index = list(range(len(df_full_nona)))
df_full_nona['article_index'] = article_index 


################
###Clean Text###
################
print('Datasets Merged, Begin Cleaning Text')

#Clean Text with Beautiful Soup
content_list = df_full_nona['Document.Content'].tolist()

paragraphs = []
article_number = []
for i in range(len(content_list)):
    soup = BeautifulSoup(content_list[i], 'html.parser').find_all("p")
    sentences = []
    for j in range(len(soup)):
        sentences.append(soup[j].get_text())
        groups = list(zip(*[iter(sentences)]*3))
    for k in range(len(groups)):
        paragraphs.append(''.join([idx for tup in groups[k] for idx in tup]))
        article_number.append(i)

temp_df = pd.DataFrame(list(zip(paragraphs,article_number)), columns = ['paragraphs','article_number'])
df = temp_df.merge(df_full_nona, left_on='article_number', right_on='article_index')


df = df.drop_duplicates(subset = ['Source.Name','paragraphs'])
df = df.drop(columns = ['Document.Content','article_number'], axis = 1)
df['par_number'] = df.index


df_code = df.sample(1000)

df.to_csv(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\OS_Test.csv', index = False)

df_code.to_csv(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\Downloading&Coding\test_coding3.csv', index = False)

