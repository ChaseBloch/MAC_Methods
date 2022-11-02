# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:40:45 2021

@author: chase
"""

import pandas as pd
import os
import glob
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk 
import random
import ast

random.seed(1234)

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\DownloadingArticles')





####################
###Merge Datasets###
####################

print('Merging Datasets')

path = r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\DownloadingArticles'  
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
        groups = list(zip(*[iter(sentences)]*5))
    for k in range(len(groups)):
        paragraphs.append(''.join([idx for tup in groups[k] for idx in tup]))
        article_number.append(i)

temp_df = pd.DataFrame(list(zip(paragraphs,article_number)), columns = ['paragraphs','article_number'])
df = temp_df.merge(df_full_nona, left_on='article_number', right_on='article_index')

df = df.drop_duplicates(subset = ['Source.Name','paragraphs'])
df = df.drop(columns = ['Document.Content','article_number'], axis = 1)


    
stemmer = nltk.stem.PorterStemmer()
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

vectorizer = StemmedTfidfVectorizer(stop_words='english', ngram_range = (1,2), max_df = .8, min_df = .001, max_features=75000)
#vectorizer = CountVectorizer(stop_words='english', ngram_range = (1,2), max_df = .8, min_df = .001, max_features=75000)

X = vectorizer.fit_transform(df['paragraphs'])
feature_names = vectorizer.get_feature_names_out()

dense = X.todense()
denselist = dense.tolist()
tdf = pd.DataFrame(denselist, columns=feature_names)


df_code = df.sample(1000)

df_code.to_csv(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods\DownloadingArticles\test_coding.csv', index = False)

