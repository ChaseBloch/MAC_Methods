# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:40:45 2021

@author: chase
"""

import pandas as pd
import os
import regex as re
import glob
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

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


################
###Clean Text###
################
print('Datasets Merged, Begin Cleaning Text')

"""
def clean_text(text):
    xmltext = re.sub(u"[^\x20-\x7f]+",u" ",text) #Gets rid of special characters
    xml_temp2 = re.sub(r'^.*?<nitf:body.content><bodyText>','', xmltext) #Gets rid of everything before <nitf:body.content><bodyText>
    stripped = xml_temp2.split('</p></bodyText>', 1)[0] #Removes everything after </p></bodyText> 
    xml_final = re.sub('<[^>]+>', ' ', stripped) #Removes everything within <>
    return(xml_final)

content_list = df_full_nona['Document.Content'].tolist()
df_cleaned = [clean_text(row) for row in content_list]

df_full_nona['clean_content'] = df_cleaned

df_full_nodups = df_full_nona.drop_duplicates(subset = ['Source.Name','clean_content'])
"""


#Clean Text with Beautiful Soup
content_list = df_full_nona['Document.Content'].tolist()

bs_content = []
for s in content_list:
    body = []
    soup = BeautifulSoup(s, 'html.parser').find_all("p")
    for p in soup:
        body.append(p.get_text())
    bs_content.append("\n".join(body))
    
df_full_nona['clean_content'] = bs_content

df_full_nodups = df_full_nona.drop_duplicates(subset = ['Source.Name','clean_content'])
df = df_full_nodups.drop(columns = 'Document.Content', axis = 1)


    

    
    
vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,1), max_df = .9, min_df = .01)

X = vectorizer.fit_transform(bs_content)
feature_names = vectorizer.get_feature_names_out()

dense = X.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)


