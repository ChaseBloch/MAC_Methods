# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:40:45 2021

@author: chase
"""

import pandas as pd
import os
import glob
from bs4 import BeautifulSoup
from pathlib import Path
import re


####################
###Merge Datasets###
####################
def clean_multi(inpath, outpath):
    os.chdir(inpath)
    all_files = glob.glob("*.csv")
    
    #li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        #li.append(df)
    
        #df = pd.concat(li, axis=0, ignore_index=True)
        
        #ResultId, Link, Source.Name, Document.Content
        df_full = df[['ResultId','Title','Date','Document.Content','Source.Name']]
        df_full_nona = df_full[df_full['Document.Content'].notna()]
        
        article_index = list(range(len(df_full_nona)))
        df_full_nona['article_index'] = article_index 
        
        
        ################
        ###Clean Text###
        ################
        
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
                
        paragraphs = [l for l in paragraphs if len(l) > 200 & len(l) < 32500] #remove paragraphs with less than 200 characters
        temp_df = pd.DataFrame(list(zip(paragraphs,article_number)), columns = ['paragraphs','article_number'])
        df = temp_df.merge(df_full_nona, left_on='article_number', right_on='article_index')
        df.insert(0, 'code', '')
        #df = df[df['paragraphs'].str.contains('covert', case = False)] 
        
        
        df = df.drop_duplicates(subset = ['Source.Name','paragraphs'])
        df = df.drop(columns = ['Document.Content','article_number'], axis = 1)
        df['par_number'] = df.index
        
        file_path = Path(outpath + 'ForCode_1_' + filename)
        if file_path.is_file():
            res = [f for f in os.listdir(outpath) if re.search(r'ForCode_\d_' + filename, f)]
            df_temp = pd.read_csv(file_path)
            df_code = df[~df['par_number'].isin(df_temp['par_number'])].sample(1000)
            print('Export ' + 'ForCode_' + str((len(res) + 1)) + '_' + filename + ' for hand-coding to outpath')
            df_code.to_csv(outpath + 'ForCode_' + str((len(res) + 1)) + '_' + filename, index = False)
        else:
            df_code = df.sample(1000)
            df_code.to_csv(outpath + 'ForCode_1_' + filename, index = False)
            print('Export full dataset to outpath')
            df.to_csv(outpath + 'Full_' + filename, index = False)
            
     
        
  
       

