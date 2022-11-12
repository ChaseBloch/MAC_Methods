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
import sys


####################
###Merge Datasets###
####################
def clean_multi(inpath, outpath):
    ori_dir = os.getcwd()
    os.chdir(inpath)
    all_files = glob.glob("*.csv")
    
    #li = []
    m = 1
    for filename in all_files:
        
        file_path = Path(outpath + 'Full_' + filename)
        if file_path.is_file():
            sys.exit("File already exists, use 'nodup_sample' instead")
            
        else:
            
            df = pd.read_csv(filename, index_col=None, header=0)
            
            #ResultId, Link, Source.Name, Document.Content
            df_full = df[['ResultId','Title','Date','Document.Content','Source.Name']]
            df_full_nona = df_full[df_full['Document.Content'].notna()]
            
            df_full_nona['article_index'] = [str(m) + str(x) for x in list(range(len(df_full_nona)))]
            
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
                    article_number.append(str(m) + str(i))
                    
            paragraphs = [l for l in paragraphs if len(l) > 200 & len(l) < 32500] #remove paragraphs with less than 200 characters
            temp_df = pd.DataFrame(list(zip(paragraphs,article_number)), columns = ['paragraphs','article_number'])
            df = temp_df.merge(df_full_nona, left_on='article_number', right_on='article_index')
            df.insert(0, 'code', '')
            #df = df[df['paragraphs'].str.contains('covert', case = False)] 
            
            
            df = df.drop_duplicates(subset = ['Source.Name','paragraphs'])
            df = df.drop(columns = ['Document.Content','article_number'], axis = 1)
            df['par_number'] = [str(m) + str(x) for x in list(df.index)]
            
            print('Export Full_' + filename + ' dataset to outpath')
            df.to_csv(outpath + 'Full_' + filename, index = False)
            
            m = m + 1
    os.chdir(ori_dir)
            
def nodup_sample(inpath, outpath, num_code):
    ori_dir = os.getcwd()
    os.chdir(inpath)
    all_files = glob.glob("*.csv")
    
    lengths_temp = []
    for filename in all_files:
        lengths_temp.append(len(pd.read_csv(filename)))
    
    lengths = []
    for length in lengths_temp:
        lengths.append(round((length/sum(lengths_temp))*num_code))
    
    i = 0
    for filename in all_files:
        file_path = Path(outpath + 'ForCode_1_' + filename)
        if file_path.is_file():
             res = [f for f in os.listdir(outpath) if re.search(r'ForCode_\d_' + filename, f)]
             li = []
             for file in res:
                 df = pd.read_csv(outpath + file, index_col=None, header=0)
                 li.append(df)
                 df = pd.concat(li, axis=0, ignore_index=True)
             df_temp = pd.read_csv(outpath + 'Full_' + filename)
             df_code = df_temp[~df_temp['par_number'].isin(df['par_number'])].sample(lengths[i])
             print('Export ' + 'ForCode_' + str((len(res) + 1)) + '_' + filename + ' for hand-coding to outpath')
             df_code.to_csv(outpath + 'ForCode_' + str((len(res) + 1)) + '_' + filename, index = False)        
             i = i + 1
        else: 
            df_temp = pd.read_csv(outpath + 'Full_' + filename)
            df_code = df_temp.sample(lengths[i])
            df_code.to_csv(outpath + 'ForCode_1_' + filename, index = False)
            i = i + 1
    os.chdir(ori_dir)
     
        
  
       

