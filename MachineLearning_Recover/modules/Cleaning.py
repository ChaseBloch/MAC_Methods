# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:40:45 2021

@author: chase
"""

import pandas as pd
import os
import glob
import re
import sys
from bs4 import BeautifulSoup
from pathlib import Path


####################
###Merge Datasets###
####################
def clean_multi(inpath, outpath):
    ori_dir = os.getcwd()
    os.chdir(inpath)
    all_files = glob.glob("*.csv")
   
    # Clean each file in the inpath and output the cleaned files.
    m = 1
    for filename in all_files:
        
        # Exit if the file already exists to avoid overwriting.
        file_path = Path(outpath + 'Full_' + filename)
        if file_path.is_file():
            sys.exit("File already exists, use 'nodup_sample' instead")
        else:
            df = pd.read_csv(filename, index_col=None, header=0)
            df_full = df[['ResultId',
                          'Title',
                          'Date',
                          'Document.Content',
                          'Source.Name'
                          ]]
            
            # Remove NA values and add an index for each article.
            df_full_nona = df_full[df_full['Document.Content'].notna()]
            df_full_nona['article_index'] = [
                str(m) + str(x) for x in list(range(len(df_full_nona)))
                ]
            
            # Clean the text with Beautiful Soup.
            content_list = df_full_nona['Document.Content'].tolist()
            paragraphs = []
            article_number = []
            for i in range(len(content_list)):
                soup = BeautifulSoup(
                    content_list[i],'html.parser').find_all("p")
                sentences = []
                for j in range(len(soup)):
                    sentences.append(soup[j].get_text())
                    groups = list(zip(*[iter(sentences)]*3))
                for k in range(len(groups)):
                    paragraphs.append(
                        ''.join([idx for tup in groups[k] for idx in tup])  
                        )
                    article_number.append(str(m) + str(i))
            
            # Remove paragraphs with less than 200 characters.   
            paragraphs = [
                l for l in paragraphs if len(l) > 200 & len(l) < 32000
                ]
            
            #Create dataset from cleaned news articles.
            temp_df = pd.DataFrame(
                list(zip(paragraphs,article_number)), 
                columns = ['paragraphs','article_number']
                )
            df = temp_df.merge(
                df_full_nona, 
                left_on='article_number', 
                right_on='article_index'
                )
            df.insert(0, 'code', '')
            
            # Use the following line to find specific substrings.
            #df = df[df['paragraphs'].str.contains('covert', case = False)] 
            
            # Drop duplicates and unnecessary columns. 
            df = df.drop_duplicates(subset = ['Source.Name','paragraphs'])
            df = df.drop(
                columns = ['Document.Content','article_number'], 
                axis = 1
                )
            
            #Create an index for each article segment.
            df['par_number'] = [str(m) + str(x) for x in list(df.index)]
            
            print('Export Full_' + filename + ' dataset to outpath')
            df.to_csv(outpath + 'Full_' + filename, index = False)
            
            m = m + 1  # Incrememnt for indeces
    os.chdir(ori_dir) # Revert to original directory
       
     
def nodup_sample(inpath, outpath, num_code):
    ori_dir = os.getcwd()
    os.chdir(inpath)
    all_files = glob.glob("*.csv")
    
    # Get the length of each imported dataset.
    lengths_temp = []
    for filename in all_files:
        lengths_temp.append(len(pd.read_csv(filename)))
    
    # Get the proportional length of each dataset from the sum of datasets.
    lengths = []
    for length in lengths_temp:
        lengths.append(round((length/sum(lengths_temp))*num_code))
    
    # Draw a proportional sample of cases from each dataset for training.
    i = 0
    for filename in all_files:
        file_path = Path(outpath + 'ForCode_1_' + filename)
        
        # If the file already exists, draw a new sample of new cases.
        if file_path.is_file():
             res = [f for f in os.listdir(outpath) 
                    if re.search(r'ForCode_\d_' + filename, f)]
             li = []
             for file in res:
                 df = pd.read_csv(outpath + file, index_col=None, header=0)
                 li.append(df)
                 df = pd.concat(li, axis=0, ignore_index=True)
             df_temp = pd.read_csv(outpath + 'Full_' + filename)
             df_code = df_temp[
                 ~df_temp['par_number'].isin(df['par_number'])
                 ].sample(lengths[i])
             print('Export ' 
                   + 'ForCode_' 
                   + str((len(res) + 1)) 
                   + '_' + filename 
                   + ' for hand-coding to outpath'
                   )
             df_code.to_csv(
                 outpath + 'ForCode_' 
                 + str((len(res) + 1))
                 + '_' + filename, index = False
                 )        
             i = i + 1
        
        # If the file does not already exist, draw original sample. 
        else: 
            df_temp = pd.read_csv(outpath + 'Full_' + filename)
            df_code = df_temp.sample(lengths[i])
            df_code.to_csv(outpath + 'ForCode_1_' + filename, index = False)
            i = i + 1
    os.chdir(ori_dir)  # Revert to original directory 
     
        
  
       

