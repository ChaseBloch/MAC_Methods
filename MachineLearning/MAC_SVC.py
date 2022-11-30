# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import os
import re
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, precision_score, 
                             recall_score, confusion_matrix, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)


os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation'
         r'\MACoding\MAC_Methods\MachineLearning')

# Local modules
from modules.preprocessing_decisions import svc_sensitivity, preprocess_plots
from modules.cleaning import clean_multi, nodup_sample
from modules.svc_gridsearch import svc_gridsearch_sens
from modules.nltk_stemmer import StemmedTfidfVectorizer, StemmedCountVectorizer



outpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation/MACoding/'
           r'MAC_Methods/MachineLearning/Downloading&Coding/Exported/')
inpath = (r'C:/Users/chase/GDrive/GD_Work/Dissertation\MACoding/'
          r'MAC_Methods/MachineLearning/Downloading&Coding/Downloaded/')

# Silence warnings caused by models without good training data.
warnings.filterwarnings('ignore') 

# Importing and preparing datasets
clean_multi(inpath, outpath)
nodup_sample(inpath, outpath, 1500)

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
df = df[df['code'].notna()]
df_test = df_test[~df_test.par_number.isin(df.par_number)]

#Run sensitivity analysis
scores = ['f1', 'precision']
preprocessing_scores = svc_sensitivity(df, scores)
preprocess_plots(preprocessing_scores, scores)



#Run Full Model
vec = StemmedCountVectorizer(
    #norm='l2',
    encoding='utf-8', 
    stop_words='english',
    ngram_range = (1,1),
    max_df = .8, min_df = 3, max_features=60000,
    strip_accents = 'ascii', lowercase=True
    )

features = vec.fit_transform(df.paragraphs).toarray()
labels = df.code
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, random_state = 1234,test_size=0.3
    )

score = 'f1'
SVC_BestParams = svc_gridsearch_sens(score, X_train, y_train)

clf = SVC(**SVC_BestParams, class_weight = {0:.05, 1:.95}).fit(X_train, y_train)

pred1 = clf.predict(X_test)
print(accuracy_score(y_test, pred1))
print(f1_score(y_test, pred1, average="macro"))
print(precision_score(y_test, pred1, average="macro"))
print(recall_score(y_test, pred1, average="macro"))
cm = confusion_matrix(y_test, pred1)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

cm_display.plot()
plt.show()

