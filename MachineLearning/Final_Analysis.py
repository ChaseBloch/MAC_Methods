# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:00:33 2022

@author: chase
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, precision_score, 
                             recall_score, confusion_matrix, accuracy_score, 
                             ConfusionMatrixDisplay)
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation'
         r'\MACoding\MAC_Methods\MachineLearning')

# Local modules
from modules.gridsearches import svc_gridsearch
from modules.nltk_stemmer import StemmedCountVectorizer, ProperNounExtractor

df = pd.read_csv(r'Downloading&Coding/Exported/df_train.csv')
df_test = pd.read_csv(r'Downloading&Coding/Exported/df_test.csv')

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
SVC_BestParams = svc_gridsearch(score, X_train, y_train)

clf = SVC(**SVC_BestParams, class_weight = {0:.1, 1:.9}, probability = True).fit(X_train, y_train)

pred1 = clf.predict(X_test)
predicted_prob = clf.predict_proba(X_test)
print(accuracy_score(y_test, pred1))
print(f1_score(y_test, pred1, average="macro"))
print(precision_score(y_test, pred1, average="macro"))
print(recall_score(y_test, pred1, average="macro"))
cm = confusion_matrix(y_test, pred1)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

cm_display.plot()
plt.show()


#Confusion matrix after low predicted probability removed.
counter = np.arange(0.01,0.96,0.01)
lcount = len(counter)
f1=[]
prec=[]
rec=[]
obs=[]
obs_perc=[]
acc=[]
for i in range(lcount):

    indexNames_1 = np.where(predicted_prob>counter[i])
    indexNames_col_1 = np.unique(indexNames_1[0])
    y_test_new = np.take(y_test,indexNames_col_1)
    y_pred_new = np.take(pred1,indexNames_col_1)

    obs.append(len(indexNames_col_1))
    obs_perc.append(len(indexNames_col_1)/len(X_test))
    acc.append(metrics.accuracy_score(y_test_new,y_pred_new))
    f1.append(metrics.f1_score(y_test_new,y_pred_new,average='macro'))
    prec.append(metrics.precision_score(y_test_new,y_pred_new,average='macro'))
    rec.append(metrics.recall_score(y_test_new,y_pred_new,average='macro'))
    
metr_df = pd.DataFrame(list(zip(counter, obs, obs_perc, acc, f1, prec, rec)), columns = ('counter','obs', 'obs_perc', 'acc', 'f1', 'prec', 'rec'))

#Figure 4 F1 Score as Predicted Probabilty Threshold Increases
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Predicted Probability Threshold')
ax1.set_ylabel('F1', color=color)
ax1.plot(counter, f1, color=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Number of Observations', color=color)  # we already handled the x-label with ax1
ax2.plot(counter, obs, color=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('f1score.pdf',bbox_inches='tight')
plt.show()


#Test on full set
features_test = vec.transform(df_test.paragraphs).toarray()
pred_test = clf.predict(features_test)
predicted_prob_test = clf.predict_proba(features_test)
df_test['code'] = pred_test
threshold = .77 # Set this value to reach desired threshold
coded_index = np.where(predicted_prob_test > threshold)[0]
coded = pd.concat([df_test.iloc[coded_index], df])
for_hand = df_test.iloc[~df_test.index.isin(coded_index)]


df_coded = coded[coded['code'] == 1]

temp = df_coded.groupby(['article_index', 'Title','Date','Source.Name'])['paragraphs'].agg('\n'.join).reset_index()
temp['year'] = [int(x[0:4]) for x in temp['Date']]
                
prop_nouns = [] 
for paragraph in temp['paragraphs']:
    prop_nouns.append(list(set(ProperNounExtractor(paragraph))))
    
temp['prop_nouns'] = prop_nouns
df_prop = temp.explode('prop_nouns').reset_index(drop=True)
df_prop['paragraphs'] = [x[0:32000] for x in df_prop['paragraphs']]

removals = df_prop['prop_nouns'].value_counts().reset_index()
#removals.to_csv('Downloading&Coding/Exported/removals.csv')

df_removals = pd.read_csv('Downloading&Coding/Exported/removals.csv')
df_removals = df_removals[df_removals['country'] == 1]

df_propmerge = df_removals.merge(df_prop, left_on = 'index', right_on = 'prop_nouns')
df_propmerge = df_propmerge.drop_duplicates(subset=['country_name','article_index'])
df_propmerge['paragraphs'] = df_propmerge[['article_index', 'paragraphs']].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1)

df_final = df_propmerge.groupby(['year','country_name'])['paragraphs'].agg('\n--------------------------------------------------\n'.join).reset_index()
df_final['output'] = df_final[['year', 'country_name']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df_final['output'] = df_final[['output','paragraphs']].apply(lambda row: ':\n'.join(row.values.astype(str)), axis=1)

df_final['output'].to_csv('Downloading&Coding/Exported/final_articles.txt', sep =' ', index = False)
df_final[['year','country_name']].to_csv('Downloading&Coding/Exported/final_articles.csv')