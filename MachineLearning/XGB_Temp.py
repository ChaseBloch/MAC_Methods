# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:28:36 2022

@author: chase
"""

import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import  STATUS_OK, Trials, fmin, hp, tpe
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn import metrics

os.chdir(r'C:\Users\chase\GDrive\GD_Work\Dissertation\MACoding\MAC_Methods'
         r'\MachineLearning')

from modules.nltk_stemmer import StemmedCountVectorizer

df = pd.read_csv(r'Downloading&Coding/Exported/df_train.csv')
df_test = pd.read_csv(r'Downloading&Coding\Exported/df_test.csv')


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

# Import parameter dictionaries for hyper parameter optimization
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

space = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'max_depth': hp.uniform('max_depth', 1, 100),
    'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.loguniform('gamma', -10, 10),
    'alpha': hp.loguniform('alpha', -10, 10),
    'lambda': hp.loguniform('lambda', -10, 10),
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 1000,
    'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
    'seed': 123,
    'scale_pos_weight': hp.uniform('scale_pos_weight',2,5)
}


def objective(space):
    clf=xgb.XGBClassifier(
                    scale_pos_weight=space['scale_pos_weight'], n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'], early_stopping_rounds=250, eval_metric="auc",
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']), 
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 500,
                        trials = trials)
best_hyperparams['max_depth'] = round(best_hyperparams['max_depth'])

clf = xgb.XGBClassifier(**best_hyperparams).fit(X_train, y_train)

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


counter = np.arange(0.5,0.95,0.001)
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



