# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:33:31 2022

@author: chase
"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from hyperopt import  STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import f1_score, accuracy_score


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py

# SVC grid searches
tuned_parameters =[
    {'kernel': ['rbf'],
     'gamma': [1e-3, 1e-4],
     'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 
     'C': [1, 10, 100, 1000]},
    ]
#Grid search for mutltiple scores
def svc_gridsearch(scores, X_train, y_train):
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(
            SVC(class_weight = {0:.24, 1:.76}), tuned_parameters, 
            scoring=score, cv = 5,
            n_jobs = -1
            )
        clf.fit(X_train, y_train)
        svc_best_params = clf.best_params_
    return(svc_best_params)


#Grid search for only one score
def svc_gridsearch_sens(score, X_train, y_train):
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(
        SVC(class_weight = {0:.24, 1:.76}), 
        tuned_parameters, scoring=score, cv = 5,
        n_jobs = -1
        )
    clf.fit(X_train, y_train)
    svc_best_params = clf.best_params_
    return(svc_best_params)


# Random Forest grid searches
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


def rf_gridsearch(scores, X_train, y_train):
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = RandomizedSearchCV(
            RandomForestClassifier(class_weight = {0:.24,1:.76}, n_jobs = -1), 
            random_grid, 
            scoring=score, 
            cv = 5,
            n_iter = 100,
            n_jobs = 1,
            verbose = 3
            )
        clf.fit(X_train, y_train)
        rf_best_params = clf.best_params_
        return(rf_best_params)
     
        
def rf_gridsearch_sens(score, X_train, y_train):
        print("# Tuning hyper-parameters for %s" % score)
        clf = RandomizedSearchCV(
            RandomForestClassifier(class_weight = {0:.1, 1:.9}),
            random_grid, 
            scoring='%s_macro' % score, 
            cv = 5,
            n_iter = 50
            )
        clf.fit(X_train, y_train)
        rf_best_params = clf.best_params_
        return(rf_best_params)
    

# Multinomial Naive Bayes Grid Searches
grid_params = {
  'alpha': np.linspace(0.5, 1.5, 6),
  'fit_prior': [True, False],  
}

def nb_gridsearch(scores, X_train, y_train):
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(
            MultinomialNB(), 
            grid_params, scoring='%s_macro' % score,
            cv = 5
        )
        clf.fit(X_train, y_train)
        mnb_best_params = clf.best_params_
        return(mnb_best_params)
    
def nb_gridsearch_sens(score, X_train, y_train):
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(
            MultinomialNB(), 
            grid_params, scoring='%s_macro' % score,
            cv = 5
        )
        clf.fit(X_train, y_train)
        mnb_best_params = clf.best_params_
        return(mnb_best_params)
    
# Gridsearch for Logistic Regression
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
solver = ['saga']

param_grid = dict(penalty=penalty,
                   C=C,
                   solver=solver)
def lr_gridsearch(scores, X_train, y_train):
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(estimator=LogisticRegression(class_weight = {0:.1, 1:.9}, max_iter = 1000),
                           param_grid=param_grid, 
                           scoring= '%s_macro' % score,
                           cv = 5)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        lr_best_params = clf.best_params_
        return(lr_best_params)
    
def lr_gridsearch_sens(score, X_train, y_train):
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(estimator=LogisticRegression(class_weight = {0:.1, 1:.9}, max_iter=1000),
                       param_grid=param_grid, 
                       scoring= '%s_macro' % score,
                       cv = 5)
    clf.fit(X_train, y_train)
    lr_best_params = clf.best_params_
    return(lr_best_params)

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
    'n_estimators': 1000,
    'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
    'seed': 123,
    'scale_pos_weight': hp.uniform('scale_pos_weight',2,9)
}


def xgb_gridsearch(X_train, y_train, X_test, y_test):
    def objective(space):
        clf=xgb.XGBClassifier(
                        scale_pos_weight=space['scale_pos_weight'], 
                        n_estimators =space['n_estimators'], 
                        max_depth = int(space['max_depth']), 
                        gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),
                        min_child_weight=int(space['min_child_weight']), 
                        colsample_bytree=int(space['colsample_bytree']),
                        n_jobs = -1,
                        early_stopping_rounds = 50,
                        eval_metric = "auc")
        
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
                            max_evals = 100,
                            trials = trials)
    best_hyperparams['max_depth'] = round(best_hyperparams['max_depth'])
    return(best_hyperparams)
       

