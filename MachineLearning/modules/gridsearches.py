# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:33:31 2022

@author: chase
"""
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py


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
            SVC(), tuned_parameters, scoring='%s_macro' % score, cv = 5
            )
        clf.fit(X_train, y_train)
        svc_best_params = clf.best_params_
    return(svc_best_params)


#Grid search for only one score
def svc_gridsearch_sens(score, X_train, y_train):
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(
        SVC(class_weight = {0:.1, 1:.9}), 
        tuned_parameters, scoring='%s_macro' % score, cv = 5
        )
    clf.fit(X_train, y_train)
    svc_best_params = clf.best_params_
    return(svc_best_params)





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
            RandomForestClassifier(class_weight = {0:.1, 1:.9}), 
            random_grid, 
            scoring='%s_macro' % score, 
            cv = 5,
            n_iter = 10
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
            n_iter = 10
            )
        clf.fit(X_train, y_train)
        rf_best_params = clf.best_params_
        return(rf_best_params)
