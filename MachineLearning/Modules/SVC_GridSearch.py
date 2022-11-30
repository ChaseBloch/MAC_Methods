# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:33:31 2022

@author: chase
"""
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py


#Grid search for mutltiple scores
def svc_gridsearch(scores, X_train, y_train, 
                   tuned_parameters =[
                       {'kernel': ['rbf'],
                        'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]},
                       {'kernel': ['linear'], 
                        'C': [1, 10, 100, 1000]}
                       ]):
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
def svc_gridsearch_sens(score, X_train, y_train, 
                        tuned_parameters =[
                            {'kernel': ['rbf'],
                             'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 
                             'C': [1, 10, 100, 1000]}
                            ]):
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(
<<<<<<< Updated upstream
        SVC(), tuned_parameters, scoring='%s_macro' % score, cv = 10
=======
        SVC(class_weight = {0:.1, 1:.9}), tuned_parameters, scoring='%s_macro' % score, cv = 5
>>>>>>> Stashed changes
    )
    clf.fit(X_train, y_train)
    svc_best_params = clf.best_params_
    return(svc_best_params)
