# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:37:13 2017

@author: Amin
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from fun import runLGB, runXGB

##############################################################################
#                         """ READ INPUT DATA """
X = np.loadtxt("../data2/input", delimiter=",")
y = np.loadtxt("../data2/target", delimiter=",")
ids = np.loadtxt("../data2/ids", delimiter=",")
num_train = 49352

##############################################################################
Ada = np.loadtxt('../data2/stack_ada', delimiter=",")
Ext = np.loadtxt('../data2/stack_ext', delimiter=",")
Ran = np.loadtxt('../data2/stack_ran', delimiter=",")
Knn = np.loadtxt('../data2/stack_knn', delimiter=",")
Svm1 = np.loadtxt('../data2/stack_svm_rbf', delimiter=",")
Svm2 = np.loadtxt('../data2/stack_svm_poly', delimiter=",")
Lg  = np.loadtxt('../data2/stack_lg', delimiter=",")
Nnt = np.loadtxt('../data2/stack_nnt', delimiter=",")
Gbc = np.loadtxt('../data2/stack_gbc', delimiter=",")
Lgb = np.loadtxt('../data2/stack_lgb', delimiter=",")
Xgb = np.loadtxt('../data2/stack_xgb', delimiter=",")
##############################################################################
Ran2 = np.loadtxt('../data2/stack_ran2', delimiter=",")
Nnt2 = np.loadtxt('../data2/stack_nnt2', delimiter=",")
Gbc2 = np.loadtxt('../data2/stack_gbc2', delimiter=",")
Lgb2 = np.loadtxt('../data2/stack_lgb2', delimiter=",")
Xgb2 = np.loadtxt('../data2/stack_xgb2', delimiter=",")
##############################################################################
XX = np.column_stack((X, Ada, Ext, Ran, Knn, Svm1, Svm2, Lg, Nnt, Gbc, Lgb, Xgb,
                     Ran2, Nnt2, Gbc2, Lgb2, Xgb2))

stack = np.zeros((124011, 3))

# Lightgbm
print("Lightgbm")
stack.fill(0.0)
scores = []
Nrun = 0
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=23)
for train_index, test_index in kfolder:
    X_train, X_cv = XX[train_index], XX[test_index]
    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
    pred, model = runLGB(X_train, y_train, X_cv, test_y=y_cv, seed_val=0, eta=0.005, depth=5, num_rounds=10000)
    score = log_loss(y_cv, pred)
    print score
    scores.append(score)
    Nrun += model.best_iteration
    #save the results
    no=0
    for index in test_index:
        for d in range (0,3):
            stack[index][d]=(pred[no][d])
        no+=1

print scores
print "Average scores = " + str(np.mean(scores))


Nrun = int(Nrun/5)+100
pred, model = runLGB(XX[:num_train], y, XX[num_train:], seed_val=0, num_rounds=Nrun)
no=0
for index in range(num_train, XX.shape[0]):
    for d in range (0,3):
        stack[index][d]=(pred[no][d])
    no+=1

#export to txt files (, del.)
print ("exporting files")
np.savetxt("stack2_lgb", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = map(int,ids)
pred_df.to_csv("Lgb.csv", index=False)


# XGBoost
print("XGBoost")
stack.fill(0.0)
scores = []
Nrun = 0
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=25)
for train_index, test_index in kfolder:
    X_train, X_cv = XX[train_index], XX[test_index]
    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
    pred, model = runXGB(X_train, y_train, X_cv, test_y=y_cv, seed_val=0, eta=0.02, max_depth=6, num_rounds=10000)
    score = log_loss(y_cv, pred)
    print score
    scores.append(score)
    Nrun += model.best_iteration
    #save the results
    no=0
    for index in test_index:
        for d in range (0,3):
            stack[index][d]=(pred[no][d])
        no+=1

print scores
print "Average scores = " + str(np.mean(scores))


Nrun = int(Nrun/5)+100
pred, model = runXGB(XX[:num_train], y, XX[num_train:], seed_val=0, num_rounds=Nrun)
no=0
for index in range(num_train, XX.shape[0]):
    for d in range (0,3):
        stack[index][d]=(pred[no][d])
    no+=1

#export to txt files (, del.)
print ("exporting files")
np.savetxt("stack_xgb", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = map(int,ids)
pred_df.to_csv("Xgb.csv", index=False)

# MLPClassifier
print("MLPClassifier")
stack.fill(0.0)
scores = []
clf = MLPClassifier(hidden_layer_sizes=2, activation='logistic')
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=12)
for train_index, test_index in kfolder:
    X_train, X_cv = XX[train_index], XX[test_index]
    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_cv)
    score = log_loss(y_cv, pred); print score
    scores.append(score)
    #save the results
    no=0
    for index in test_index:
        for d in range (0,3):
            stack[index][d]=(pred[no][d])
        no+=1

print scores
print "Average scores = " + str(np.mean(scores))

clf.fit(XX[:num_train], y)
pred = clf.predict_proba(XX[num_train:])
no=0
for index in range(num_train, XX.shape[0]):
#    print index
    for d in range (0,3):
        stack[index][d]=(pred[no][d])
    no+=1

#export to txt files (, del.)
print ("exporting files")
np.savetxt("stack_nnt", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = map(int,ids)
pred_df.to_csv("Nnt.csv", index=False)


# RandomForestClassifier
print("RandomForestClassifier")
stack.fill(0.0)
scores = []
n_est = 200; depth = 18
clf = RandomForestClassifier(max_depth=depth, n_estimators=n_est, random_state=0)
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=3)
for train_index, test_index in kfolder:
    X_train, X_cv = XX[train_index], XX[test_index]
    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_cv)
    score = log_loss(y_cv, pred); print score
    scores.append(score)
    #save the results
    no=0
    for index in test_index:
        for d in range (0,3):
            stack[index][d]=(pred[no][d])
        no+=1

print scores
print "Average scores = " + str(np.mean(scores))

clf.fit(XX[:num_train], y)
pred = clf.predict_proba(XX[num_train:])
no=0
for index in range(num_train, XX.shape[0]):
#    print index
    for d in range (0,3):
        stack[index][d]=(pred[no][d])
    no+=1

#export to txt files (, del.)
print ("exporting files ExtraTreesClassifier")
np.savetxt("stack_ran", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = map(int,ids)
pred_df.to_csv("Ran.csv", index=False)


# GradientBoostingClassifier
print("GradientBoostingClassifier")
stack.fill(0.0)
scores = []
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=13)
for train_index, test_index in kfolder:
    X_train, X_cv = XX[train_index], XX[test_index]
    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_cv)
    score = log_loss(y_cv, pred); print score
    scores.append(score)
    #save the results
    no=0
    for index in test_index:
        for d in range (0,3):
            stack[index][d]=(pred[no][d])
        no+=1

print scores
print "Average scores = " + str(np.mean(scores))

clf.fit(XX[:num_train], y)
pred = clf.predict_proba(XX[num_train:])
no=0
for index in range(num_train, XX.shape[0]):
#    print index
    for d in range (0,3):
        stack[index][d]=(pred[no][d])
    no+=1

#export to txt files (, del.)
print ("exporting files")
np.savetxt("stack_gbc", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = map(int,ids)
pred_df.to_csv("Gbc.csv", index=False)

