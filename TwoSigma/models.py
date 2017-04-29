#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:35:48 2017

@author: amin
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

############################################################################################################
import xgboost as xgb
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.02
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['num_threads'] = 4
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20, verbose_eval=50)
    else:
        model = xgb.train(plst, xgtrain, num_rounds)

    xgtest = xgb.DMatrix(test_X)
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
##############################################################################################################
import lightgbm as lgb
def runLGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['task'] = 'train'
    param['boosting_type'] = 'gbdt'
    param['objective'] = 'multiclass'
    param['metric'] = 'multi_logloss',
    param['num_classes'] = 3
    param['learning_rate'] = 0.005
    param['num_leaves'] = 2**5
    param['max_depth'] = -1
    param['silent'] = True
    param['num_class'] = 3
    param['min_child_weight'] = 1.
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['feature_fraction']= 0.82,
    param['bagging_fraction']= 0.8,
    param['bagging_freq']= 5,
    param['verbose']= 0
    param['num_threads'] = 6

    xgtrain = lgb.Dataset(train_X, label=train_y)

    if test_y is not None:
        xgtest = lgb.Dataset(test_X, label=test_y, reference=xgtrain)
        model = lgb.train(param, xgtrain, num_rounds, valid_sets=xgtest, early_stopping_rounds=20, verbose_eval=50)
    else:
        model = lgb.train(param, xgtrain, num_rounds)

    pred_test_y = model.predict(test_X)
    return pred_test_y, model

##############################################################################
#                         """ READ INPUT DATA """
XX = np.loadtxt("../data3/input", delimiter=",")
y = np.loadtxt("../data3/target", delimiter=",")
y = map(int,y)
ids = np.loadtxt("../data3/ids", delimiter=",")
num_train = 49352

# stack target to train
#S_train = np.column_stack((y,XX[:num_train]))
# stack id to test
#S_test = np.column_stack((ids,XX[num_train:]))
##############################################################################
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

##############################################################################
stack = np.zeros((124011, 3))
#
# AdaBoostClassifier
print("AdaBoostClassifier")
stack.fill(0.0)
n_est = 100; eta = 0.001; scores = []
clf = AdaBoostClassifier(learning_rate=eta, n_estimators=n_est, random_state=0)
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=1)
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
print ("exporting files AdaBoostClassifier")
np.savetxt("stack_ada", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = ids
pred_df.to_csv("Ada.csv", index=False)


# ExtraTreesClassifier
print("ExtraTreesClassifier")
stack.fill(0.0)
scores = []
n_est = 200; depth = 25
clf = ExtraTreesClassifier(max_depth=depth, n_estimators=n_est, random_state=0)
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=2)
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
np.savetxt("stack_ext", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = ids
pred_df.to_csv("Ext.csv", index=False)


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
pred_df["listing_id"] = ids
pred_df.to_csv("Ran.csv", index=False)

# LogisticRegression
print("LogisticRegression")
stack.fill(0.0)
scores = []
clf = LogisticRegression()
kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=8)
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
np.savetxt("stack_lg", stack, delimiter=",", fmt='%.6f')

pred_df = pd.DataFrame(pred)
pred_df.columns = ["high", "medium", "low"]
pred_df["listing_id"] = ids
pred_df.to_csv("Lg.csv", index=False)

# MLPClassifier
print("MLPClassifier")
stack.fill(0.0)
scores = []
clf = MLPClassifier(hidden_layer_sizes=10, activation='logistic')
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
pred_df["listing_id"] = ids
pred_df.to_csv("Nnt.csv", index=False)



## RadiusNeighborsClassifier
#print("RadiusNeighborsClassifier")
#stack.fill(0.0)
#scores = []
#clf = KNeighborsClassifier(n_neighbors=40)
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=4)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    clf.fit(X_train, y_train)
#    pred = clf.predict_proba(X_cv)
#    score = log_loss(y_cv, pred); print score
#    scores.append(score)
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#clf.fit(XX[:num_train], y)
#pred = clf.predict_proba(XX[num_train:])
#no=0
#for index in range(num_train, XX.shape[0]):
##    print index
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files ExtraTreesClassifier")
#np.savetxt("stack_knn", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Knn.csv", index=False)
#
#
## SVM
#print("SupportVectorClassification")
#stack.fill(0.0)
#scores = []
#clf = SVC(kernel='rbf', probability=True)
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=5)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    clf.fit(X_train, y_train)
#    pred = clf.predict_proba(X_cv)
#    score = log_loss(y_cv, pred); print score
#    scores.append(score)
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#clf.fit(XX[:num_train], y)
#pred = clf.predict_proba(XX[num_train:])
#no=0
#for index in range(num_train, XX.shape[0]):
##    print index
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files ExtraTreesClassifier")
#np.savetxt("stack_svm_rbf", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Svm_rbf.csv", index=False)
#
## SVM
#print("SupportVectorClassification")
#stack.fill(0.0)
#scores = []
#clf = SVC(kernel='linear', probability=True)
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=6)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    clf.fit(X_train, y_train)
#    pred = clf.predict_proba(X_cv)
#    score = log_loss(y_cv, pred); print score
#    scores.append(score)
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#clf.fit(XX[:num_train], y)
#pred = clf.predict_proba(XX[num_train:])
#no=0
#for index in range(num_train, XX.shape[0]):
##    print index
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files ExtraTreesClassifier")
#np.savetxt("stack_svm_linear", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Svm_linear.csv", index=False)
#
## SVM
#print("SupportVectorClassification")
#stack.fill(0.0)
#scores = []
#clf = SVC(kernel='poly', probability=True)
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=7)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    clf.fit(X_train, y_train)
#    pred = clf.predict_proba(X_cv)
#    score = log_loss(y_cv, pred); print score
#    scores.append(score)
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#clf.fit(XX[:num_train], y)
#pred = clf.predict_proba(XX[num_train:])
#no=0
#for index in range(num_train, XX.shape[0]):
##    print index
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files ExtraTreesClassifier")
#np.savetxt("stack_svm_poly", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Svm_poly.csv", index=False)
#
#

#
### GaussianProcessClassifier
##print("GaussianProcessClassifier")
##stack.fill(0.0)
##scores = []
##clf = GaussianProcessClassifier( warm_start=True)
##kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=9)
##for train_index, test_index in kfolder:
##    X_train, X_cv = XX[train_index], XX[test_index]
##    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
##    clf.fit(X_train, y_train)
##    pred = clf.predict_proba(X_cv)
##    score = log_loss(y_cv, pred); print score
##    scores.append(score)
##    #save the results
##    no=0
##    for index in test_index:
##        for d in range (0,3):
##            stack[index][d]=(pred[no][d])
##        no+=1
##
##print scores
##print "Average scores = " + str(np.mean(scores))
##
##clf.fit(XX[:num_train], y)
##pred = clf.predict_proba(XX[num_train:])
##no=0
##for index in range(num_train, XX.shape[0]):
###    print index
##    for d in range (0,3):
##        stack[index][d]=(pred[no][d])
##    no+=1
##
###export to txt files (, del.)
##print ("exporting files")
##np.savetxt("stack_gau", stack, delimiter=",", fmt='%.6f')
##
##pred_df = pd.DataFrame(pred)
##pred_df.columns = ["high", "medium", "low"]
##pred_df["listing_id"] = ids
##pred_df.to_csv("Gau.csv", index=False)
#
#
## QuadraticDiscriminantAnalysi
#print("QuadraticDiscriminantAnalysi")
#stack.fill(0.0)
#scores = []
#clf = QuadraticDiscriminantAnalysis()
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=10)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    clf.fit(X_train, y_train)
#    pred = clf.predict_proba(X_cv)
#    score = log_loss(y_cv, pred); print score
#    scores.append(score)
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#clf.fit(XX[:num_train], y)
#pred = clf.predict_proba(XX[num_train:])
#no=0
#for index in range(num_train, XX.shape[0]):
##    print index
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files")
#np.savetxt("stack_qua", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Qua.csv", index=False)
#
## GaussianNB
#print("GaussianNB")
#stack.fill(0.0)
#scores = []
#clf = GaussianNB()
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=11)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    clf.fit(X_train, y_train)
#    pred = clf.predict_proba(X_cv)
#    score = log_loss(y_cv, pred); print score
#    scores.append(score)
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#clf.fit(XX[:num_train], y)
#pred = clf.predict_proba(XX[num_train:])
#no=0
#for index in range(num_train, XX.shape[0]):
##    print index
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files")
#np.savetxt("stack_gnb", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Gnb.csv", index=False)
#
#

#
## GradientBoostingClassifier
#print("GradientBoostingClassifier")
#stack.fill(0.0)
#scores = []
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=13)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    clf.fit(X_train, y_train)
#    pred = clf.predict_proba(X_cv)
#    score = log_loss(y_cv, pred); print score
#    scores.append(score)
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#clf.fit(XX[:num_train], y)
#pred = clf.predict_proba(XX[num_train:])
#no=0
#for index in range(num_train, XX.shape[0]):
##    print index
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files")
#np.savetxt("stack_gbc", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Gbc.csv", index=False)
#
#
## Lightgbm
#print("Lightgbm")
#stack.fill(0.0)
#scores = []
#Nrun = 0
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=14)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    pred, model = runLGB(X_train, y_train, X_cv, test_y=y_cv, seed_val=0, num_rounds=10000)
#    print log_loss(y_cv, pred)
#    Nrun += model.best_iteration
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#
#Nrun = int(Nrun/5)+100
#pred, model = runLGB(XX[:num_train], y, XX[num_train:], seed_val=0, num_rounds=Nrun)
#no=0
#for index in range(num_train, XX.shape[0]):
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files")
#np.savetxt("stack_lgb", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Lgb.csv", index=False)
#
#
## XGBoost
#print("XGBoost")
#stack.fill(0.0)
#scores = []
#Nrun = 0
#kfolder=StratifiedKFold(y, n_folds=5 ,shuffle=True, random_state=15)
#for train_index, test_index in kfolder:
#    X_train, X_cv = XX[train_index], XX[test_index]
#    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
#    pred, model = runXGB(X_train, y_train, X_cv, test_y=y_cv, seed_val=0, num_rounds=10000)
#    score = log_loss(y_cv, pred)
#    print score
#    scores.append(score)
#    Nrun += model.best_iteration
#    #save the results
#    no=0
#    for index in test_index:
#        for d in range (0,3):
#            stack[index][d]=(pred[no][d])
#        no+=1
#
#print scores
#print "Average scores = " + str(np.mean(scores))
#
#
#Nrun = int(Nrun/5)+100
#pred, model = runXGB(XX[:num_train], y, XX[num_train:], seed_val=0, num_rounds=Nrun)
#no=0
#for index in range(num_train, XX.shape[0]):
#    for d in range (0,3):
#        stack[index][d]=(pred[no][d])
#    no+=1
#
##export to txt files (, del.)
#print ("exporting files")
#np.savetxt("stack_xgb", stack, delimiter=",", fmt='%.6f')
#
#pred_df = pd.DataFrame(pred)
#pred_df.columns = ["high", "medium", "low"]
#pred_df["listing_id"] = ids
#pred_df.to_csv("Xgb.csv", index=False)




