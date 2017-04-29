#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:30:46 2017

@author: amin
"""

import numpy as np
import pandas as pd

##############################################################################
#                         """ READ INPUT DATA """
X = np.loadtxt("../data/input", delimiter=",")
y = np.loadtxt("../data/target", delimiter=",")
ids = np.loadtxt("../data/ids", delimiter=",")
num_train = 49352

###############################################################################
#Ada = np.loadtxt('../data/stack_ada', delimiter=",")
#Ext = np.loadtxt('../data/stack_ext', delimiter=",")
#Ran = np.loadtxt('../data/stack_ran', delimiter=",")
#Knn = np.loadtxt('../data/stack_knn', delimiter=",")
#Svm1 = np.loadtxt('../data/stack_svm_rbf', delimiter=",")
#Svm2 = np.loadtxt('../data/stack_svm_poly', delimiter=",")
#Lg  = np.loadtxt('../data/stack_lg', delimiter=",")
#Nnt = np.loadtxt('../data/stack_nnt', delimiter=",")
#Gbc = np.loadtxt('../data/stack_gbc', delimiter=",")
#Lgb = np.loadtxt('../data/stack_lgb', delimiter=",")
#Xgb = np.loadtxt('../data/stack_xgb', delimiter=",")
###############################################################################
## layer2
#Ran2 = np.loadtxt('../data/stack2_ran', delimiter=",")
#Nnt2 = np.loadtxt('../data/stack2_nnt', delimiter=",")
#Gbc2 = np.loadtxt('../data/stack2_gbc', delimiter=",")
#Lgb2 = np.loadtxt('../data/stack2_lgb', delimiter=",")
#Xgb2 = np.loadtxt('../data/stack2_xgb', delimiter=",")
###############################################################################
#XX = np.column_stack((X, Ada, Ext, Ran, Knn, Svm1, Svm2, Lg, Nnt, Gbc, Lgb, Xgb))
##                      Ran2, Nnt2, Gbc2, Lgb2, Xgb2))
#
#train_old = np.loadtxt('../data/train_old.csv', delimiter=",")
#test_old = np.loadtxt('../data/test_old.csv', delimiter=",")
#
train_file="train_stacknet.csv"
test_file="test_stacknet.csv"
#
## stack target to train
#S_train = np.column_stack((train_old,XX[:num_train]))
## stack id to test
#S_test = np.column_stack((test_old,XX[num_train:]))
S_train = np.column_stack((y,X[:num_train]))

S_test = np.column_stack((ids,X[num_train:]))

#export to txt files (, del.)
print ("exporting files")
np.savetxt(train_file, S_train, delimiter=",", fmt='%.5f')
np.savetxt(test_file, S_test, delimiter=",", fmt='%.5f')
