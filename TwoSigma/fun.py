# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 16:08:07 2017

@author: Amin
"""
import xgboost as xgb
# cross validation def to tune parameter
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, 
           eta=0.02, max_depth=4, n_estimators=100, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['n_estimators'] = n_estimators
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['num_threads'] = 6

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

def XGBCV(train_X, train_y, eta=0.1, max_depth=4, n_estimators=100, nfold=5, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['n_estimators'] = n_estimators
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1.2
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    param['num_threads'] = 6
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    model = xgb.cv(plst, xgtrain, num_rounds, nfold=nfold, verbose_eval=50, early_stopping_rounds=20)

    num_run = model.shape[0]

    return model.iloc[num_run-1,0], model.iloc[num_run-1,2], num_run

# train xgboost and prediction
def XGB(train_X, train_y, test_X, eta=0.1, max_depth=4, n_estimators=100, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['n_estimators'] = n_estimators
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1.2
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    param['num_threads'] = 6
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [(xgtrain,'train')]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, verbose_eval=50, early_stopping_rounds=20)

    xgtest = xgb.DMatrix(test_X)
    pred = model.predict(xgtest)

    return pred, model

########################################################################################
import lightgbm as lgb
def runLGB(train_X, train_y, test_X, test_y=None, feature_names=None, 
           eta=0.1, depth=4, seed_val=0, num_rounds=10000):
    param = {}
    param['task'] = 'train'
    param['boosting_type'] = 'gbdt'
    param['objective'] = 'multiclass'
    param['metric'] = 'multi_logloss',
    param['num_classes'] = 3
    param['learning_rate'] = eta
    param['num_leaves'] = 2**depth
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

def LGBCV(train_X, train_y, eta=0.1, depth=4, nfold=5, seed_val=0, num_rounds=10000):
    param = {}
    param['task'] = 'train'
    param['boosting_type'] = 'gbdt'
    param['objective'] = 'multiclass'
    param['metric'] = 'multi_logloss',
    param['num_classes'] = 3
    param['learning_rate'] = eta
    param['num_leaves'] = 2**depth
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

    xgtrain = lgb.Dataset(train_X, train_y)

    model = lgb.cv(param, xgtrain, num_boost_round=num_rounds, nfold=5, verbose_eval=50, early_stopping_rounds=20)

    num_run = len(model.values()[0])

    return model.values()[0][num_run-1], num_run

def LGB(train_X, train_y, test_X, eta=0.1, depth=4, seed_val=0, num_rounds=10000):
    param = {}
    param['task'] = 'train'
    param['boosting_type'] = 'gbdt'
    param['objective'] = 'multiclass'
    param['metric'] = 'multi_logloss',
    param['num_classes'] = 3
    param['learning_rate'] = eta
    param['num_leaves'] = 2**depth
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

    xgtrain = lgb.Dataset(train_X, train_y)

    model = lgb.train(param, xgtrain, num_boost_round=num_rounds, verbose_eval=50)

    pred = model.predict(test_X)

    return pred, model



