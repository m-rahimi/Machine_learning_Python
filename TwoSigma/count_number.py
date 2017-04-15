# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 13:55:20 2017

@author: Amin
"""

import numpy as np
import pandas as pd
#Our feature construction class will inherit from these two base classes of sklearn.
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class Count_target(BaseEstimator, TransformerMixin):
    """
    Adds the column "manager_skill" to the dataset, based on the Kaggle kernel
    "Improve Perfomances using Manager features" by den3b. The function should
    be usable in scikit-learn pipelines.
    
    Parameters
    ----------
    threshold : Minimum count of rental listings a manager must have in order
                to get his "own" score, otherwise the mean is assigned.

    Attributes
    ----------
    mapping : pandas dataframe
        contains the manager_skill per manager id.
        
    mean_skill : float
        The mean skill of managers with at least as many listings as the 
        threshold.
    """
    def __init__(self, threshold = 5):
        
        self.threshold = threshold
        
    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, becase they are all set together
        # in fit        
        if hasattr(self, 'mapping_'):
            
            self.mapping_ = {}
            self.mean_skill_ = 0.0
        
    def fit(self, X,y, target):
        """Compute the skill values per manager for later use.
        
        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            The rental data. It has to contain a column named "manager_id".
            
        y : pandas series or numpy array, shape [n_samples]
            The corresponding target values with encoding:
            low: 0.0
            medium: 1.0
            high: 2.0
        """        
        self._reset()
        
        temp = pd.concat([X[target] ,pd.get_dummies(y)], axis = 1).groupby(target).sum()
        temp.columns = ['low', 'medium', 'high']
        temp['count'] = X.groupby(target).count().iloc[:,0]
              
        self.mapping_ = temp #[['manager_skill']]
            
        return self
        
    def transform(self, X, target):
        """Add manager skill to a new matrix.
        
        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            Input data, has to contain "manager_id".
        """        
        X = pd.merge(left = X, right = self.mapping_, how = 'left', left_on = target, right_index = True)
        X.fillna(0, inplace = True)
        
        return X

# read train and test json file
train = pd.read_json('../train.json')
num_train = train.shape[0]
test = pd.read_json('../test.json')

# built target variable
target = {'low':0, 'medium':1, 'high':2}
y = np.array(train['interest_level'].apply(lambda x: target[x]))

# concat train and test
train.drop('interest_level', axis=1, inplace=True)
tt = pd.concat([train, test], axis=0)
#tt = tt.reset_index()

X = tt[['manager_id', 'building_id']]

# Encoding label
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

# manager
lbl.fit(list(X['manager_id'].values))
X['manager_id'] = lbl.transform(list(X['manager_id'].values))

# convert manager_id to skills
#Initialize the object
trans = Count_target()
#First, fit it to the training data:
trans.fit(X[:num_train], y, 'manager_id')
#Now transform the training data
manager = trans.transform(X, 'manager_id')

prob = pd.get_dummies(y, prefix="pred").astype(int)
p_low, p_med, p_high = prob[["pred_0", "pred_1", "pred_2"]].mean()

# add noise to the data for High-Cardinality Categorical
Num = 100
Nlow = int(p_low*Num); Nmed = int(p_med*Num) + 1; Nhigh = Num - Nlow - Nmed

manager['low'] += Nlow
manager['medium'] += Nmed
manager['high'] += Nhigh

manager['sum'] = manager['low'] + manager['medium'] + manager['high']

# High-Cardinality Categorical
k=20
f=1
manager['lambda'] = 1 / (1 + np.exp((k - manager['count'])/f))

manager['Plow_m'] = manager['lambda']*(manager['low']/manager['sum']) + (1-manager['lambda']) * p_low
manager['Pmed_m'] = manager['lambda']*(manager['medium']/manager['sum']) + (1-manager['lambda']) * p_med
manager['Phigh_m'] = manager['lambda']*(manager['high']/manager['sum']) + (1-manager['lambda']) * p_high

# building
lbl = preprocessing.LabelEncoder()

# building
lbl.fit(list(X['building_id'].values))
X['building_id'] = lbl.transform(list(X['building_id'].values))

# convert building _id to skills
#Initialize the object
trans = Count_target()
#First, fit it to the training data:
trans.fit(X[:num_train], y, 'building_id')
#Now transform the training data
building = trans.transform(X, 'building_id')

# add noise to the data for High-Cardinality Categorical
Num = 100
Nlow = int(p_low*Num); Nmed = int(p_med*Num) + 1; Nhigh = Num - Nlow - Nmed

building['low'] += Nlow
building['medium'] += Nmed
building['high'] += Nhigh

building['sum'] = building['low'] + building['medium'] + building['high']

# High-Cardinality Categorical
k=20
f=1
building['lambda'] = 1 / (1 + np.exp((k - building['count'])/f))

building['Plow_b'] = building['lambda']*(building['low']/building['sum']) + (1-building['lambda']) * p_low
building['Pmed_b'] = building['lambda']*(building['medium']/building['sum']) + (1-building['lambda']) * p_med
building['Phigh_b'] = building['lambda']*(building['high']/building['sum']) + (1-building['lambda']) * p_high

########################################
X = pd.concat([X, manager[['Plow_m', 'Pmed_m', 'Phigh_m']], building[['Plow_b', 'Pmed_b', 'Phigh_b']]], axis=1)


