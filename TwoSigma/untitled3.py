#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:42:54 2017

@author: amin
"""


import numpy as np
import pandas as pd

# read train and test json file
train = pd.read_json('../train.json')
num_train = train.shape[0]
test = pd.read_json('../test.json')

from math import sin, cos, sqrt, atan2, radians

def distance(lat1, lon1, lat2, lon2):
    #return distance as meter if you want km distance, remove "* 1000"
    radius = 6373 # * 1000

    lat1 = radians(lat1)
    lat2 = radians(lat2)
    dlon = radians(lon2) - radians(lon1)
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius * c

# built target variable
target = {'low':0, 'medium':1, 'high':2}
y = np.array(train['interest_level'].apply(lambda x: target[x]))

# concat train and test
#train.drop('interest_level', axis=1, inplace=True)
tt = pd.concat([train, test], axis=0)
tt = tt.reset_index()

X = tt.loc[:, ['latitude', 'longitude', 'listing_id', 'interest_level']]

# Outlier removal
for i in ['latitude', 'longitude']:
    while(1):
        med = X[i].median()
        ix = abs(X[i] - med) > 3*X[i].std()
        if ix.sum()==0: # no more outliers -> stop
            break
        X.loc[ix, i] = np.nan # exclude outliers


res = 40 # grid size
# Define grids
nx = np.linspace(X.longitude.min(), X.longitude.max(), res)
ny = np.linspace(X.latitude.min(), X.latitude.max(), res)

#  distance(ny[0], nx[0], ny[1], nx[0])
dx = nx[1] - nx[0]; dy = ny[1] - ny[0]

ll = X.shape[0]
XX = pd.DataFrame(index=range(ll))
XX_l = pd.DataFrame(index=range(ll))
XX_m = pd.DataFrame(index=range(ll))
XX_h = pd.DataFrame(index=range(ll))
XX['listing_id'] = 0
XX_l['listing_id'] = 0
XX_m['listing_id'] = 0
XX_h['listing_id'] = 0
for ii in range(6):
    name = "dis" + str(ii)
    XX[name] = 0
    XX_l[name] = 0
    XX_m[name] = 0
    XX_h[name] = 0


indx = 0
for i in range(res-1):
    for j in range(res-1):
        # Identify listings within the square
        ix = (X.longitude >= nx[i])&(X.longitude < nx[i+1])&(X.latitude >= ny[j])&(X.latitude < ny[j+1])
        iy = (X.longitude >= nx[i] - 2*dx)&(X.longitude < nx[i+1] + 2*dx)&(X.latitude >= ny[j] - dy)&(X.latitude < ny[j+1] + dy)

        center = X.loc[ix, :]
        around = X.loc[iy, :]

        for ii in range(center.shape[0]):
            print indx
            point = list(center.iloc[ii])
            dd = around.apply(lambda x: distance(point[0], point[1], x[0], x[1]), axis=1)
            name = dd.apply(lambda x : "dis" + str(int(x*10)) if x<0.5 else "dis5")
            interest = pd.concat([name ,pd.get_dummies(around['interest_level'])], axis = 1).groupby(0).sum()
            for ss in list(interest.index):
                if "low" in interest.columns:
                    XX_l.loc[indx, ss] +=  interest.loc[ss, "low"]
                if "medium" in interest.columns:
                    XX_m.loc[indx, ss] +=  interest.loc[ss, "medium"]
                if "high" in interest.columns:
                    XX_h.loc[indx, ss] +=  interest.loc[ss, "high"]
            name = name.value_counts()
            XX.loc[indx, 'listing_id'] = int(point[2])
            for ss in list(name.index):
                XX.loc[indx, ss] = name[ss]
            indx += 1


XX.to_csv('XX.csv')
XX_l.to_csv('XX_l.csv')
XX_m.to_csv('XX_m.csv')
XX_h.to_csv('XX_h.csv')





