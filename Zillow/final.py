
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:

import numpy as np
import pandas as pd
import seaborn as sns
import gc
import time


# In[3]:

store = pd.HDFStore('../../../Data/store_final.h5')
t1 = time.time()
train = store["train"]
prop = store["prop"]
t2 = time.time()
print 'it took ', t2-t1, ' seconds to read the dataframes'


# In[4]:

y = train.logerror
mid = np.percentile(y, 50)
y = y - mid
q1 = np.percentile(y, 25)
q3 = np.percentile(y, 75)
print q1, q3
interval = q3 - q1
fac = 5.0
interval = interval * fac / 2.
hi = interval + mid
lo = -interval + mid
print hi, lo


# In[5]:

# split the data to 9 months for train and 3 months for test
x1 = train[(train.month < 8 ) & (train.year == 2017)]    # use for train
x0 = train[(train.month > 7) & (train.year == 2017)]     # use for test

print "Size of the x1 data frame: ", x1.shape
print "Size of the x0 data frame: ", x0.shape

# drop dublicate
dup = x1["id_parcel"].duplicated(keep='first')
x1 = x1[~dup]
print "Size of the properties data frame: ", x1.shape


y1 = x1.loc[~dup, 'logerror'].values
y0 = x0['logerror'].values

'''y1 = x1['logerror'].values
y0 = x0['logerror'].values'''

index_hi = y1 > hi   # drop 1480 points
index_lo = y1 < lo    # drop 947 points
print sum(index_hi), sum(index_lo)

y1 = y1[(~index_lo) & (~index_hi)]
x1 = x1[(~index_lo) & (~index_hi)]

print "Size of the x1 data frame: ", x1.shape
print "Size of the x0 data frame: ", x0.shape


# In[7]:

import xgb
import multiprocessing

ncpu = multiprocessing.cpu_count()
print "number of cores " + str(ncpu)

model = xgb.XGBoostReg(
        eval_metric = 'mae',
        nthread = ncpu,
        eta = 0.01,
        max_depth = 9,
        subsample = 1.,
        colsample_bytree = 0.65,
        min_child_weight = 90,
        silent = 1
        )
nround = 570
from sklearn.metrics import mean_absolute_error


# In[8]:

model.fit(x1.drop(["id_parcel", "month", "year", "logerror"], axis=1), y1, num_boost_round= nround) # Train the model without outliers


# In[9]:

from sklearn.metrics import mean_absolute_error

print "Error on training data ", mean_absolute_error(y1, model.predict(x1.drop(["id_parcel", "month", "year" , "logerror"], axis=1)))
print "Error on 3 months test ", mean_absolute_error(y0, model.predict(x0.drop(["id_parcel", "month", "year", "logerror"], axis=1)))


# In[21]:

score_2months = mean_absolute_error(y0, model.predict(x0.drop(["id_parcel", "month", "year", "logerror"], axis=1)))


# # New approach

# In[12]:

# Remove train duplicates
duplicate = train["id_parcel"].duplicated(keep='first')
train = train[~duplicate]


# In[16]:

# Exclude train from prop
id_parcel = train["id_parcel"].values
prop = prop.set_index("id_parcel")
prop = prop.drop(id_parcel)
prop = prop.reset_index()
prop.shape


# In[17]:

y = train.logerror
mid = np.percentile(y, 50)
y = y - mid
q1 = np.percentile(y, 25)
q3 = np.percentile(y, 75)
print q1, q3

#fac = 8.0
interval = q3 - q1
interval = interval * fac / 2.
hi_train = interval + mid
lo_train = -interval + mid

fac = 65.0
interval = q3 - q1
interval = interval * fac / 2.
hi_test = interval + mid
lo_test = -interval + mid

print lo_train, hi_train
print lo_test, hi_test


# In[18]:

y = train['logerror'].values
x = train.drop(['month', 'year','logerror'], axis=1)
print "Size of the train data frame: ", x.shape
print "Size of the prop  data frame: ", prop.shape

print("Generate a list of outliers should be droped for training")
index_hi = y > hi_train
index_lo = y < lo_train
print sum(index_hi), sum(index_lo)

outliers_train = []
for ii in range(y.shape[0]):
    if index_hi[ii] or index_lo[ii]:
        outliers_train.append(ii)

print("Generate a list of outliers should be droped for testing")
index_hi = y > hi_test
index_lo = y < lo_test
print sum(index_hi), sum(index_lo)

outliers_test = []
for ii in range(y.shape[0]):
    if index_hi[ii] or index_lo[ii]:
        outliers_test.append(ii)


# In[19]:

def splitDataFrameIntoSmaller(df, chunkSize = 100000):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(i*chunkSize)
    listOfDf.append(len(df))
    return listOfDf

split_index = splitDataFrameIntoSmaller(prop)


# In[20]:

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

skf = KFold(n_splits = 10, shuffle = True, random_state = 44)

train_pred = np.zeros(train.shape[0], dtype=np.float16)
prop_pred = np.zeros(prop.shape[0], dtype=np.float16)
scores1 = []; scores2 = []

for train_index, test_index in skf.split(x, y):

    train_index_wo = [ix for ix in train_index if ix not in outliers_train]
    test_index_wo = [ix for ix in test_index if ix not in outliers_test]

    x1, x0 = x.iloc[train_index_wo], x.iloc[test_index_wo]
    y1, y0 = y[train_index_wo], y[test_index_wo]

    model.fit(x1.drop(["id_parcel"], axis=1), y1, num_boost_round= nround) # Train the model without outliers

    #calculate score without second outliers
    scores1.append(mean_absolute_error(y0, model.predict(x0.drop(["id_parcel"], axis=1))))
    print "Score without outliers for the ", len(scores1), " fold is ", scores1[len(scores1)-1]

    #calculate score with outliers
    x0 = x.iloc[test_index]
    y0 = y[test_index]

    pred = model.predict(x0.drop(["id_parcel"], axis=1))
    scores2.append(mean_absolute_error(y0, pred))
#    print "Score with outliers for the ", len(scores2), " fold is ", scores2[len(scores2)-1]

    for ii, idx in enumerate(test_index):
        train_pred[idx] = pred[ii]

    for ii in range(0, len(split_index)-1):
        n1 = split_index[ii]; n2 = split_index[ii+1]
        pred = model.predict(prop.iloc[n1:n2].drop(['id_parcel'], axis=1))
        prop_pred[n1:n2] += pred

print "Average score without outliers over all folds : " , np.mean(scores1), " ", np.std(scores1)
print "Average score with    outliers over all folds : " , np.mean(scores2), " ", np.std(scores2)


# In[ ]:

out = pd.DataFrame()
out["ParcelId"] = prop["id_parcel"]
months = ["201610"] #, "201611", "201612", "201710", "201711", "201712"]
for col in months:
    out[col] = map(lambda x: x/10.0, prop_pred)

out_train = pd.DataFrame()
out_train["ParcelId"] = train["id_parcel"]
for col in months:
    out_train[col] = train_pred #+ 0.02 #IMPORTANT POINT: I add a constant to train prediction


print("Read the missing")
miss = store["miss"]

med = train.logerror.median()
for col in months:
    miss[col] = med

miss = miss[["id_parcel"]+months]
miss.columns = ["ParcelId"] + months

out = pd.concat([out, out_train, miss], axis=0)

from datetime import datetime
out.to_csv('test_2017.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')


# In[27]:

print str(score_2months)+","+str(np.mean(scores1))

