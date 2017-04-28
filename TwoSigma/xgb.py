#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:53:05 2017

@author: amin
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from collections import defaultdict

# cross validation def to tune parameter
def XGBCV(train_X, train_y, eta=0.05, max_depth=6, n_estimators=100, nfold=5, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['n_estimators'] = n_estimators
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1.
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['num_threads'] = 6
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    model = xgb.cv(plst, xgtrain, num_rounds, nfold=nfold, verbose_eval=50, early_stopping_rounds=20)

    num_run = model.shape[0]

    return model.iloc[num_run-1,0], model.iloc[num_run-1,2], num_run

# train xgboost and prediction
def XGB(train_X, train_y, test_X, eta=0.05, max_depth=6, n_estimators=100, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['n_estimators'] = n_estimators
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1.
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['num_threads'] = 2
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [(xgtrain,'train')]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, verbose_eval=50, early_stopping_rounds=20)

    xgtest = xgb.DMatrix(test_X)
    pred = model.predict(xgtest)

    return pred, model

#############################################################################################
# read train and test json file
train = pd.read_json('../data/train.json')
num_train = train.shape[0]
test = pd.read_json('../data/test.json')

# CV statistic fo High-Cardinality Categorical
import random

index=list(range(train.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train)
b=[np.nan]*len(train)
c=[np.nan]*len(train)

for i in range(5):
    building_level={}
    for j in train['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train.shape[0])/5):int(((i+1)*train.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train['manager_level_low']=a
train['manager_level_medium']=b
train['manager_level_high']=c

a=[]
b=[]
c=[]
building_level={}
for j in train['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train.shape[0]):
    temp=train.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test['manager_level_low']=a
test['manager_level_medium']=b
test['manager_level_high']=c

# built target variable
target = {'high':0, 'medium':1, 'low':2}
y = np.array(train['interest_level'].apply(lambda x: target[x]))

# concat train and test
train.drop('interest_level', axis=1, inplace=True)
tt = pd.concat([train, test], axis=0)

#
print("*****")
dis = pd.read_csv("../data/XX.csv")
tt = pd.merge(tt, dis, on='listing_id', how='left')
tt = tt.fillna(0)

#tt = pd.merge(tt, leak, on='listing_id', how='left')
import re

def cap_share(x):
    return sum(1 for c in x if c.isupper())/float(len(x)+1)

for df in [tt]:
    # do you think that users might feel annoyed BY A DESCRIPTION THAT IS SHOUTING AT THEM?
    df['num_cap_share'] = df['description'].apply(cap_share)
    
    # how long in lines the desc is?
    df['num_nr_of_lines'] = df['description'].apply(lambda x: x.count('<br /><br />'))
   
    # is the description redacted by the website?        
    df['num_redacted'] = 0
    df['num_redacted'].ix[df['description'].str.contains('website_redacted')] = 1

    
    # can we contact someone via e-mail to ask for the details?
    df['num_email'] = 0
    df['num_email'].ix[df['description'].str.contains('@')] = 1
    
    #and... can we call them?
    
    reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
    def try_and_find_nr(description):
        if reg.match(description) is None:
            return 0
        return 1

    df['num_phone_nr'] = df['description'].apply(try_and_find_nr)
    
import math
def cart2rho(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho


def cart2phi(x, y):
    phi = np.arctan2(y, x)
    return phi


def rotation_x(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return x*math.cos(alpha) + y*math.sin(alpha)


def rotation_y(row, alpha):
    x = row['latitude']
    y = row['longitude']
    return y*math.cos(alpha) - x*math.sin(alpha)


def add_rotation(degrees, df):
    namex = "rot" + str(degrees) + "_X"
    namey = "rot" + str(degrees) + "_Y"

    df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
    df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)

    return df

def operate_on_coordinates(df):
    for df in [df]:
        #polar coordinates system
        df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        #rotations
        for angle in [15,30,45,60]:
            df = add_rotation(angle, df)

    return df

tt = operate_on_coordinates(tt)
##############################################################################
# built features
features = ['manager_level_low', 'manager_level_medium', 'manager_level_high', "bathrooms", "bedrooms", "latitude", "longitude", "price",
             "listing_id", 'num_cap_share', 'num_nr_of_lines', 'num_redacted', 'num_email', 'num_phone_nr', 'num_rho', 'num_phi',
           'num_rot15_X', 'num_rot15_Y', 'num_rot30_X', 'num_rot30_Y', 'num_rot45_X', 'num_rot45_Y', 'num_rot60_X', 'num_rot60_Y']
X = tt[features]

X['sum4'] = tt[['dis0', 'dis1', 'dis2', 'dis3', 'dis4']].sum(axis=1)
X['sum3'] = tt[['dis0', 'dis1', 'dis2', 'dis3']].sum(axis=1)
X['sum2'] = tt[['dis0', 'dis1', 'dis2']].sum(axis=1)
X['sum1'] = tt[['dis0', 'dis1']].sum(axis=1)



image_date = pd.read_csv("../data/leak.csv")
# rename columns so you can join tables later on
image_date.columns = ["listing_id", "time_stamp"]

# reassign the only one timestamp from April, all others from Oct/Nov
image_date.loc[80240,"time_stamp"] = 1478129766

image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
image_date["img_date_month"]            = image_date["img_date"].dt.month
image_date["img_date_week"]             = image_date["img_date"].dt.week
image_date["img_date_day"]              = image_date["img_date"].dt.day
image_date["img_date_dayofweek"]        = image_date["img_date"].dt.dayofweek
image_date["img_date_dayofyear"]        = image_date["img_date"].dt.dayofyear
image_date["img_date_hour"]             = image_date["img_date"].dt.hour
image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

image_date.drop('img_date', axis=1, inplace=True)
X = pd.merge(X, image_date, on="listing_id", how="left")

X['listing_id'] -= X['listing_id'].min()

# test price
X.loc[:,'price_bed'] = X['price'] / (X['bedrooms']+1)
X.loc[:,'price_bed'] = X.loc[:,'price_bed'].apply(lambda x: np.log(np.sqrt(x)))

#X.loc[:,'price'] = X.loc[:,'price'].apply(lambda x: np.log(np.sqrt(x)))

# classify bathrooms
X.loc[X['bathrooms'] > 5.0, 'bathrooms'] = 5.5

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(X['bathrooms'])
yy = label_encoder.transform(X['bathrooms'])

onehot_encoder = OneHotEncoder(sparse=False)
dummy = pd.DataFrame(onehot_encoder.fit_transform(yy.reshape(X.shape[0],1)).astype(int),
                     columns=['bath'+str(i) for i in range(0,max(yy)+1)])

X = pd.concat([X.reset_index(), dummy], axis=1)
X.set_index('index', inplace=True)

X['bathrooms'] = tt['bathrooms']

# classify bedrooms
X.loc[X['bedrooms'] > 4.0, 'bedrooms'] = 4.0

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(X['bedrooms'])
yy = label_encoder.transform(X['bedrooms'])

onehot_encoder = OneHotEncoder(sparse=False)
dummy = pd.DataFrame(onehot_encoder.fit_transform(yy.reshape(X.shape[0],1)).astype(int),
                     columns=['bed'+str(i) for i in range(0,max(yy)+1)])

X = pd.concat([X.reset_index(), dummy], axis=1)
X.set_index('index', inplace=True)

X['bedrooms'] = tt['bedrooms']

# Number of photo
X['num_photo'] = tt['photos'].apply(len)

# number of features
X['num_features'] = tt['features'].apply(len)

# number of words
fmt = lambda s: s.replace("\u00a0", "").strip().lower()
tt['description'] = tt['description'].apply(fmt)

X['words'] = tt['description'].apply(lambda x: len(x.split(" ")))

from nltk.stem import PorterStemmer
import re
# Removes symbols, numbers and stem the words to reduce dimentional space
stemmer = PorterStemmer()
def clean(x):
    regex = re.compile('[^a-zA-Z ]')
    # For user clarity, broken it into three steps
    i = regex.sub(' ', x).lower()
    i = i.split(" ")
    i= [stemmer.stem(l) for l in i]
    i= " ".join([l.strip() for l in i if (len(l)>2) ]) # Keeping words that have length greater than 2
    return i

tt['description'] = tt['description'].apply(lambda x : clean(x))

X['words2'] = tt['description'].apply(lambda x: len(x.split(" ")))

# date
X['created'] = pd.to_datetime(tt['created'])

X['month'] = X['created'].dt.month
X['day'] = X['created'].dt.day
X['hour'] = X['created'].dt.hour
X['week'] = X['created'].dt.week

X['time_dif'] = X['created'].max() - X['created']

X['day_dif'] = X['time_dif'].dt.days

X.drop('created', axis=1, inplace=True)
X.drop('time_dif', axis=1, inplace=True)

# manager ID
lbl = LabelEncoder()
lbl.fit(tt['manager_id'])
X['manager'] = lbl.transform(tt['manager_id'])

freq = defaultdict(int)
for ii in X['manager']:
    freq[ii] += 1

temp = np.zeros(X.shape[0], int)
jj = 0
for ii in X.index:
    temp[jj] = freq[X.loc[ii, 'manager']]
    jj += 1
temp_pd = pd.DataFrame(temp, columns=['manager_freq'])
X = pd.concat([X.reset_index(), temp_pd], axis=1)
X.set_index('index', inplace=True)

# building
lbl = LabelEncoder()
lbl.fit(tt['building_id'])
X['building'] = lbl.transform(tt['building_id'])

freq = defaultdict(int)
for ii in X['building']:
    freq[ii] += 1

temp = np.zeros(X.shape[0], int)
jj = 0
for ii in X.index:
    temp[jj] = freq[X.loc[ii, 'building']]
    jj += 1
temp_pd = pd.DataFrame(temp, columns=['building_freq'])
X = pd.concat([X.reset_index(), temp_pd], axis=1)
X.set_index('index', inplace=True)

#XGBCV(X[:num_train], y, nfold=5)  #0.5551

# address display
lbl = LabelEncoder()
lbl.fit(tt['display_address'])
X['address'] = lbl.transform(tt['display_address'])

#XGBCV(X[:num_train], y, nfold=5) 0.5547

freq = defaultdict(int)
for ii in X['address']:
    freq[ii] += 1

temp = np.zeros(X.shape[0], int)
jj = 0
for ii in X.index:
    temp[jj] = freq[X.loc[ii, 'address']]
    jj += 1
temp_pd = pd.DataFrame(temp, columns=['address_freq'])
X = pd.concat([X.reset_index(), temp_pd], axis=1)
X.set_index('index', inplace=True)

#XGBCV(X[:num_train], y, nfold=5) 0.5537

# street
add = tt['display_address'].apply(lambda x : ''.join( c for c in x if  c not in '.,?:!/;&' ))

stoplist = set('street st st. ave. avenue ave w west e east n north s south and center place road'.split()) # use to drop some words from list
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in add]

number = {'first':1, 'second':2, 'third':3, 'forth':4, 'fifth':5, 'sixth':6, 'seventh':7, 'eighth':8, 'ninth':9, 'tenth':10}
texts = [[str(number[x]) if x in number else x for x in text] for text in texts]

texts2 = []
for text in texts:
    if len(text) > 1:
        try :
            ii = int(text[0])
            texts2.append(text.pop(0))
        except ValueError:
            texts2.append(text)
    else :
        texts2.append(text)

texts3 = []
for text in texts2:
    texts3.append([x.replace('th', '').replace('nd', '').
                   replace('rd', '').replace('st', '').replace("'s", '') for x in text])

lbl = LabelEncoder()
lbl.fit(texts3)
X['street'] = lbl.transform(texts3)

#XGBCV(X[:num_train], y, nfold=5) 0.5538

freq = defaultdict(int)
for ii in X['street']:
    freq[ii] += 1

temp = np.zeros(X.shape[0], int)
jj = 0
for ii in X.index:
    temp[jj] = freq[X.loc[ii, 'street']]
    jj += 1
temp_pd = pd.DataFrame(temp, columns=['street_freq'])
X = pd.concat([X.reset_index(), temp_pd], axis=1)
X.set_index('index', inplace=True)

#XGBCV(X[:num_train], y, nfold=5) 0.5533

# address2
#lbl = LabelEncoder()
#lbl.fit(tt['street_address'])
#X['address2'] = lbl.transform(tt['street_address'])

#XGBCV(X[:num_train], y, nfold=5) 0.5538

# some ratio features
X['bed_bath'] = X['bedrooms'] / (X['bathrooms']+1)
X['lon_lat'] = X['longitude'] / (X['latitude']+1)
X['lonlat'] = X['longitude'] * X['latitude']
X['dis_lon'] = X['longitude'] - X['longitude'].mean()
X['dis_lat'] = X['latitude'] - X['latitude'].mean()
X['dis'] = X['dis_lon'] * X['dis_lon'] + X['dis_lat'] * X['dis_lat']
X['dis_multy'] = X['dis_lon'] * X['dis_lat']

X['dis2_lon'] = X['longitude'] - X['longitude'].median()
X['dis2_lat'] = X['latitude'] - X['latitude'].median()
X['dis2'] = X['dis_lon'] * X['dis_lon'] + X['dis_lat'] * X['dis_lat']
X['dis2_multy'] = X['dis_lon'] * X['dis_lat']
print "########"

X["num_furniture"] = X["bathrooms"] + X["bedrooms"]
X["price_latitue"] = (X["price"])/ (X["latitude"]+1.0)
X["price_longtitude"] = (X["price"])/ (X["longitude"]-1.0)
X["total_days"] =   (X["month"] - 4.0)*30 + X["day"] +  X["hour"] /25.0

X['price_sq'] = (X.price / (1 + X.bedrooms + 0.5*X.bathrooms)).values
X["pricePerBed"] = X['price'] / (1+X['bedrooms'])
X["pricePerBath"] = X['price'] / (1+X['bathrooms'])
X["pricePerRoom"] = X['price'] / (X['bedrooms'] + X['bathrooms']+1)
X["bedPerBath"] = (1 + X['bedrooms']) / (1+X['bathrooms'])
X["bedBathDiff"] = X['bedrooms'] - X['bathrooms']
X["bedBathSum"] = X["bedrooms"] + X['bathrooms']
X["bedsPerc"] = X["bedrooms"] / (1+X['bedrooms'] + X['bathrooms'])


#########################################################
# vectorize features
from gensim import corpora, models, similarities

ff = train['features']

features = ff.apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x])).tolist()

stoplist = set(''.split()) # use to drop some words from list
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in features]

# calculate the frequency of each word
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# remove words that appear only once
texts = [[token for token in text if frequency[token] > 5] for text in texts]

# built a dictionary
dictionary = corpora.Dictionary(texts)
print "size of the dic "+str(len(dictionary))
print "number of documents "+str(dictionary.num_docs)
print "number of words "+str(dictionary.num_pos)

# Generate corpus and tfidf
corpus = [dictionary.doc2bow(text) for text in texts] # generate corpus
tfidf = models.TfidfModel(corpus)

# generate the vector of features in all data
ff = tt['features']

features = ff.apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x])).tolist()

stoplist = set(''.split()) # use to drop some words from list
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in features]

vectors = np.zeros((tt.shape[0], len(dictionary)), dtype=np.int)
transfer = np.zeros((tt.shape[0], len(dictionary)), dtype=np.float)
for row, text in enumerate(texts):
    new_vectors = dictionary.doc2bow(text)
    trans_vec = tfidf[new_vectors]
    for new_vec in new_vectors:
        index, value = new_vec
        vectors[row][index] = value
    for new_vec in trans_vec:
        index, value = new_vec
        transfer[row][index] = value

for i in range(vectors.shape[0]):
    for j in range(vectors.shape[1]):
        if vectors[i, j] > 1:
            vectors[i, j] = 1

# trasfer

from scipy import sparse
vec_mat = sparse.csr_matrix(vectors)
XX = sparse.hstack([X, vec_mat]).tocsr()

XX = XX.toarray()
print ("scalling")
#scale the data
from sklearn.preprocessing import StandardScaler
stda=StandardScaler()
stda.fit(XX[:num_train])
XX=stda.transform(XX)

result = []
for eta in [0.02]:
    for depth in [6]:

        print eta, depth
        name = "xgb_" + str(eta) + "_" + str(depth) + ".csv"
        score1, score2, Nrun = XGBCV(XX[:num_train], y, eta=eta, max_depth=depth, nfold=5)

        result.append([score1, score2, eta, depth, Nrun])

        preds, model = XGB(XX[:num_train], y, XX[num_train:], eta=eta, max_depth=depth, num_rounds=Nrun)

        preds_df = pd.DataFrame(preds)
        preds_df.columns = ["high", "medium", "low"]
        preds_df["listing_id"] = test.listing_id.values
        preds_df.to_csv(name, index=False)

print result

np.savetxt("input", XX, delimiter=",", fmt='%.7f')
np.savetxt("target", y, delimiter=",", fmt='%.f')
ids = test['listing_id'].values
np.savetxt("ids", ids, delimiter=",", fmt='%.f')
########################################################3
"""result
features                test                  board           out
"""
