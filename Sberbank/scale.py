#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:33:37 2017

@author: amin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#test = pd.read_csv("naivexgb.csv")
#test["price_doc"] = test["price_doc"] * 0.97
#train = pd.read_csv("Data/train.csv")

test = pd.read_csv("new4.csv")

#test["price_doc"] = test["price_doc"] * 1.01   
#test.to_csv('new4_scale2.csv', index=False)

testlog = np.log(test["price_doc"])

#med = testlog.mean()
#diff = med - testlog

#scale = med - diff * 1.0

#index = diff > 0 # small value
#scale[index] = med - diff[index] * 0.99
     
#index = diff < 0 # large value
#scale[index] = med - diff[index] * 1.03
     
## Q3
scale = testlog.copy(deep=True)

q3 = np.percentile(testlog, 75)
q2 = np.percentile(testlog, 50)
q1 = np.percentile(testlog, 25)

diff = q3 - testlog
index = diff < 0 # large value
scale[index] = q3 - diff[index] * 1.05
     
#diff = q2 - testlog
#index = (testlog > q2) & (testlog < q3)
#scale[index] = q2 - diff[index] * 0.99

#diff = q1 - testlog
#index = diff > 0 # small value
#scale[index] = q1 - diff[index] * 1.01
     


output = pd.DataFrame({'id': test.id, 'price_doc': np.exp(scale)})
output.to_csv('new4_scale10.csv', index=False)

testlog.mean()
testlog.median()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

test = pd.read_csv("new4.csv")
testlog = np.log(test["price_doc"])

## Q3
scale = testlog.copy(deep=True)

q3 = np.percentile(testlog, 75)
q2 = np.percentile(testlog, 50)
q1 = np.percentile(testlog, 25)

index = (testlog > q1) & (testlog < q2)
diff = q1 - testlog
scale[index] = q1 - diff[index] * 0.97
     
output = pd.DataFrame({'id': test.id, 'price_doc': np.exp(scale)})
output.to_csv('new4_scale15.csv', index=False)

