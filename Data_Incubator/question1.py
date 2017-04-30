# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:11:35 2017

@author: Amin
"""

"""
Coins with values 1 through N (inclusive) are placed into a bag. All the coins 
from the bag are iteratively drawn (without replacement) at random. 
For the first coin, you are paid the value of the coin. For subsequent coins, 
you are paid the absolute difference between the drawn coin and the previously drawn coin. 
For example, if you drew 5,3,2,4,1 your payments would be 5,2,1,2,3 for a total payment of 1313.
"""

import random
import numpy as np
 
def drawn(N):
    payments = 0
    previous = 0
    coins = [x for x in range(1,N+1)]
    for ii in range(N):
        jj = random.randint(0,len(coins)-1)
        payments += abs(coins[jj] - previous)
        previous = coins[jj]    
        coins.remove(coins[jj])
        
    return payments

N = 10
result = []
average = []
error = []
for ii in range(10000):
    payments = drawn(N)
    result.append(payments)
    
    if ii%100 == 0 :
        average.append(np.mean(result))
        error.append(np.std(result))
        
# plot the result as a function of iteration to make sure they are converage
import matplotlib.pyplot as plt
plt.plot(average)
plt.ylabel('average')
plt.show()

plt.plot(error)
plt.ylabel('Error')
plt.show()

# What is the mean of your total payment for N=10? 
print average[-1]  # 38.4593475407
# What is the standard deviation of your total payment for N=10? 
print error[-1] # 6.31740403301

N = 20
result = []
average = []
error = []
for ii in range(50000):
    payments = drawn(N)
    result.append(payments)
    
    if ii%100 == 0 :
        average.append(np.mean(result))
        error.append(np.std(result))
        
# plot the result as a function of iteration to make sure they are converage
import matplotlib.pyplot as plt
plt.plot(average)
plt.ylabel('average')
plt.show()

plt.plot(error)
plt.ylabel('Error')
plt.show()

# What is the mean of your total payment for N=20? 
print average[-1]  # 143.452916775
# What is the standard deviation of your total payment for N=20? 
print error[-1]  #  18.3538159637

N = 10
Ngreater45 = 0
for ii in range(10000):
    payments = drawn(N)
    if payments >= 45 :
        Ngreater45 += 1
# What is the probability that your total payment is greater than or equal to 45 for N=10? 
print Ngreater45 / 10000.0        # 0.1821

N = 20
Ngreater160 = 0
for ii in range(50000):
    payments = drawn(N)
    if payments >= 160 :
        Ngreater160 += 1
# What is the probability that your total payment is greater than or equal to 160 for N=20?
print Ngreater160 / 50000.0         # 0.20036