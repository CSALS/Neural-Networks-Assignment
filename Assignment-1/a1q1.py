# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 07:01:10 2019

@author: Charan

References ->
https://stackoverflow.com/questions/16888888/how-to-read-a-xlsx-file-using-the-pandas-library-in-ipython
https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array/54508052#54508052
https://stackoverflow.com/questions/36998260/prepend-element-to-numpy-array
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataSet = 'D:\\study\\NNFL\\assigment_and_slides\\assignment1\\data.xlsx'
print("hello")
#header = None states that there is no header row or else it would take first row of our data as header.
df = pd.read_excel(dataSet,sheet_name='Sheet1',header=None)

valueArray = df.to_numpy() #dataframe DF to numpyarray valueArray
m = valueArray.shape[0] # Number of training examples
X = valueArray[0:,0:2] # Feature Vector [x1,x2] m*n
X = np.insert(X,0,1,axis=1) # axis = 1 (every row insert 1 at 0 column) , Adding x0 = 1 in feature matrix m*(n+1)
y = valueArray[0:,2:] # Class label Vector [y]
w = np.random.rand(3,1) #Weight matrix with random values (n+1)*1
X = X/np.max(X)
y = y/np.max(y)
alpha = 0.0000001
iters = 5
cost = []
it = []
def computeCostFunction(X,y,w):
    return (0.5*np.sum((X.dot(w) - y)**2))
  
for i in range(iters):
    h = X.dot(w)
    for j in range(3):
        subSum = 0.0
        for x in range(m):
            subSum = subSum + ((h[x][0] - y[x][0]) * X[x][j])
            w[j][0] = w[j][0] - (alpha) * subSum
    cost = np.append(cost,computeCostFunction(X,y,w))
    it = np.append(it,i+1)

cost = cost/10000
plt.plot(it,cost,'b')


#for i in range(iters):
        #w = w - (alpha) * np.sum((X.dot(w) - y) * X, axis=0)
        #cost = computeCostFunction(X, y, w)


