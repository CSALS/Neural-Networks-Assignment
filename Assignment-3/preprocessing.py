#!/usr/bin/env python
# coding: utf-8

# In[17]:


import scipy.io
import numpy as np


# In[18]:


def scaler():
    data = scipy.io.loadmat('data_for_cnn.mat')
    X = data['ecg_in_window']
    data = scipy.io.loadmat('class_label.mat')
    Y = data['label']
    valueArray = np.concatenate((X, Y), axis = 1)
    np.random.shuffle(valueArray)
    X = valueArray[0:, :1001]
    Y = valueArray[0:, -1]
    Y = np.reshape(Y,(Y.shape[0],1))
    Y = Y.astype(int)
    #Feature Scaling
    X = (X - np.mean(X, axis = 0))/np.std(X, axis = 1)
    #Output in correct format
    Y = np.eye(len(np.unique(Y,axis=0)))[(Y.T).flatten()]
    return X, Y

