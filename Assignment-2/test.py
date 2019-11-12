#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from preprocessing import getData
from pandas import *


# In[2]:


def sigmoid(Z):
    return 1/(1 + np.exp(-Z))
def sigmoidDerivative(A):
    return A*(1-A)
def PreTrain(Input, Output, HiddenNeurons):
    #Parameters
    Features = Input.shape[0]
    Classes = Output.shape[0]
    num_iterations = 3000
    alpha = 0.086
    np.random.seed(15)
    #Random Initialization of weights
    W1 = np.random.rand(HiddenNeurons, Features)
    b1 = np.random.rand(HiddenNeurons, 1)
    W2 = np.random.rand(Classes, HiddenNeurons)
    b2 = np.random.rand(Classes, 1)
    #Start Training
    for iteration in range(num_iterations):
        #Forward Propagation
        Z1 = W1.dot(Input) + b1
        A1 = sigmoid(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = Y_tilda = sigmoid(Z2)
        #Backpropagation
        Delta_2 = (Y_tilda - Output) * sigmoidDerivative(A2)
        Delta_1 = W2.T.dot(Delta_2) * sigmoidDerivative(A1)
        #Weight and bias updation
        W2 = W2 - alpha * np.dot(Delta_2, A1.T)
        W1 = W1 - alpha * np.dot(Delta_1, Input.T)
        b2 = b2 - alpha * np.sum(Delta_2, axis = 1, keepdims = True)
        b1 = b1 - alpha * np.sum(Delta_1, axis = 1, keepdims = True)
    #Completed Training
    return [W1, b1, sigmoid(W1.dot(Input) + b1)]


# In[3]:


X, Y = getData('data.mat')
#Holdout method
train_percent = 0.65
train_size = int(train_percent*X.shape[0])
train_X = X[:train_size,:]
test_X = X[train_size:,:]
train_Y = Y[:train_size,:]
test_Y = Y[train_size:,:]
train_X = train_X.T
train_Y = train_Y.T
test_X = test_X.T
test_Y = test_Y.T


# In[4]:


#deep layer stacked autoencoder based extreme learning machine.
HiddenLayer = [60,75]


# In[6]:


#Pre-training the two autoencoders
[W1, b1, Output1] = PreTrain(train_X, train_X, HiddenLayer[0])
[W2, b2, Output2] = PreTrain(Output1, Output1, HiddenLayer[1])


# In[13]:


#Stack these two autoencoders and send the output as input to ELM Classifier
"""
      W1    W2      W3(Randomly initialized)
Input --- H1 --- H2 --- ELM Classifier

"""
#Finding Output of our stacked autoencoder
Z1 = W1.dot(train_X)
A1 = sigmoid(Z1)
Z2 = W2.dot(A1)
A2 = sigmoid(Z2)
ELM_Input = A2.T
#Random Initialization of ELM Classifier parameters
InputNeurons = ELM_Input.shape[1]
HiddenNeurons = 425
RandomA = np.random.randn(InputNeurons, HiddenNeurons)
RandomB = np.random.randn(1, HiddenNeurons)
#Evaluating Hidden Layer Matrix
InputSize = ELM_Input.shape[0]
HiddenLayer = np.zeros((InputSize, HiddenNeurons))
# H = exp(-b||x - a||)
for i in range(InputSize):
    for j in range(HiddenNeurons):
        HiddenLayer[i][j] = np.exp(-RandomB[0][j] * np.linalg.norm(ELM_Input[i] - RandomA.T[j], 1))
        if HiddenLayer[i][j] == 0:
            HiddenLayer[i][j] = 1e-10
#Evaluating Weight Matrix
WeightMatrix = np.linalg.inv(HiddenLayer.T.dot(HiddenLayer)).dot(HiddenLayer.T).dot(train_Y.T)


# In[15]:


#Testing
# 1. Do forward propagation on the two autoencoders and send output as input to ELM Classifier
Z1 = W1.dot(test_X) + b1
A1 = sigmoid(Z1)
Z2 = W2.dot(A1) + b2
A2 = sigmoid(Z2)
ELM_TestInput = A2.T
ELM_TestOutput = test_Y.T
# 2. Testing on ELM Classifier
test_size = ELM_TestInput.shape[0]
TestHiddenLayer = np.zeros((test_size, HiddenNeurons))
# H = exp(-b||x - a||)
for i in range(test_size):
    for j in range(HiddenNeurons):
        TestHiddenLayer[i][j] = np.exp(-RandomB[0][j] * np.linalg.norm(ELM_TestInput[i] - RandomA.T[j], 1))
        if TestHiddenLayer[i][j] == 0:
            TestHiddenLayer[i][j] = 1e-10
PredictedOutput = TestHiddenLayer.dot(WeightMatrix)
#Compare with actual output
count = 0
TrueZeros = TrueOnes = FalseZeros = FalseOnes = 0
for i in range(test_size):
    actualClass = np.argmax(ELM_TestOutput[i])
    predictedClass = np.argmax(PredictedOutput[i])
    if actualClass == predictedClass:
        count += 1
        if actualClass == 0:
            TrueZeros += 1
        else:
            TrueOnes += 1
    else:
        if actualClass == 0:
            FalseZeros += 1
        else:
            FalseOnes += 1
conf_mat = ([[TrueZeros, FalseZeros], [FalseOnes, TrueOnes]])
acc = count/test_size*100


# In[16]:


print(acc)


# In[17]:


conf_mat

