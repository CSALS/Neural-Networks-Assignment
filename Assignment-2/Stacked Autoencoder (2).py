#!/usr/bin/env python
# coding: utf-8



import numpy as np
# from preprocessing import getData
import scipy.io
def getData(name):
    mat = scipy.io.loadmat(name)
    valueArray = mat['x']
    np.random.shuffle(valueArray)
    X = valueArray[0:,:72]
    Y = valueArray[0:,-1]
    Y = np.reshape(Y,(Y.shape[0],1))
    Y = Y.astype(int)
    #Feature Scaling
    X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
    #np.eye() - ones on diagonal , zeros on rest
    Y = np.eye(len(np.unique(Y,axis=0)))[(Y.T).flatten()]
    return X,Y





def sigmoid(Z):
    return 1/(1 + np.exp(-Z))
def sigmoidDerivative(A):
    return A*(1-A)
def PreTrain(Input, Output, HiddenNeurons):
    #Parameters
    Features = Input.shape[0]
    Classes = Output.shape[0]
    num_iterations = 2000
    alpha = 0.1
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


X, Y = getData('data5.mat')
#Holdout method
train_percent = 0.7
train_size = int(train_percent*X.shape[0])
train_X = X[:train_size,:]
test_X = X[train_size:,:]
train_Y = Y[:train_size,:]
test_Y = Y[train_size:,:]
train_X = train_X.T
train_Y = train_Y.T
test_X = test_X.T
test_Y = test_Y.T


train_X.shape



#Stacked Autoencoder based Deep Neural Network
HiddenLayer = [42,24,12]



#Pre-training the three autoencoders
[W1, b1, Output1] = PreTrain(train_X, train_X, HiddenLayer[0])
[W2, b2, Output2] = PreTrain(Output1, Output1, HiddenLayer[1])
[W3, b3, Output3] = PreTrain(Output2, Output2, HiddenLayer[2])


#Fine Tuning by stacking all those three autoencoders
"""
      W1     W2      W3      W4
Input --- H1 ---- H2 ---- H3 --- Output
      b1     b2      b3      b4
Use W1, W2, W3 from pre-trained autoencoders
Randomly initalize W4
"""
#Parameters
Input = train_X
Output = train_Y
Classes = Output.shape[0]
num_iterations = 3000
alpha = 0.1
# np.random.seed(17)
#Random initialize W4, b4
W4 = np.random.rand(Classes, HiddenLayer[2])
b4 = np.random.rand(Classes, 1)
#Start Training
for iteration in range(num_iterations):
    #Forward Propagation
    Z1 = W1.dot(Input) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = sigmoid(Z3)
    Z4 = W4.dot(A3) + b4
    A4 = Y_tilda = sigmoid(Z4)
    #Backpropagation
    delta_4 = (Y_tilda - Output) * sigmoidDerivative(A4)
    delta_3 = W4.T.dot(delta_4) * sigmoidDerivative(A3)
    delta_2 = W3.T.dot(delta_3) * sigmoidDerivative(A2)
    delta_1 = W2.T.dot(delta_2) * sigmoidDerivative(A1)
    #Weight and bias updation
    W4 = W4 - alpha * np.dot(delta_4, A3.T)
    b4 = b4 - alpha * np.sum(delta_4, axis = 1, keepdims = True)
    W3 = W3 - alpha * np.dot(delta_3, A2.T)
    b3 = b3 - alpha * np.sum(delta_3, axis = 1, keepdims = True)
    W2 = W2 - alpha * np.dot(delta_2, A1.T)
    b2 = b2 - alpha * np.sum(delta_2, axis = 1 ,keepdims = True)
    W1 = W1 - alpha * np.dot(delta_1, train_X.T)
    b1 = b1 - alpha * np.sum(delta_1, axis = 1, keepdims = True)
#After end of loop we have optimal parameters


# Testing

#Forward Propagation
Z1 = W1.dot(test_X) + b1
A1 = sigmoid(Z1)
Z2 = W2.dot(A1) + b2
A2 = sigmoid(Z2)
Z3 = W3.dot(A2) + b3
A3 = sigmoid(Z3)
Z4 = W4.dot(A3) + b4
A4 = Y_tilda = sigmoid(Z4)
test_size = test_X.shape[1]
count = 0
for i in range(test_size):
    if np.argmax(test_Y.T[i,:]) == np.argmax(Y_tilda.T[i,:]):
        count += 1

accuracy = count*100/test_size

print( accuracy)



