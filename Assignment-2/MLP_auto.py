import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import getData

class MLP:
    def __init__(self, layers_arr, activ_arr):
        '''
        This is the constructor for the class MLP used for autoencoder.
        The following attributes are initialized in this constructor
        W_list: a list of weight matrices. Each weight matrix 
                connects one layer to the next one.
        b_list: a list of bias weights. Each layer has a bias weight.
        grad_W: list of gradient matrices. Each matrix has gradients 
                of the J wrt weights in the corresponding weight matrix in W_list.
        b_grad: list of gradients of J wrt biases.
        A: a list of vectors. Each vector represents a layer of neurons and their values.
        derr: a list of vectors containing errors generated using backpropogation.
        L: number of layers in the encoder part of the autoencoder excluding input layer
        L2: total number of layers in the autoencoder excluding input layer
        activ_arr: contains activation function to be used for a particular layer
        '''
        self.W_list = []
        self.b_list = []
        self.grad_W = []
        self.grad_b = []
        self.A = []
        self.derr = []
        self.L = len(layers_arr)-1
        self.activ_arr = activ_arr
        self.n = layers_arr[0]
        # initializing encoder part of the autoencoder
        for l in range(self.L):
            self.b_list.append(np.random.randn(1,).astype(np.float64)[0])
            self.grad_b.append(np.random.randn(1,).astype(np.float64)[0])
            self.W_list.append(np.random.randn( layers_arr[l+1], layers_arr[l]).astype(np.float64))
            self.grad_W.append(np.random.randn( layers_arr[l+1], layers_arr[l]).astype(np.float64))
        for l in range(self.L+1):
            self.A.append(np.ndarray(shape=( layers_arr[l], 1 )).astype(np.float64))
            self.derr.append(np.ndarray(shape=( layers_arr[l], 1 )).astype(np.float64))

        # initializing decoder part of the autoencoder
        for l in range(self.L-1, -1, -1):
            self.b_list.append(self.b_list[l])
            self.W_list.append(self.W_list[l].T)
            self.grad_b.append(self.grad_b[l])
            self.grad_W.append(self.grad_W[l].T)
        for l in range(self.L-1, -1, -1):
            self.A.append(self.A[l].copy())
            self.derr.append(self.derr[l].copy())
            self.activ_arr.append(self.activ_arr[l])
        self.L2 = 2*self.L

        # vectorized versions of activation functions
        self.relu_v = np.vectorize(self.relu)
        self.sigmoid_v = np.vectorize(self.sigmoid)
        self.linear_v = np.vectorize(self.linear)
    
    def compute_Z(self, l):
        '''
        returns the output of a layer of neurons before activation
        '''
        return np.matmul(self.W_list[l-1], self.A[l-1]) + self.b_list[l-1]

    def activation(self, Z, activ, deriv = 0):
        '''
        This function returns the activated output of a layer of neurons.
        '''
        if(activ=='sigmoid'):
            return self.sigmoid_v(Z, deriv)
        elif(activ=='relu'):
            return self.relu_v(Z, deriv)
        else:
            return self.linear(Z, deriv)

    def relu(self, x, deriv = 0):
        if deriv==1: 
            return 0 if x<0 else 1 
        return 0 if x<0 else x
    
    def linear(self, x, deriv = 0):
        if deriv==1: 
            return x
        return x

    def sigmoid(self, x, deriv = 0):
        if deriv==1:
            return self.sigmoid(x)*(1-self.sigmoid(x))
        return 1/(1+np.exp(-x))

    def forward_prop(self, X_i):
        '''
        This function takes ith data vector and propogates
        it forward in the neural network.
        '''
        self.A[0] = X_i.reshape(-1,1)
        for l in range(1,self.L2+1):
            self.A[l] = self.activation(self.compute_Z(l), self.activ_arr[l-1])
    
    def output_hidden(self, X):
        '''
        This function feeds the feature vector X forward
        in the encoder part and returns the activations at
        the middle hidden layer.
        '''
        res = np.ndarray((X.shape[0],self.A[self.L].shape[0]), dtype=np.float64)
        for i in range(X.shape[0]):
            self.A[0] = X[i].reshape(-1,1)
            for l in range(1,self.L+1):
                self.A[l] = self.activation(self.compute_Z(l), self.activ_arr[l-1])
            res[i] = self.A[self.L].reshape(-1,)
        return res
        
    def train(self, X, y, alpha, batch_size, max_iter):
        '''
        This function takes the training data and target values,
        applies forward propogation, then applies backward propogation
        to update the weight matrices.
        mini-batch gradient descent has been used to update weights.
        '''
        m = y.shape[0]
        for iteration in range(max_iter):
            for i in range(0,m-batch_size+1,batch_size):
                for l in range(self.L): self.grad_b[l]=0
                for l in range(self.L): self.grad_W[l].fill(0)

                for j in range(i,i+batch_size):
                    # forward propogation
                    self.forward_prop(X[j])

                    # Backpropogation of errors
                    self.derr[self.L2] = (self.A[self.L2]-y[j].reshape(-1,1)) * self.activation(self.compute_Z(self.L2), self.activ_arr[self.L2-1], 1)
                    for l in range(self.L2-1, 0,-1):
                        self.derr[l] = self.activation(self.compute_Z(l), self.activ_arr[l-1], 1)*np.matmul(self.W_list[l].T, self.derr[l+1])
                
                    for l in range(self.L, 0,-1):
                        self.grad_b[l-1] += 2*np.mean(self.derr[l])
                        self.grad_W[l-1] += 2*np.matmul(self.derr[l], self.A[l-1].T)
                
                # weight update for encoder layers after backpropogating each batch
                # weights of the decoder are tied with those of encoder layers.
                for l in range(self.L, 0,-1):
                    self.b_list[l-1] -= (alpha/batch_size)*np.mean(self.grad_b[l-1])
                    self.W_list[l-1] -= (alpha/batch_size)*self.grad_W[l-1]
                    self.b_list[self.L2-l] = self.b_list[l-1]
                    self.W_list[self.L2-l] = self.W_list[l-1].T
            
            print("iteration: {0} ".format(iteration+1),end="  ")
            self.eval_cost(X,y)

    def eval_cost(self, X, y):
        cost = 0
        for i in range(y.shape[0]):
            # forward propogation
            self.forward_prop(X[i])
            cost += np.sum((self.A[self.L2]-y[i].reshape(-1,1))**2)
        print(" ",cost/y.shape[0])




