"""
This file contains the classes that implement Linear Regression
"""
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:
    """
    This class implements Linear Regression using both
    Batch Gradient Descent and Stochastic Gradient descent.
    Class attributes:
		W  	     : current set of weights
		W_arr    : list of weights at each iteration
		Cost	 : current cost
		cost_arr : list of costs at each iteration
    """
    def init_weights(self, s):
        """
        This method initializes the weight matrix
        as a column vector with shape = (X rows+1, 1)
        """
        np.random.seed(11)
        self.W = np.random.randn(s[1],1)
        self.W_arr = []
        self.cost_arr = []
        self.cost = 0
        
    def get_cost(self, X, y, W):
        """
        This function returns the cost with the given set of weights
        using the formula
        	J =1/2 ∑_(i=0)^m▒(h_ω (x^i )-y^i )^2         
        """
        total_cost = sum(np.square(np.matmul(X,W)-y.reshape(-1,1)))[0]
        return (0.5/X.shape[0])*total_cost
    
    def add_bias(self, X):
        """
        This function adds bias (a column of ones) to the feature vector X.
        """
        bias = np.ones((X.shape[0],1))
        return np.concatenate((bias,X), axis=1)
    
    def get_h_i(self, X, i, W):
        """
        This function returns the hypothesis of ith feature vector
        with the given weights W.
			h_w (x^i )=∑_(j=0)^n▒〖w_j 〖x^i〗_j 〗=x^i w
        """
        h_i = np.matmul(X[i].reshape(1,-1),W)
        return h_i[0][0]

    def batch_grad_descent(self, X, y, alpha, max_iter):
        """
        This function implements the Batch Gradient Descent algorithm.
        It runs for multiple iterations until either the weights converge or 
        iterations reach max_iter. At each iteration the weights are updated using 
        the following rule
            repeat until convergence{
                w_j^(t+1)=w_j^t-α∑_(i=1)^m▒〖(h_w (x^i )-y^i ) x_j^i 〗
            }            
        """
        W_new = self.W.copy()
        for _ in range(max_iter):
            temp = np.matmul(X,self.W) - y.reshape(-1,1)
            for j in range(X.shape[1]):
                W_new[j][0] = self.W[j][0] - (alpha/X.shape[0])*(sum(temp*X[:,j:j+1])[0])
            self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
            if len(self.W_arr)>1:
                if sum(abs(self.W_arr[-2]-self.W_arr[-1]))<0.0001:
                    break

        return W_new
			
    def stochastic_grad_descent(self, X, y, alpha, max_iter):
        """
        This function implements the Stochastic Gradient Descent algorithm.
        It runs for multiple iterations until either the weights converge or 
        iterations reach max_iter. Weights are updated for every row of the 
        training set.

            repeat until convergence{
                randomly shuffle the feature matrix rows
                for each feature vector x^i {
                    update all weights j -> 0 to n+1
                    w_j^(t+1)=w_j^t-α(h_w (x^i )-y^i ) x_j^i
                }
            }            
        """

        mat = np.concatenate((X,y.reshape(-1,1)), axis=1)
        for _ in range(max_iter):
            W_new = self.W.copy()
            np.random.shuffle(mat)
            X = mat[:,0:3]
            y = mat[:,3]
            for i in range(X.shape[0]):    
                temp = np.matmul(X[i,:],self.W) - y[i]
                for j in range(X.shape[1]):
                    W_new[j][0] = self.W[j][0] - (alpha)*(temp[0]*X[i,j])
                self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
            if len(self.W_arr)>1:
                if sum(abs(self.W_arr[-2]-self.W_arr[-1]))<0.0001:
                    break
        return self.W

    def train(self, X, y, alpha, max_iter=100, option="batch"):
        """
        This function initiates the training process.
        It runs batch gradient descent by default and can also run 
        Stochastic gradient descent if the argument is passed.
        
        returns the cost list which has costs at every training iteration.
        """
		# adding bias column to feature matrix X.
        X = self.add_bias(X)
        self.init_weights(X.shape)
        if option=="batch":
            self.batch_grad_descent(X,y,alpha,max_iter)
        elif option=="stochastic":
            self.stochastic_grad_descent(X,y,alpha,max_iter)
        self.cost = self.cost_arr[-1]
        return self.cost_arr
        
    def test(self,X,W=""):
        """
        This function takes a feature matrix as test data and 
        predicts the target values using the trained weights.

        returns the predicted target values.
        """
        if W=="":
            W = self.W

        X = self.add_bias(X)
        y_pred = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                y_pred[i] += X[i][j]*W[j][0]
        return y_pred

if __name__ == "__main__":
    model = LinearRegression()

    # data input
    data = pd.read_excel("./data.xlsx", header=None)
    X = data.loc[:,0:1].values
    y = data.loc[:,2].values

    # data preprocessing (Normal scaling)
    X = (X - X.mean())/X.std()

    # Training the model by choosing alpha and max_iter values.
	# gradient descent algorithm can be set as either ‘batch’ or ‘stochastic’
	# in this function call.
    alpha = 0.007
    algo = 'stochastic'
    # alpha = 0.26
    # algo = 'batch'
    max_iter = 100
    arr = model.train(X,y,alpha,max_iter,algo)
    
    print("weights: ",model.W)
    print("Total Cost: ",model.cost)

    # visualization of cost function.

    W_arr = np.array(model.W_arr)
    res = 100
    bounds = [2,0.6]
    xx = np.linspace((np.min(W_arr[:,1])-bounds[0]), (np.max(W_arr[:,1])+bounds[0]), res)
    yy = np.linspace(np.min(W_arr[:,2])-bounds[1], np.max(W_arr[:,2])+bounds[1]+1, res)
    minw0 = W_arr[-1][0][0]

    r = np.ndarray((res,res))
    s = np.ndarray((res,res))
    z = np.ndarray((res,res))
	
    for i in range(res):
        for j in range(res):
            z[i][j] = model.get_cost(model.add_bias(X), y, np.array([minw0,xx[i],yy[j]]).reshape(-1,1))
            r[i][j] = xx[i]
            s[i][j] = yy[j]
	
    print(r.shape,s.shape,z.shape)
    ax = plt.axes(projection='3d')
    ax.plot_surface(r, s, z,cmap='coolwarm')
    # ax.plot(W_arr[:,1], W_arr[:,2], model.cost_arr,c='red')
    ax.text2D(0.05, 0.95, "3D surface plot of cost function ({2})\n alpha={0} max_iter={1}".format(alpha,max_iter,algo), transform=ax.transAxes)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("cost")
    # plt.savefig("./Results/lin_reg/{2}_{0}_{1}_surf.png".format(alpha,max_iter,algo))
    plt.show()

    # 2d contour plot of cost function
    plt.figure()
    plt.title("2d contour plot of cost function ({2})\n alpha={0} max_iter={1}".format(alpha,max_iter,algo))
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.contour(r,s,z.reshape(res,res),levels=25)
    # plt.scatter(W_arr[:,1].ravel(),W_arr[:,2].ravel(),c=model.cost_arr)
    plt.plot(W_arr[:,1].ravel(),W_arr[:,2].ravel())
    # plt.savefig("./Results/lin_reg/{2}_{0}_{1}_contour.png".format(alpha,max_iter,algo))
    plt.show()
    
    # 2d line plot of cost vs iteration
    plt.figure()
    plt.plot(model.cost_arr)
    plt.title("Cost Function vs iteration plot ({2})\n alpha={0} max_iter={1}".format(alpha,max_iter,algo))
    plt.xlabel("iteration")
    plt.ylabel("cost")
    # plt.savefig("./Results/lin_reg/{2}_{0}_{1}_cost_iter.png".format(alpha,max_iter,algo))

    plt.show()