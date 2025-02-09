import numpy as np
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


