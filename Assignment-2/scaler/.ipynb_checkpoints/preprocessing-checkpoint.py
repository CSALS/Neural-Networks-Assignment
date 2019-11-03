import numpy
import scipy.io
def preprocess(fileName):
    mat = scipy.io.loadmat(fileName)
    valueArray = mat['x']
    X = valueArray[0:,:72]
    Y = valueArray[0:,-1]
    Y = numpy.reshape(Y,(Y.shape[0],1))
    Y = Y.astype(int)
    #Feature Scaling
    X = (X - numpy.mean(X,axis=0))/numpy.std(X,axis=0)
    Y = numpy.eye(len(numpy.unique(Y,axis=0)))[(Y.T).flatten()]
    return X,Y


X,Y = preprocess('data.mat')


