import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.cluster import KMeans


def gaussian(x, mu, std):
    print(np.exp((-1/(2*(std**2))) * np.sum((x - mu)**2)))
    return np.exp((-1/(2*(std**2))) * np.sum((x - mu)**2))

def multiquadric(x, mu, std):
    return np.sum(np.sqrt(np.sum(np.square(x-mu)) + (std**2)))

def linear(x, mu, std):
    print(np.sum(abs(x - mu)))
    return np.sum(abs(x - mu))

def train(X_train, y_train):
    # m = number of feature vectors
    m = X_train.shape[0]
    # n = number of features
    n = X.shape[1]
    kmeans = KMeans(n_clusters=k, max_iter=max_iter , random_state=0).fit(X_train)
    
    means = kmeans.cluster_centers_
    assignments = kmeans.predict(X_train)
    stds = []
    for i in range(k):
        temp = X_train[(assignments==i)]
        print("sum is ", sum(abs(temp-means[i])))
        stds.append((1/temp.shape[0])*sum(abs(temp-means[i])))
    stds = np.array(stds)

    H = np.ndarray((m,k))
    for i in range(m):
        for j in range(k):
            H[i][j] = kernel(X_train[i], means[j], stds[j])
    W = np.dot(np.linalg.pinv(H),y_train)
    return {'W': W, 'means':means, 'stds':stds}

def test(X_test, y_test, W, means, stds):
    # testing data
    mt = X_test.shape[0]
    Ht = np.ndarray((mt,k))
    for i in range(mt):
        for j in range(k):
            Ht[i][j] = kernel(X_test[i], means[j], stds[j])
    yt = np.dot(Ht,W)

    cost = np.mean((yt-y_test)**2)
    print(f'cost: {cost}')
    acc = 0
    for i in range(y_test.shape[0]):
        acc += (np.argmax(yt[i]) == np.argmax(y_test[i]))
    acc /=y_test.shape[0]
    print(f'accuracy = {acc}')
    #with open('./Results/rbfnn/log.txt','a+') as log_file:
        #log_file.write(f'\n\n k = {k} function = {kernel} max_iter = {max_iter} cost = {cost}  accuracy = {acc}')
    return (cost, acc)

if __name__=='__main__':
    # data input
    data = pd.DataFrame(loadmat('./data.mat')['x'])
    data = data.sample(frac=1).reset_index(drop=True)
    X = data.loc[:,[i for i in range(72)]].values
    y = data.loc[:,72:73].values
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    X = (X- X.mean())/X.std()

    kernel = multiquadric
    kernel = multiquadric
    kernel = linear

    # splitting data
    max_iter = 100
    maxAcc = 0.0
    maxK = 0
    for k in range(100,1000,5):
        print('Iteration ', k)
        #K fold
        k_fold = 4
        Nk = X.shape[0]//k_fold
        accs = []
        for i in range(0, X.shape[0], Nk):
            X_test = X[i:i+Nk,:]
            X_train = np.delete(X,range(i,i+Nk),0)
            y_test = y_cat[i:i+Nk]
            y_train = np.delete(y_cat,range(i,i+Nk),0)

            params = train(X_train, y_train)
            cost,acc = test(X_test, y_test, params['W'], params['means'], params['stds'])
            accs.append(acc)
        print("\nAvg Accuracy: ", np.mean(accs))
        if(maxAcc < np.mean(accs)):
            maxAcc = np.mean(accs)
            maxK = k

maxK, maxAcc

if __name__=='__main__':
    # data input
    data = pd.DataFrame(loadmat('./data.mat')['x'])
    data = data.sample(frac=1).reset_index(drop=True)
    X = data.loc[:,[i for i in range(72)]].values
    y = data.loc[:,72:73].values
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    X = (X- X.mean())/X.std()

    kernel = multiquadric
    kernel = multiquadric
    kernel = linear

    # splitting data
    max_iter = 100
    maxAcc = 0.0
    maxK = 0
    for k in range(100,1000,5):
        print('Iteration ', k)
        #K fold
        k_fold = 4
        Nk = X.shape[0]//k_fold
        accs = []
        for i in range(0, X.shape[0], Nk):
            X_test = X[i:i+Nk,:]
            X_train = np.delete(X,range(i,i+Nk),0)
            y_test = y_cat[i:i+Nk]
            y_train = np.delete(y_cat,range(i,i+Nk),0)

            params = train(X_train, y_train)
            cost,acc = test(X_test, y_test, params['W'], params['means'], params['stds'])
            accs.append(acc)
        print("\nAvg Accuracy: ", np.mean(accs))
        if(maxAcc < np.mean(accs)):
            maxAcc = np.mean(accs)
            maxK = k
