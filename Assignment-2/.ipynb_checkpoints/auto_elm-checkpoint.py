from preprocessing import NormalScaler
import numpy as np
import pandas as pd
from scipy.io import loadmat
from elm import ELM
from MLP_auto import MLP


if __name__=='__main__':
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
    data = data.sample(frac=1).reset_index(drop=True)
    X = data.loc[:,:71].values
    y = data.loc[:,72:73].values
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    scaler = NormalScaler()
    for j in range(X.shape[1]):
        scaler.fit(X[:,j])
        X[:,j] = scaler.transform(X[:,j])

    # m = number of feature vectors
    m = X.shape[0]
    # n = number of features
    n = X.shape[1]

    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0]),:]
    y_train = y_cat[:int(train_percent*X.shape[0]),:]
    X_test = X[int(train_percent*X.shape[0]):,:]
    y_test = y_cat[int(train_percent*X.shape[0]):,:]

    alpha = 0.6
    max_iter = 25
    batch_size = 12

    # pretraining 3 autoencoders
    model11 = MLP([n, 42], ['sigmoid'])
    print("pre training autoencoder 1")
    model11.train(X_train,X_train, alpha, batch_size, max_iter)
  
    out1 = model11.output_hidden(X_train)

    model12 = MLP([42, 24], ['sigmoid'])
    print("pre training autoencoder 2")
    model12.train(out1, out1, alpha, batch_size, max_iter)
    
    # stacking the pretrained autoencoders
    model = MLP([n, 42, 24], ['sigmoid','sigmoid'])
    
    # initializing pretrained weights
    model.W_list[0] = model11.W_list[0]
    model.W_list[-1] = model11.W_list[0].T
    
    model.W_list[1] = model12.W_list[0]
    model.W_list[-2] = model12.W_list[0].T
    
    # finetuning the stacked autoencoder
    # print("training stacked autoencoder")
    # model.train(X_train, X_train, alpha, batch_size, 50)

    print("\nELM part of the neural network\n")
    
    elm_X_train = np.ndarray((X_train.shape[0], model.A[2].shape[0]))
    elm_X_test = np.ndarray((X_test.shape[0], model.A[2].shape[0]))

    for i in range(X_train.shape[0]):
        model.forward_prop(X_train[i])
        elm_X_train[i] = model.A[2].reshape(-1,)
    for i in range(X_test.shape[0]):
        model.forward_prop(X_test[i])
        elm_X_test[i] = model.A[2].reshape(-1,)

    elm_model = ELM(128, elm_X_train, y_train, 'tanh')
    elm_model.test(elm_X_test,y_test)
    elm_model.test(elm_X_train,y_train)
