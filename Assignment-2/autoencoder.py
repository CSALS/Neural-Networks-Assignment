import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import getData
from MLP_auto import MLP as MLP_auto
from MLP import MLP

if __name__=='__main__':
    X, y_cat = getData('data.mat')

    # m = number of feature vectors
    m = X.shape[0]
    # n = number of features
    n = X.shape[1]

    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0]),:]
    y_train = y_cat[:int(train_percent*X.shape[0]),:]
    X_test = X[int(train_percent*X.shape[0]):,:]
    y_test = y_cat[int(train_percent*X.shape[0]):,:]

    # hidden layers array
    Layers = [42,24,12]
    alpha = 0.5
    max_iter = 30 

    # pretraining 3 autoencoders
    model11 = MLP_auto([n, Layers[0]], ['sigmoid'])
    print("pre-training autoencoder 1")
    model11.train(X_train,X_train, alpha, 12, max_iter)
  
    out1 = model11.output_hidden(X_train)

    model12 = MLP_auto([Layers[0], Layers[1]], ['sigmoid'])
    print("pre-training autoencoder 2")
    model12.train(out1, out1, alpha, 12, max_iter)
    
    out2 = model12.output_hidden(out1)

    model13 = MLP_auto([Layers[1], Layers[2]], ['sigmoid'])
    print("pre-training autoencoder 3")
    model13.train(out2, out2, alpha, 12, max_iter)
    
    # finetuning the stacked autoencoder
    print("fine tuning stacked autoencoder")

    # deep neural network using stacke autoencoder
    final_model = MLP([n, *Layers, 2], ['sigmoid','sigmoid','sigmoid','sigmoid'])
    # final_model.W_list[0:3] = model.W_list[0:3]
    final_model.W_list[0] = model11.W_list[0]
    final_model.W_list[1] = model12.W_list[0]
    final_model.W_list[2] = model13.W_list[0]



    # training deep neural network
    alpha = 0.5
    batch_size = 12
    max_iter = 200
    final_model.train(X_train, y_train,X_test,y_test, alpha, batch_size, max_iter)
    print(final_model.accuracy(X_test,y_test))
    print(final_model.conf_mat(X_test, y_test))


