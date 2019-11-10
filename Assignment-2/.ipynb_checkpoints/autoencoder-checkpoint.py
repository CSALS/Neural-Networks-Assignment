import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import getData
from MLP_auto import MLP as MLP_auto
from MLP import MLP

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

    # # stacking the pretrained autoencoders
    # model = MLP_auto([n, *Layers], ['sigmoid','sigmoid','sigmoid'])
     
    # # initializing pretrained weights
    # model.W_list[0] = model11.W_list[0]
    # model.W_list[5] = model11.W_list[0].T
    
    # model.W_list[1] = model12.W_list[0]
    # model.W_list[4] = model12.W_list[0].T
    
    # model.W_list[2] = model13.W_list[0]
    # model.W_list[3] = model13.W_list[0].T
    
    # finetuning the stacked autoencoder
    print("fine tuning stacked autoencoder")
    # model.train(X_train, X_train, alpha, 12, 20)

    # d_X_train = np.ndarray((X_train.shape[0], model.A[3].shape[0]))
    # d_X_test = np.ndarray((X_test.shape[0], model.A[3].shape[0]))

    # for i in range(X_train.shape[0]):
    #     model.forward_prop(X_train[i])
    #     d_X_train[i] = model.A[3].reshape(-1,)
    # for i in range(X_test.shape[0]):
    #     model.forward_prop(X_test[i])
    #     d_X_test[i] = model.A[3].reshape(-1,)
    
    # alpha = 0.3
    # batch_size = 16
    # max_iter = 100
    # final_model = MLP([Layers[-1], 2], ['sigmoid'])
    # final_model.train(d_X_train, y_train, d_X_test, y_test, alpha, batch_size, max_iter)
    # print(final_model.accuracy(d_X_test, y_test))

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
