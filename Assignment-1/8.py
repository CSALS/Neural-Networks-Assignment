#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Multiclass Logistic Regression using "one vs all" and "one vs one" 
#and holdout crossvalidation technique


# In[2]:


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#only for jupyter notebook
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataSet = 'data4.xlsx'


# In[4]:


#header = None states that there is no header row or else it would take first row of our data as header.
df = pd.read_excel(dataSet,sheet_name='Sheet1',header=None)


# In[25]:


valueArray = df.to_numpy()
np.random.shuffle(valueArray)
#Hold out cross validation 60 - 40
trainSize = int((6*valueArray.shape[0])/10)
trainData , testData = valueArray[0:trainSize,0:] , valueArray[trainSize:,0:]
X_train , X_test = trainData[0:,0:7] , testData[0:,0:7]
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()
y_train , y_test = trainData[0:,7:] , testData[0:,7:]


# In[26]:


#Binary Logistic Regression -> classifies between 1 and 0
def sigmoid(z):
    return 1.0/ (1.0 + math.exp(-z))

def hypothesis(X,w,bias):
    sum = 0.0
    for index_feature,feature in enumerate(X):
        sum += w[index_feature][0] * feature
    sum += bias
    return sigmoid(sum)

def gradient(X,y,index_feature,w,bias):
    grad = 0.0
    if index_feature == -1 :
        #find gradient for bias
        for index in range(X.shape[0]):
            grad += (hypothesis(X[index],w,bias) - y[index][0])
    else:
        #find gradient for w[index][0]
        for index in range(X.shape[0]):
            grad += (hypothesis(X[index],w,bias) - y[index][0]) * X[index][index_feature]
            
    return grad

def logisticRegression(X,y,w,bias,alpha,iterations):
    for index in range(iterations):
        #update 7 weights and bias
        #w = w - alpha/m * gradient
        m = X.shape[0]
        w0 = w[0][0] - (alpha/m) * gradient(X,y,0,w,bias)
        w1 = w[1][0] - (alpha/m) * gradient(X,y,1,w,bias)
        w2 = w[2][0] - (alpha/m) * gradient(X,y,2,w,bias)
        w3 = w[3][0] - (alpha/m) * gradient(X,y,3,w,bias)
        w4 = w[4][0] - (alpha/m) * gradient(X,y,4,w,bias)
        w5 = w[5][0] - (alpha/m) * gradient(X,y,5,w,bias)
        w6 = w[6][0] - (alpha/m) * gradient(X,y,6,w,bias)
        b  = bias - (alpha/m) * gradient(X,y,-1,w,bias)
        w[0][0] = w0
        w[1][0] = w1
        w[2][0] = w2
        w[3][0] = w3
        w[4][0] = w4
        w[5][0] = w5
        w[6][0] = w6
        bias = b
    return [w,bias]


# In[27]:


"""
One vs All

3 classes = 1 , 2 , 3 
We will construct 3 models

Model 1 -> 1 = 1 and 2,3 = 0
Model 2 -> 2 = 1 and 1,3 = 0
Model 3 -> 3 = 1 and 1,2 = 0

final_predicted_value = max(model1_hypothesis,model2_hypothesis,model3_hypothesis,model4_hypothesis)

"""


# In[28]:


def oneVsAllTrain(X,y,alpha,iterations):
    np.random.seed(111) 
    w = np.random.rand(7,1) #Weight matrix with random values 7*1 matrix
    #w[0][0] = w[1][0] = w[2][0] = w[3][0] = w[4][0] = w[5][0] = w[6][0] = 0.0
    bias = 1
    
    y1 = y2 = y3 = np.zeros(shape = (y.shape[0],1))
    #Model - 1
    for index_target,target in enumerate(y):
        if target == 2.0 or target == 3.0 :
            y1[index_target][0] = 0.0
        elif target == 1.0:
            y1[index_target][0] = 1.0
    
    w = np.random.rand(7,1)
    bias = 1
    parameters = logisticRegression(X,y1,w,bias,alpha,iterations)
    model1_w = parameters[0]
    model1_bias = parameters[1]
    #Model - 2
    for index_target,target in enumerate(y):
        if target == 1.0 or target == 3.0 :
            y2[index_target][0] = 0.0
        elif target == 2.0:
            y2[index_target][0] = 1.0
        
    
    w = np.random.rand(7,1)
    bias = 1
    parameters = logisticRegression(X,y2,w,bias,alpha,iterations)
    model2_w = parameters[0]
    model2_bias = parameters[1]
    
    #Model - 3
    for index_target,target in enumerate(y):
        if target == 1.0 or target == 2.0 :
            y3[index_target][0] = 0.0
        elif target == 3.0:
            y3[index_target][0] = 1.0
    
    w = np.random.rand(7,1)
    bias = 1
    parameters = logisticRegression(X,y3,w,bias,alpha,iterations)
    model3_w = parameters[0]
    model3_bias = parameters[1]

    return [model1_w,model1_bias,model2_w,model2_bias,model3_w,model3_bias]


# In[29]:


models = oneVsAllTrain(X_train,y_train,0.45,1000)


# In[30]:


def oneVsAllTest(models,X,y):
    y_predicted = []
    model1_w = models[0]
    model1_bias = models[1]
    model2_w = models[2]
    model2_bias = models[3]
    model3_w = models[4]
    model3_bias = models[5]
    for index in range(y.shape[0]):
        h1 = hypothesis(X[index],model1_w,model1_bias)
        h2 = hypothesis(X[index],model2_w,model2_bias)
        h3 = hypothesis(X[index],model3_w,model3_bias)
        h = [h1,h2,h3]
        h = np.asarray(h)
        predicted_class = np.argmax(h) + 1
        y_predicted.append(predicted_class)
    return y_predicted


# In[31]:


y_predicted = oneVsAllTest(models,X_test,y_test)


# In[32]:


"""
        Confusion matrix for multiclass classifier
        
Actual Output                 Predicted Output
                    class1    class2    class3
class 1             u11       u12       u13

class 2             u21       u22       u23

class 3             u31       u32       u33

Individual Accuracy of class i = uii / ui1 + ui2 + ui3
Overall Accuracy = u11 + u22 + u33 / sum(uij)
"""


# In[33]:


u11 = u12 = u13 = u21 = u22 = u23 = u31 = u32 = u33 = 0
for index in range(y_test.shape[0]):
    if y_test[index][0] == 1 :
        if y_predicted[index] == 1:
            u11 += 1
        elif y_predicted[index] == 2:
            u12 += 1
        else:
            u13 += 1
    elif y_test[index][0] == 2:
        if y_predicted[index] == 1:
            u21 += 1
        elif y_predicted[index] == 2:
            u22 += 1
        else:
            u23 += 1
    elif y_test[index][0] == 3:
        if y_predicted[index] == 1:
            u31 += 1
        elif y_predicted[index] == 2:
            u32 += 1
        else:
            u33 += 1


# In[34]:


print("Confusion Matrix is :")
print(u11," ",u12," ",u13)
print(u21," ",u22," ",u23)
print(u31," ",u32," ",u33)


# In[35]:


IA_class1 = (float(u11)/float(u11 + u12 + u13))*100
IA_class2 = (float(u22)/float(u21 + u22 + u23))*100
IA_class3 = (float(u33)/float(u31 + u32 + u33))*100
Overall_Acc = (float(u11 + u22 + u33)/float(u11 + u12 + u13 + u21 + u22 + u23 + u31 + u32 + u33))*100


# In[36]:


print("Individual Accuracy of class 1 is : ",IA_class1,"%")
print("Individual Accuracy of class 2 is : ",IA_class2,"%")
print("Individual Accuracy of class 3 is : ",IA_class3,"%")
print("Overall Accuracy is : ",Overall_Acc,"%")


# In[37]:


"""
One vs One

3 classes = 1 , 2 , 3 
We will construct 3 models

Model 1 -> 1 = 0 and 2 = 1
Model 2 -> 1 = 0 and 3 = 1
Model 3 -> 2 = 0 and 3 = 1

final_predicted_value = mode(model1_prediction,model2_prediction,model3_prediction,model4_hypothesis)

"""


# In[38]:


"""One vs One training phase"""
alpha = 0.25
iterations = 1000

#Preparing training data for model1
trainData1 = []
for index in range(trainData.shape[0]):
    if trainData[index][7] != 3:
        trainData1.append(trainData[index])
trainData1 = np.asarray(trainData1)
for index in range(trainData1.shape[0]):
    if trainData1[index][7] == 1:
        trainData1[index][7]= 0
for index in range(trainData1.shape[0]):
    if trainData1[index][7] == 2:
        trainData1[index][7]= 1
#

X_train_1  = trainData1[0:,0:7] 
X_train_1 = (X_train_1 - X_train_1.mean())/X_train_1.std()
y_train_1  = trainData1[0:,7:]

np.random.seed(131) 
w = np.random.rand(7,1) #Weight matrix with random values 7*1 matrix
bias = 1
#Training model1
parameters = logisticRegression(X_train_1,y_train_1,w,bias,alpha,iterations)
model1_w = parameters[0]
model1_bias = parameters[1]

#Preparing training data for model2
trainData2 = []
for index in range(trainData.shape[0]):
    if trainData[index][7] != 2:
        trainData2.append(trainData[index])
trainData2 = np.asarray(trainData2)
for index in range(trainData2.shape[0]):
    if trainData2[index][7] == 1:
        trainData2[index][7]= 0
for index in range(trainData2.shape[0]):
    if trainData2[index][7] == 3:
        trainData2[index][7]= 1
#
X_train_2  = trainData2[0:,0:7]
X_train_2 = (X_train_2 - X_train_2.mean())/X_train_2.std()
y_train_2  = trainData2[0:,7:]

np.random.seed(131) 
w = np.random.rand(7,1) #Weight matrix with random values 7*1 matrix
bias = 1
#Training model2
parameters = logisticRegression(X_train_2,y_train_2,w,bias,alpha,iterations)
model2_w = parameters[0]
model2_bias = parameters[1]

#Preparing training data for model3
trainData3 = []
for index in range(trainData.shape[0]):
    if trainData[index][7] != 1:
        trainData3.append(trainData[index])
trainData3 = np.asarray(trainData3)
for index in range(trainData3.shape[0]):
    if trainData3[index][7] == 2:
        trainData3[index][7]= 0
for index in range(trainData3.shape[0]):
    if trainData3[index][7] == 3:
        trainData3[index][7]= 1
#
X_train_3  = trainData3[0:,0:7]
X_train_3 = (X_train_3 - X_train_3.mean())/X_train_3.std()
y_train_3  = trainData3[0:,7:]

np.random.seed(131) 
w = np.random.rand(7,1) #Weight matrix with random values 7*1 matrix
bias = 1
#Training model3
parameters = logisticRegression(X_train_3,y_train_3,w,bias,alpha,iterations)
model3_w = parameters[0]
model3_bias = parameters[1]


# In[39]:


"""One vs One testing phase"""
y_predicted = np.zeros(shape = (y_test.shape[0],1))

for index in range(X_test.shape[0]):
    y1 = y2 = y3 = 0
    h1 = hypothesis(X_test[index],model1_w,model1_bias)
    h2 = hypothesis(X_test[index],model2_w,model2_bias)
    h3 = hypothesis(X_test[index],model3_w,model3_bias)
    #model - 1 prediction
    if h1 >= 0.5 :
        y1 = 2
    else :
        y1 = 1
    #model - 2 prediction
    if h2 >= 0.5 :
        y2 = 3
    else :
        y2 = 1
    #model - 3 prediction
    if h3 >= 0.5 :
        y3 = 3
    else :
        y3 = 2
    one = two = three = 0
    y = [y1,y2,y3]
    for i in range(3):
        if y[i] == 1:
            one += 1
        elif y[i] == 2:
            two += 1
        else:
            three += 1
    if one > two and one > three :
        y_predicted[index] = 1
    elif two > one and two > three :
        y_predicted[index] = 2
    elif three > one and three > two :
        y_predicted[index] = 3
    else :
        if h1 >= h2 and h1 >= h3 :
            y_predicted[index] = 1
        elif h2 >= h1 and h2 >= h3 :
            y_predicted[index] = 2
        else :
            y_predicted[index] = 3
#     elif y1 == 2 and y2 == 1 and y3 == 3:
#         if h1 >= h2 and h1 >= h3 :
#             y_predicted[index] = 1
#         elif h2 >= h1 and h2 >= h3 :
#             y_predicted[index] = 2
#         else :
#             y_predicted[index] = 3
    #print(y)
        


# In[40]:


"""
        Confusion matrix for multiclass classifier
        
Actual Output                 Predicted Output
                    class1    class2    class3
class 1             u11       u12       u13

class 2             u21       u22       u23

class 3             u31       u32       u33

Individual Accuracy of class i = uii / ui1 + ui2 + ui3
Overall Accuracy = u11 + u22 + u33 / sum(uij)
"""


# In[41]:


u11 = u12 = u13 = u21 = u22 = u23 = u31 = u32 = u33 = 0
for index in range(y_test.shape[0]):
    if y_test[index][0] == 1 :
        if y_predicted[index][0] == 1:
            u11 += 1
        elif y_predicted[index][0] == 2:
            u12 += 1
        else:
            u13 += 1
    elif y_test[index][0] == 2:
        if y_predicted[index][0] == 1:
            u21 += 1
        elif y_predicted[index][0] == 2:
            u22 += 1
        else:
            u23 += 1
    elif y_test[index][0] == 3:
        if y_predicted[index][0] == 1:
            u31 += 1
        elif y_predicted[index][0] == 2:
            u32 += 1
        else:
            u33 += 1


# In[42]:


print("Confusion Matrix is :")
print(u11," ",u12," ",u13)
print(u21," ",u22," ",u23)
print(u31," ",u32," ",u33)


# In[43]:


IA_class1 = (float(u11)/float(u11 + u12 + u13))*100
IA_class2 = (float(u22)/float(u21 + u22 + u23))*100
IA_class3 = (float(u33)/float(u31 + u32 + u33))*100
Overall_Acc = (float(u11 + u22 + u33)/float(u11 + u12 + u13 + u21 + u22 + u23 + u31 + u32 + u33))*100


# In[44]:


print("Individual Accuracy of class 1 is : ",IA_class1,"%")
print("Individual Accuracy of class 2 is : ",IA_class2,"%")
print("Individual Accuracy of class 3 is : ",IA_class3,"%")
print("Overall Accuracy is : ",Overall_Acc,"%")


# In[ ]:





# In[ ]:




