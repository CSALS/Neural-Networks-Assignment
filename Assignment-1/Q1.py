#!/usr/bin/env python
# coding: utf-8

# In[36]:


#Linear Regression Using Batch Gradient Descent


# In[37]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#only for jupyter notebook
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


dataSet = 'data.xlsx'


# In[39]:


#header = None states that there is no header row or else it would take first row of our data as header.
df = pd.read_excel(dataSet,sheet_name='Sheet1',header=None)


# In[40]:


valueArray = df.to_numpy() #dataframe DF to numpyarray valueArray
m = valueArray.shape[0] # Number of training examples   numpyArray.shape = (no of rows,no of cols)
X = valueArray[0:,0:2] # Feature Vector [x1,x2] m*2 0: = 0,1,2 rows and 0:2 = 0,1 columns
X = np.insert(X,0,1,axis=1) # axis = 1 (every row insert 1 at 0 column) , Adding x0 = 1 in feature matrix m*3
y = valueArray[0:,2:] # Class label Vector [y]  0: = 0,1,2 rows and 2: = 2 column   y = m*1 matrix


# In[41]:


#Normalization
X[0:,1:2] = (X[0:,1:2] - X[0:,1:2].mean())/(X[0:,1:2].std())
X[0:,2:] = (X[0:,2:] - X[0:,2:].mean())/(X[0:,2:].std())


# In[42]:


#Implementing Batch Gradient Descent


# In[43]:


def computeCostFunction(X,y,w):
    sum = 0.0
    for index in range(X.shape[0]):
        sum += (y[index][0] - (w[0][0] * X[index][0] + w[1][0] * X[index][1] + w[2][0] * X[index][2]))**2
        #print(sum)
    return 0.5 * (sum/m)


# In[44]:


alpha = 0.13
iters = 50
costs = []
iterations = []
weight1 = []
weight2 = []
np.random.seed(11) 
w = np.random.rand(3,1) #Weight matrix with random values 3*1 matrix
print(w)


# In[45]:


for i in range(iters):
    #wj <- wj + for all points (h(x) - y)*xj
    sum0 = sum1 = sum2 = 0.0 # will hold new values of weights after weight update
    for xIndex in range(m):
        sum0 += (X[xIndex][0]*w[0][0] + X[xIndex][1]*w[1][0] + X[xIndex][2]*w[2][0] - y[xIndex][0])*X[xIndex][0]
        sum1 += (X[xIndex][0]*w[0][0] + X[xIndex][1]*w[1][0] + X[xIndex][2]*w[2][0] - y[xIndex][0])*X[xIndex][1]
        sum2 += (X[xIndex][0]*w[0][0] + X[xIndex][1]*w[1][0] + X[xIndex][2]*w[2][0] - y[xIndex][0])*X[xIndex][2]
    w[0][0] = w[0][0] - alpha*sum0/m
    w[1][0] = w[1][0] - alpha*sum1/m
    w[2][0] = w[2][0] - alpha*sum2/m
    cost = computeCostFunction(X,y,w)
    costs.append(cost)
    iterations.append(i+1)
    weight1.append(w[1][0])
    weight2.append(w[2][0])


# In[46]:


#RESULTS


# In[47]:


#2D - Plot of cost function vs number of iterations
plt.title('Cost Function J vs Number of Iterations')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(iterations,costs)
plt.show()


# In[48]:


#3D Surface Graph of cost function vs w1 and w2


# In[55]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(weight1,weight2,costs,'red')
plt.show()


# In[30]:


costs[len(iterations)-1]


# In[31]:


w


# In[32]:


y_pred = []
for index in range(m):
        y_pred.append([(w[0][0] * X[index][0] + w[1][0] * X[index][1] + w[2][0] * X[index][2])])
mse = np.sum((y-y_pred)**2)
rmse = np.sqrt(mse/m)
print(rmse)


# In[33]:


y


# In[34]:


y_pred


# In[ ]:





# In[ ]:




