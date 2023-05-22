#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris=load_iris(as_frame=True)


# In[2]:


import pandas as pd
pd.concat([iris.data, iris.target], axis=1).plot.scatter(x='petal length (cm)', y='petal width (cm)', c='target',colormap='viridis', figsize=(10,4))


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
X = iris.data[["petal length (cm)", "petal width (cm)"]]
y = iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)
per_0 = Perceptron()
per_0.fit(X_train, y_train_0)


# In[5]:


y_train_1 = (y_train == 1).astype(int)
y_test_1 = (y_test == 1).astype(int)
per_1 = Perceptron()
per_1.fit(X_train, y_train_1)


# In[6]:


y_train_2 = (y_train == 2).astype(int)
y_test_2 = (y_test == 2).astype(int)
per_2 = Perceptron()
per_2.fit(X_train, y_train_2)


# In[7]:


weights = []


# In[8]:


for perceptron in [per_0, per_1, per_2]:
    w0 = perceptron.intercept_[0]
    w1 = perceptron.coef_[0][0]
    w2 = perceptron.coef_[0][1]
    weights.append((w0, w1, w2))


# In[9]:


weights


# In[10]:


from sklearn.metrics import accuracy_score

per_acc = [(accuracy_score(y_train_0, per_0.predict(X_train)), 
            accuracy_score(y_test_0, per_0.predict(X_test))),
           (accuracy_score(y_train_1, per_1.predict(X_train)), 
            accuracy_score(y_test_1, per_1.predict(X_test))),
           (accuracy_score(y_train_2, per_2.predict(X_train)), 
            accuracy_score(y_test_2, per_2.predict(X_test)))]
per_acc


# In[11]:


import pickle
with open("per_acc.pkl", 'wb') as file:
    pickle.dump(per_acc, file)


# In[12]:


with open("per_wght.pkl", 'wb') as file:
    pickle.dump(weights, file)


# In[13]:


import numpy as np
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,1,1,0])


# In[14]:


per_xor = Perceptron()
per_xor.fit(X, y)


# In[15]:


y_pred = per_xor.predict(X)
print(accuracy_score(y, y_pred))


# In[16]:


print(per_xor.intercept_)


# In[17]:


print(per_xor.coef_)


# In[18]:


from sklearn.neural_network import MLPClassifier


# In[19]:


model = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', max_iter=1000, learning_rate_init=0.1)
model.fit(X,y)


# In[20]:


accuracy = model.score(X, y)
accuracy


# In[21]:


predictions = model.predict(X)
predictions


# In[22]:


model = MLPClassifier(activation='relu', solver='adam', max_iter=1000, learning_rate_init=0.1)
model.fit(X,y)
accuracy = model.score(X, y)
predictions = model.predict(X)
accuracy


# In[23]:


model = MLPClassifier(activation='logistic', solver='adam', max_iter=1000, learning_rate_init=0.1)
model.fit(X,y)
accuracy = model.score(X, y)
predictions = model.predict(X)
accuracy


# In[24]:


model = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=1000, learning_rate_init=0.01)
model.fit(X,y)
accuracy = model.score(X, y)
predictions = model.predict(X)
accuracy


# In[25]:


model = MLPClassifier(activation='relu', solver='lbfgs', max_iter=8000, learning_rate_init=0.01)
model.fit(X,y)
accuracy = model.score(X, y)
predictions = model.predict(X)
print(accuracy, predictions)


# In[26]:


with open('mlp_xor.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[33]:


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
model = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='lbfgs', max_iter=1000, learning_rate_init=0.1)
model.coefs_= [np.array([[-0.5, 1], [1, 1]]), np.array([[-1], [1]])]
model.intercepts_= [np.array([-1.5, -0.5]), np.array([-0.5])]
model.fit(X,y)
accuracy = model.score(X,y)
predictions = model.predict(X)
print(accuracy, predictions)
with open('mlp_xor_fixed.pkl', 'wb') as file:
    pickle.dump(model, file)
