#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd


# In[2]:


mnist = fetch_openml('mnist_784', version=1)


# In[3]:


print((np.array(mnist.data.loc[42]).reshape(28,28)>0).astype(int))


# In[4]:


mnist.keys()


# In[5]:


X = mnist.data
y = mnist.target
print(X)
print(y)


# In[6]:


y = y.sort_values()
print(y)


# In[7]:


X = X.reindex(y.index)
print(X)


# In[8]:


X_train, X_test=X[:56000], X[56000:]
y_train, y_test=y[:56000], y[56000:]


# In[9]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[10]:


np.unique(y_train)


# In[11]:


np.unique(y_test)


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


np.unique(y_train)


# In[15]:


np.unique(y_test)


# In[16]:


from sklearn.linear_model import SGDClassifier


# In[17]:


y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')


# In[18]:


print(y_train_0)


# In[19]:


print(y_test_0)


# In[20]:


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[21]:


accuracy_train = sgd_clf.score(X_train, y_train_0)


# In[22]:


sgd_clf.fit(X_test, y_test_0)
accuracy_test = sgd_clf.score(X_test, y_test_0)


# In[23]:


print(accuracy_train)
print(accuracy_test)


# In[24]:


accuracy = [accuracy_train, accuracy_test]
print(accuracy)


# In[25]:


import pickle
with open('sgd_acc.pkl', 'wb') as w1:
    pickle.dump(accuracy, w1)


# In[26]:


from sklearn.model_selection import cross_val_score
train_cross = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)


# In[28]:


print(train_cross)


# In[29]:


with open('sgd_cva.pkl', 'wb') as w2:
    pickle.dump(train_cross, w2)


# In[30]:


clf_all = SGDClassifier(random_state=42, n_jobs=-1)
clf_all.fit(X_train, y_train)


# In[31]:


from sklearn.model_selection import cross_val_predict
y_test_pred = cross_val_predict(clf_all, X_test, y_test, cv = 3, n_jobs = -1)


# In[32]:


from sklearn.metrics import confusion_matrix

conf_m = confusion_matrix(y_test, y_test_pred)


# In[33]:


with open('sgd_cmx.pkl', 'wb') as w3:
    pickle.dump(conf_m, w3)

