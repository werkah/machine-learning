
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


data_breast_cancer = datasets.load_breast_cancer()
print(data_breast_cancer['DESCR'])


# In[3]:


X_b = data_breast_cancer.data[:, (3, 4)] #area, smoothness
y_b = data_breast_cancer.target


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=.2, random_state=42)


# In[5]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[6]:


svm_clf = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])

svm_scaler = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge",random_state=42)),])


# In[7]:


svm_clf.fit(X_train, y_train)
svm_scaler.fit(X_train, y_train)

acc_svm_train = svm_clf.score(X_train, y_train)
acc_svm_test = svm_clf.score(X_test, y_test)
acc_svm_scaled_train = svm_scaler.score(X_train, y_train)
acc_svm_scaled_test = svm_scaler.score(X_test, y_test)

bc_acc = [acc_svm_train, acc_svm_test, acc_svm_scaled_train, acc_svm_scaled_test]


# In[8]:


print(bc_acc)


# In[9]:


import pickle

with open('bc_acc.pkl', 'wb') as file:
    pickle.dump(bc_acc, file)


# In[10]:


data_iris = datasets.load_iris()
print(data_iris['DESCR'])


# In[11]:


import numpy as np


# In[12]:


X_ir = data_iris.data[:, (2, 3)]  #petal l/w
y_ir = (data_iris["target"] == 2).astype(np.float64)


# In[13]:


Xi_train, Xi_test, yi_train, yi_test = train_test_split(X_ir, y_ir, test_size=.2, random_state=42)


# In[14]:


isvm_clf = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])

isvm_scaler = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge",random_state=42)),])


# In[15]:


isvm_clf.fit(Xi_train, yi_train)
isvm_scaler.fit(Xi_train, yi_train)

iacc_svm_train = isvm_clf.score(Xi_train, yi_train)
iacc_svm_test = isvm_clf.score(Xi_test, yi_test)
iacc_svm_scaled_train = isvm_scaler.score(Xi_train, yi_train)
iacc_svm_scaled_test = isvm_scaler.score(Xi_test, yi_test)

ir_acc = [iacc_svm_train, iacc_svm_test, iacc_svm_scaled_train, iacc_svm_scaled_test]


# In[16]:


print(ir_acc)


# In[17]:


with open('iris_acc.pkl', 'wb') as file:
    pickle.dump(ir_acc, file)

