#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets 
data_breast_cancer=datasets.load_breast_cancer()


# In[2]:


from sklearn.datasets import load_iris
data_iris=load_iris()


# In[3]:


from sklearn.decomposition import PCA


# In[4]:


pca_dbc = PCA(n_components=0.9)
pca_di = PCA(n_components=0.9)


# In[5]:


reduced_dbc = pca_dbc.fit_transform(data_breast_cancer.data)
reduced_di = pca_di.fit_transform(data_iris.data)


# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[7]:


dbc_scaled = scaler.fit_transform(data_breast_cancer.data)
di_scaled =scaler.fit_transform(data_iris.data)


# In[8]:


pca_dbc_scaled = PCA(n_components=0.9)
pca_di_scaled = PCA(n_components=0.9)


# In[9]:


dbc_scaled_t = pca_dbc_scaled.fit_transform(dbc_scaled.data)
di_scaled_t = pca_di_scaled.fit_transform(di_scaled.data)


# In[10]:


pca_b = list(pca_dbc_scaled.explained_variance_ratio_)
pca_i = list(pca_di_scaled.explained_variance_ratio_)


# In[11]:


import pickle
with open("pca_bc.pkl", 'wb') as file:
    pickle.dump(pca_b,file)


# In[12]:


with open("pca_ir.pkl", 'wb') as f:
    pickle.dump(pca_i, f)


# In[13]:


import numpy as np
i_idx = []
b_idx = []


# In[14]:


for row in pca_di_scaled.components_:
    i_idx.append(np.argmax(row))


# In[15]:


for row in pca_dbc_scaled.components_:
    b_idx.append(np.argmax(row))


# In[16]:


with open("idx_bc.pkl", 'wb') as file:
    pickle.dump(b_idx, file)


# In[17]:


with open("idx_ir.pkl", 'wb') as file:
    pickle.dump(i_idx, file)




