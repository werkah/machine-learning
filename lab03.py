#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn.neighbors
from sklearn.preprocessing import PolynomialFeatures
import pickle

size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[2]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


#linear


# In[4]:


X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


# In[5]:


lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


# In[6]:


y_train_pred = lin_reg.predict(X_train)
lin_reg_train_mse = mean_squared_error(y_train, y_train_pred)


# In[7]:


y_test_pred = lin_reg.predict(X_test)
lin_reg_test_mse = mean_squared_error(y_test, y_test_pred)


# In[8]:


#knn


# In[9]:


knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train, y_train)
knn_3_reg_train_pred = knn_3_reg.predict(X_train)
knn_3_reg_test_pred = knn_3_reg.predict(X_test)


# In[10]:


knn_3_train_mse = mean_squared_error(y_train, knn_3_reg_train_pred)
knn_3_test_mse = mean_squared_error(y_test, knn_3_reg_test_pred)


# In[11]:


knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_train, y_train)
knn_5_reg_train_pred = knn_5_reg.predict(X_train)
knn_5_reg_test_pred = knn_5_reg.predict(X_test)


# In[12]:


knn_5_train_mse = mean_squared_error(y_train, knn_5_reg_train_pred)
knn_5_test_mse = mean_squared_error(y_test, knn_5_reg_test_pred)


# In[13]:


#poly


# In[14]:


poly_feature_2 = PolynomialFeatures(degree = 2, include_bias = False)
X_2_poly_train = poly_feature_2.fit_transform(X_train)
X_2_poly_test = poly_feature_2.fit_transform(X_test)
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_2_poly_train, y_train)


# In[15]:


y_pred_poly_2_train = poly_2_reg.predict(X_2_poly_train)
y_pred_poly_2_test = poly_2_reg.predict(X_2_poly_test)


# In[16]:


poly_2_train_mse = mean_squared_error(y_train, y_pred_poly_2_train)
poly_2_test_mse = mean_squared_error(y_test, y_pred_poly_2_test)


# In[17]:


poly_feature_3 = PolynomialFeatures(degree = 3, include_bias = False)
X_3_poly_train = poly_feature_3.fit_transform(X_train)
X_3_poly_test = poly_feature_3.fit_transform(X_test)
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_3_poly_train, y_train)


# In[18]:


y_pred_poly_3_train = poly_3_reg.predict(X_3_poly_train)
y_pred_poly_3_test = poly_3_reg.predict(X_3_poly_test)


# In[19]:


poly_3_train_mse = mean_squared_error(y_train, y_pred_poly_3_train)
poly_3_test_mse = mean_squared_error(y_test, y_pred_poly_3_test)


# In[20]:


poly_feature_4 = PolynomialFeatures(degree = 4, include_bias = False)
X_4_poly_train = poly_feature_4.fit_transform(X_train)
X_4_poly_test = poly_feature_4.fit_transform(X_test)
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_4_poly_train, y_train)


# In[21]:


y_pred_poly_4_train = poly_4_reg.predict(X_4_poly_train)
y_pred_poly_4_test = poly_4_reg.predict(X_4_poly_test)


# In[22]:


poly_4_train_mse = mean_squared_error(y_train, y_pred_poly_4_train)
poly_4_test_mse = mean_squared_error(y_test, y_pred_poly_4_test)


# In[23]:


poly_feature_5 = PolynomialFeatures(degree = 5, include_bias = False)
X_5_poly_train = poly_feature_5.fit_transform(X_train)
X_5_poly_test = poly_feature_5.fit_transform(X_test)
poly_5_reg = LinearRegression()
poly_5_reg.fit(X_5_poly_train, y_train)


# In[24]:


y_pred_poly_5_train = poly_5_reg.predict(X_5_poly_train)
y_pred_poly_5_test = poly_5_reg.predict(X_5_poly_test)


# In[25]:


poly_5_train_mse = mean_squared_error(y_train, y_pred_poly_5_train)
poly_5_test_mse = mean_squared_error(y_test, y_pred_poly_5_test)


# In[26]:


mse = [ [lin_reg_train_mse, lin_reg_test_mse], [knn_3_train_mse, knn_3_test_mse], [knn_5_train_mse, knn_5_test_mse], [poly_2_train_mse, poly_2_test_mse], [poly_3_train_mse, poly_3_test_mse], [poly_4_train_mse, poly_4_test_mse],[poly_5_train_mse, poly_5_test_mse]]


# In[27]:


df_mse = pd.DataFrame(mse, index=["lin_reg", "knn_3_reg", "knn_5_reg", "poly_2_reg", "poly_3_reg", "poly_4_reg", "poly_5_reg"], columns = ["train_mse", "test_mse"])
df_mse


# In[28]:


with open('mse.pkl', 'wb') as f:
    pickle.dump(df_mse, f)


# In[29]:


reg = [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), (poly_2_reg, poly_feature_2), (poly_3_reg, poly_feature_3), (poly_4_reg, poly_feature_4), (poly_5_reg, poly_feature_5)]
reg


# In[30]:


with open('reg.pkl', 'wb') as f:
    pickle.dump(reg, f)





