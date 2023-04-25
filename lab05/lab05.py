
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[18]:


import numpy as np 
import pandas as pd 


# In[3]:


from sklearn.tree import DecisionTreeClassifier


# In[4]:


X=data_breast_cancer.data
y=data_breast_cancer.target


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)


# In[8]:


tree_clf.fit(X_train,y_train)


# In[9]:


from sklearn.metrics import f1_score


# In[10]:


y_pred_train = tree_clf.predict(X_train)
y_pred_test = tree_clf.predict(X_test)
f1_train_val = f1_score(y_train, y_pred_train)
f1_test_val = f1_score(y_test, y_pred_test)
acc_test = tree_clf.score(X_train, y_train)
acc_train = tree_clf.score(X_test, y_test)
depth = 3


# In[11]:


f1acc_tree = [depth, f1_train_val, f1_test_val, acc_test, acc_train] 


# In[12]:


import pickle


# In[13]:


with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(f1acc_tree, f)


# In[15]:


from sklearn.tree import export_graphviz
import graphviz
f = "bc"
export_graphviz(tree_clf, out_file=f, feature_names=data_breast_cancer.feature_names,
                class_names=data_breast_cancer.target_names, rounded=True, filled=True)
graphviz.render('dot', 'png', f)


# In[17]:


from sklearn.tree import DecisionTreeRegressor


# In[19]:


size = 300 
X=np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0=1,2,1,-4,2
y=w4*(X**4)+w3*(X**3)+w2*(X**2)+w1*X+w0+np.random.randn(size)*8-4
df=pd.DataFrame({'x': X,'y': y})
df.plot.scatter(x='x',y='y')


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train.reshape(-1,1)
X_test  = X_test.reshape(-1,1)


# In[34]:


tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)


# In[35]:


from sklearn.metrics import mean_squared_error


# In[36]:


tree_reg.fit(X_train, y_train)


# In[37]:


y_train_pred = tree_reg.predict(X_train)
mse_train = mean_squared_error(y_train_pred, y_train)


# In[38]:


y_test_pred = tree_reg.predict(X_test)
mse_test = mean_squared_error(y_test_pred, y_test)


# In[39]:


f_d = "reg"
export_graphviz(tree_reg, out_file=f_d, rounded=True, filled=True)
graphviz.render('dot','png', f_d)


# In[40]:


mse_tree = [tree_reg.max_depth, mse_train, mse_test]
print(mse_tree)
with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(mse_tree, f)

