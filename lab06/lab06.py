
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[2]:


X = data_breast_cancer.data
y = data_breast_cancer.target


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


X_train_tex_sym = X_train[["mean texture", "mean symmetry"]]
X_test_tex_sym = X_test[["mean texture", "mean symmetry"]]


# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.neighbors
tree_clf = DecisionTreeClassifier(max_depth=3)
log_reg = LogisticRegression(solver="lbfgs")
knn_clf = sklearn.neighbors.KNeighborsClassifier()


# In[6]:


from sklearn.ensemble import VotingClassifier
vot_clf_hard = VotingClassifier(estimators=[('tr',tree_clf),('log',log_reg),('knn',knn_clf)], voting='hard')
vot_clf_soft = VotingClassifier(estimators=[('tr',tree_clf),('log',log_reg),('knn',knn_clf)], voting='soft')


# In[7]:


vot_clf_hard.fit(X_train_tex_sym,y_train)
vot_clf_soft.fit(X_train_tex_sym,y_train)
tree_clf.fit(X_train_tex_sym,y_train)
log_reg.fit(X_train_tex_sym,y_train)
knn_clf.fit(X_train_tex_sym,y_train)


# In[8]:


from sklearn.metrics import accuracy_score
acc_train_tree_clf = accuracy_score(y_train, tree_clf.predict(X_train_tex_sym))
acc_train_log_reg = accuracy_score(y_train, log_reg.predict(X_train_tex_sym))
acc_train_knn_clf = accuracy_score(y_train, knn_clf.predict(X_train_tex_sym))
acc_train_vot_clf_hard = accuracy_score(y_train, vot_clf_hard.predict(X_train_tex_sym))
acc_train_vot_clf_soft = accuracy_score(y_train, vot_clf_soft.predict(X_train_tex_sym))


# In[9]:


acc_test_tree_clf = accuracy_score(y_test, tree_clf.predict(X_test_tex_sym))
acc_test_log_reg = accuracy_score(y_test, log_reg.predict(X_test_tex_sym))
acc_test_knn_clf = accuracy_score(y_test, knn_clf.predict(X_test_tex_sym))
acc_test_vot_clf_hard = accuracy_score(y_test, vot_clf_hard.predict(X_test_tex_sym))
acc_test_vot_clf_soft = accuracy_score(y_test, vot_clf_soft.predict(X_test_tex_sym))


# In[10]:


acc_vote = [(acc_train_tree_clf, acc_test_tree_clf),(acc_train_log_reg, acc_test_log_reg),(acc_train_knn_clf, acc_test_knn_clf),(acc_train_vot_clf_hard,acc_test_vot_clf_hard),(acc_train_vot_clf_soft,acc_test_vot_clf_soft)]


# In[11]:


print(acc_vote)


# In[12]:


import pickle


# In[13]:


with open('acc_vote.pkl', 'wb') as file:
    pickle.dump(acc_vote, file)


# In[14]:


clf = [tree_clf, log_reg, knn_clf, vot_clf_hard, vot_clf_soft]


# In[15]:


print(clf)


# In[16]:


with open('vote.pkl', 'wb') as file:
    pickle.dump(clf, file)


# In[17]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30)
bag_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5)
pas_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False)
pas_clf_half = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=False)
rnd_clf = RandomForestClassifier(n_estimators=30)
ada_clf = AdaBoostClassifier(n_estimators=30)
gb_clf = GradientBoostingClassifier(n_estimators=30)


# In[18]:


bag_clf.fit(X_train_tex_sym, y_train)
bag_clf_half.fit(X_train_tex_sym, y_train)
pas_clf.fit(X_train_tex_sym, y_train)
pas_clf_half.fit(X_train_tex_sym, y_train)
rnd_clf.fit(X_train_tex_sym, y_train)
ada_clf.fit(X_train_tex_sym, y_train)
gb_clf.fit(X_train_tex_sym, y_train)


# In[19]:


acc_bag_train = accuracy_score(y_train, bag_clf.predict(X_train_tex_sym))
acc_bagh_train = accuracy_score(y_train, bag_clf_half.predict(X_train_tex_sym))
acc_pas_train = accuracy_score(y_train, pas_clf.predict(X_train_tex_sym))
acc_pash_train = accuracy_score(y_train, pas_clf_half.predict(X_train_tex_sym))
acc_rnd_train = accuracy_score(y_train, rnd_clf.predict(X_train_tex_sym))
acc_ada_train = accuracy_score(y_train, ada_clf.predict(X_train_tex_sym))
acc_gb_train = accuracy_score(y_train, gb_clf.predict(X_train_tex_sym))


# In[20]:


acc_bag_test = accuracy_score(y_test, bag_clf.predict(X_test_tex_sym))
acc_bagh_test = accuracy_score(y_test, bag_clf_half.predict(X_test_tex_sym))
acc_pas_test = accuracy_score(y_test, pas_clf.predict(X_test_tex_sym))
acc_pash_test = accuracy_score(y_test, pas_clf_half.predict(X_test_tex_sym))
acc_rnd_test = accuracy_score(y_test, rnd_clf.predict(X_test_tex_sym))
acc_ada_test = accuracy_score(y_test, ada_clf.predict(X_test_tex_sym))
acc_gb_test = accuracy_score(y_test, gb_clf.predict(X_test_tex_sym))


# In[21]:


acc_bag = [(acc_bag_train, acc_bag_test),(acc_bagh_train, acc_bagh_test),(acc_pas_train, acc_pas_test),(acc_pash_train, acc_pash_test),(acc_rnd_train, acc_rnd_test),(acc_ada_train, acc_ada_test),(acc_gb_train, acc_gb_test)]


# In[22]:


print(acc_bag)


# In[23]:


with open('acc_bag.pkl', 'wb') as file:
    pickle.dump(acc_bag, file)


# In[24]:


bag = [bag_clf, bag_clf_half, pas_clf, pas_clf_half, rnd_clf, ada_clf, gb_clf]


# In[25]:


print(bag)


# In[26]:


with open('bag.pkl', 'wb') as file:
    pickle.dump(bag, file)


# In[27]:


sampling = BaggingClassifier(n_estimators=30, bootstrap=True, bootstrap_features=False, max_features=2,max_samples=0.5)


# In[28]:


sampling.fit(X_train, y_train)
acc_samp_train = accuracy_score(y_train, sampling.predict(X_train))
acc_samp_test = accuracy_score(y_test, sampling.predict(X_test))


# In[30]:


acc_fea = [acc_samp_train, acc_samp_test]
print(acc_fea)


# In[31]:


with open('acc_fea.pkl', 'wb') as file:
    pickle.dump(acc_fea, file)


# In[32]:


fea = [sampling]
print(fea)


# In[33]:


with open('fea.pkl', 'wb') as file:
    pickle.dump(fea, file)




