#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
fashion_mnist=tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test)=fashion_mnist.load_data()
assert X_train.shape==(60000,28,28)
assert X_test.shape==(10000,28,28)
assert y_train.shape==(60000,)
assert y_test.shape==(10000,)


# In[2]:


X_train = X_train/255
X_test = X_test/255


# In[3]:


import matplotlib.pyplot as plt
plt.imshow(X_train[142], cmap="binary")
plt.axis('off') 
plt.show()


# In[4]:


class_names=["koszulka","spodnie","pulower","sukienka","kurtka","sandał","koszula","but","torba","kozak"]
class_names[y_train[142]]


# In[5]:


from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))


# In[6]:


model.summary()
tf.keras.utils.plot_model(model,"fashion_mnist.png", show_shapes=True)


# In[7]:


model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),metrics='accuracy')


# In[8]:


import datetime
import os
log_dir = os.path.join(os.getcwd(), "image_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# In[9]:


history = model.fit(X_train, y_train, epochs=20, validation_split = 0.1, callbacks=[tensorboard_callback])


# In[10]:


import numpy as np
image_index=np.random.randint(len(X_test))
image=np.array([X_test[image_index]])
confidences=model.predict(image)
confidence=np.max(confidences[0])
prediction=np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[11]:


model.save("fashion_clf.h5")


# In[12]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
housing=fetch_california_housing()


# In[13]:


X_train_all, X_test, y_train_all, y_test = train_test_split(housing.data,housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_all,y_train_all, random_state=42)


# In[14]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)


# In[15]:


model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]))
model2.add(keras.layers.Dense(1))
model2.compile(loss="mean_squared_error", optimizer='sgd')


# In[16]:


early_stop = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)


# In[17]:


log_dir = os.path.join(os.getcwd(), "housing_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# In[19]:


history1 = model2.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[early_stop, tensorboard_cb])


# In[20]:


model2.save("reg_housing_1.h5")


# In[21]:


model3 = keras.models.Sequential()
model3.add(keras.layers.Dense(40, activation="relu", input_shape=X_train.shape[1:]))
model3.add(keras.layers.Dense(1))
model3.compile(loss="mean_squared_error", optimizer='sgd')


# In[22]:


tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# In[23]:


history2 = model3.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[early_stop, tensorboard_cb])


# In[24]:


model3.save("reg_housing_2.h5")


# In[25]:


model4 = keras.models.Sequential()
model4.add(keras.layers.Dense(40, activation="softmax", input_shape=X_train.shape[1:]))
model4.add(keras.layers.Dense(1))
model4.compile(loss="mean_squared_error", optimizer='sgd')


# In[26]:


tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# In[27]:


history3 = model4.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[early_stop, tensorboard_cb])


# In[28]:


model4.save("reg_housing_3.h5")
