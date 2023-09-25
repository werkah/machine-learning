#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.keras.utils.get_file(
"bike_sharing_dataset.zip",
"https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
cache_dir=".",
extract=True
)


# In[2]:


#pip install --upgrade pandas


# In[3]:


import pandas as pd
df = pd.read_csv('datasets/hour.csv',
                 parse_dates={'datetime': ['dteday', 'hr']},
                 date_format='%Y-%m-%d %H',
                 index_col='datetime')


# In[4]:


print((df.index.min(), df.index.max()))


# In[5]:


(365 + 366) * 24 - len(df)


# In[6]:


df = df.resample('H').asfreq()


# In[7]:


df['casual'].fillna(0, inplace=True)
df['registered'].fillna(0, inplace=True)
df['cnt'].fillna(0, inplace=True)


# In[8]:


df['temp'].interpolate(method='linear', inplace=True)
df['atemp'].interpolate(method='linear', inplace=True)
df['hum'].interpolate(method='linear', inplace=True)
df['windspeed'].interpolate(method='linear', inplace=True)


# In[9]:


df['holiday'].fillna(method='ffill', inplace=True)
df['weekday'].fillna(method='ffill', inplace=True)
df['workingday'].fillna(method='ffill', inplace=True)
df['weathersit'].fillna(method='ffill', inplace=True)


# In[10]:


df.notna().sum()


# In[11]:


df[['casual', 'registered', 'cnt', 'weathersit']].describe()


# In[12]:


df.casual /= 1e3
df.registered /= 1e3
df.cnt /= 1e3
df.weathersit /= 4


# In[13]:


df_2weeks = df[:24 * 7 * 2]
df_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[14]:


df_daily = df.resample('W').mean()
df_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[15]:


previous_day_values = df['cnt'].shift(24) * 1000  
mae_daily = abs(df['cnt'] * 1000 - previous_day_values).mean()
previous_week_values = df['cnt'].shift(24 * 7) * 1000  
mae_weekly = abs(df['cnt'] * 1000 - previous_week_values).mean() 


# In[16]:


mae_daily /= 1e3
mae_weekly /= 1e3


# In[17]:


import pickle


# In[18]:


mae_baseline = (mae_daily, mae_weekly)
with open('mae_baseline.pkl', 'wb') as file:
    pickle.dump(mae_baseline, file)


# In[19]:


cnt_train = df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df['cnt']['2012-07-01 00:00':]


# In[20]:


seq_len = 1 * 24
train_ds = tf.keras.utils.timeseries_dataset_from_array(
cnt_train.to_numpy(),
targets=cnt_train[seq_len:],
sequence_length=seq_len,
batch_size=32,
shuffle=True,
seed=42)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
cnt_valid.to_numpy(),
targets=cnt_valid[seq_len:],
sequence_length=seq_len,
batch_size=32)


# In[21]:


model = tf.keras.Sequential([
tf.keras.layers.Dense(1, input_shape=[seq_len])])


# In[22]:


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)


# In[23]:


model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])


# In[24]:


history = model.fit(train_ds, epochs=20, validation_data=valid_ds)


# In[25]:


model.save('model_linear.h5')


# In[26]:


mae_linear = model.evaluate(valid_ds)[1]
mae_linear_tuple = (mae_linear,)
with open('mae_linear.pkl', 'wb') as file:
    pickle.dump(mae_linear_tuple, file)


# In[27]:


model_rnn1 = tf.keras.Sequential([
tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])


# In[28]:


model_rnn1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                   loss=tf.keras.losses.Huber(),
                   metrics=['mae'])


# In[29]:


history_rnn1 = model_rnn1.fit(train_ds, epochs=20, validation_data=valid_ds)


# In[30]:


model_rnn1.save('model_rnn1.h5')


# In[31]:


mae_rnn1 = model_rnn1.evaluate(valid_ds)[1]


# In[32]:


mae_rnn1_tuple = (mae_rnn1,)
with open('mae_rnn1.pkl', 'wb') as file:
    pickle.dump(mae_rnn1_tuple, file)


# In[33]:


model_rnn32 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1)])


# In[34]:


model_rnn32.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                    loss=tf.keras.losses.Huber(),
                    metrics=['mae'])


# In[35]:


history_rnn32 = model_rnn32.fit(train_ds, epochs=20, validation_data=valid_ds)


# In[36]:


model_rnn32.save('model_rnn32.h5')


# In[37]:


mae_rnn32 = model_rnn32.evaluate(valid_ds)[1]


# In[38]:


mae_rnn32_tuple = (mae_rnn32,)
with open('mae_rnn32.pkl', 'wb') as file:
    pickle.dump(mae_rnn32_tuple, file)


# In[39]:


model_deep = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1)])


# In[40]:


model_deep.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='huber', metrics=['mae'])
history = model_deep.fit(train_ds, epochs=20, validation_data=valid_ds)


# In[41]:


model_deep.save('model_rnn_deep.h5')


# In[42]:


mae_rnn_deep = model_deep.evaluate(valid_ds)[1]


# In[43]:


mae_deep_tuple = (mae_rnn_deep,)
with open('mae_rnn_deep.pkl', 'wb') as file:
    pickle.dump(mae_deep_tuple, file)


# In[49]:


col = ['cnt', 'weathersit', 'atemp', 'workingday']


# In[50]:


train_sub = df['2011-01-01 00:00':'2012-06-30 23:00']
valid_sub = df['2012-07-01 00:00':]


# In[51]:


seq_len = 1 * 24


# In[52]:


train_ds = tf.keras.utils.timeseries_dataset_from_array(
    train_sub[col].to_numpy(),
    targets=train_sub['cnt'][seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    valid_sub[col].to_numpy(),
    targets=valid_sub['cnt'][seq_len:],
    sequence_length=seq_len,
    batch_size=32)


# In[59]:


model_mv = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[seq_len, len(col)]),
    tf.keras.layers.Dense(1)])


# In[60]:


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)


# In[61]:


model_mv.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])


# In[62]:


history_mv = model_mv.fit(train_ds, epochs=20, validation_data=valid_ds)


# In[68]:


model_mv.save("model_rnn_mv.h5")


# In[69]:


mae_rnn_mv = model_mv.evaluate(valid_ds)[1]


# In[70]:


mv_tp = (mae_rnn_mv,)
with open("mae_rnn_mv.pkl", "wb") as file:
    pickle.dump((mv_tp,), file)


