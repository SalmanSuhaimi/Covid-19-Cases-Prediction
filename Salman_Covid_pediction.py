#%%

# 1. Setup
import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from time_series_helper import WindowGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications, models

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
 
 
# %%
# 2.Load dataset with pandas
df = pd.read_csv('cases_malaysia_covid.csv')
selected_columns = ['date', 'cases_new', 'cases_import', 'cases_recovered', 'cases_active']
df = df[selected_columns]

date_time = pd.to_datetime(df.pop('date'), format='%d/%m/%Y')

#%%
df.info()

#%%
# Convert dtypes new_cases from obj to int
# Assuming 'cases_new' contains strings representing numbers with possible commas
df['cases_new'] = df['cases_new'].replace({',': ''}, regex=True)

# Convert the 'cases_new' column to integer
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce', downcast='integer')

# Print the data types to verify the changes
print(df.dtypes)

# %%
df.describe().T

# %%
# 3.Basic data inspection
df.set_index(date_time, inplace=True)
plot_cols = ['cases_new', 'cases_import', 'cases_recovered', 'cases_active']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)
 
plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)

# %%
#4. Data cleaning
print(df.isnull().sum()) #cases_new have null value

#%%
# Box plot
df.boxplot(figsize=(12, 8))
plt.show()

#%%
df.hist(figsize=(20,20), edgecolor='black')
plt.show()

#%% 
# Null value use median to filled
df['cases_new'].fillna(df['cases_new'].median(), inplace=True)
print(df.isnull().sum()) # double check, is it have any null value

#%%
print(df.duplicated().sum()) # total duplicated = 10
print(df.shape)
#%%
# Duplicate row
df.drop_duplicates(inplace=True)
# Double check duplicate
print(df.duplicated().sum())
print(df.shape)
# %%
# Train, validation, test split for tme series data
column_indices = {name: i for i, name in enumerate(df.columns)}
 
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
 
num_features = df.shape[1]

# %%
# Data normalization
train_mean = train_df.mean()
train_std = train_df.std()
 
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# %%
# 8.Data inspection after normalization
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

# %%
one_predict = WindowGenerator(input_width=25, label_width=25, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])
#%%
#Plot window
one_predict.plot('cases_new')

#%%
# Create a TensorBoard callback for single-step
log_dir_single = "logs_single/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_single = tf.keras.callbacks.TensorBoard(log_dir=log_dir_single, histogram_freq=1)

# Create a TensorBoard callback for multi-step
log_dir_multi = "logs_multi/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_multi = tf.keras.callbacks.TensorBoard(log_dir=log_dir_multi, histogram_freq=1)

# %%
import tensorflow as tf
from tensorflow import keras
 
lstm_model = keras.Sequential()
lstm_model.add(keras.layers.LSTM(128, return_sequences=True))
lstm_model.add(keras.layers.Dropout(0.2))
lstm_model.add(keras.layers.Dense(1))

# %%
# Function to perform model compile and training
def mape(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])  # Flatten the true values
    y_pred = tf.reshape(y_pred, [-1])  # Flatten the predicted values
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))

MAX_EPOCHS = 40
 
def compile_and_fit(model, window, patience=3, callbacks_list=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )
 
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(), mape]
    )
 
    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping] + (callbacks_list or [])
    )
  
# %%
# Compile the model and train
history_1 = compile_and_fit(lstm_model, one_predict, callbacks_list=[tensorboard_callback_single])

# %%
# Evaluate the model

print(lstm_model.evaluate(one_predict.val))
print(lstm_model.evaluate(one_predict.test))

# %%
# Plot the resultt
one_predict.plot(model=lstm_model, plot_col='cases_new')

#%%
#Display model summary
lstm_model.summary()
#Display model structure
tf.keras.utils.plot_model(lstm_model)
# %%
# Scenario2 : Multi step window
#OUT_STEPS = 24
#multi_window = WindowGenerator(input_width=24,
#                               label_width=OUT_STEPS,
#                               shift=OUT_STEPS)
 
#multi_window.plot()
#multi_window
multi_predict = WindowGenerator(input_width=25, label_width=25, shift=25, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])

#%%
#Plot window
multi_predict.plot(plot_col='cases_new')

# %%
# Build multi step model
multi_lstm = keras.Sequential()
multi_lstm.add(keras.layers.LSTM(256, return_sequences=False))
multi_lstm.add(keras.layers.Dense(25*1))
multi_lstm.add(keras.layers.Reshape([25,1]))

# %%
# Compile and train model for multi step
history_2 = compile_and_fit(multi_lstm, multi_predict, callbacks_list=[tensorboard_callback_multi])

# %%
# Evaluate the model
print(multi_lstm.evaluate(multi_predict.val))
print(multi_lstm.evaluate(multi_predict.test))

# %%
# Plot the resultt
multi_predict.plot(model=lstm_model, plot_col='cases_new')

# %%
#%%
#Display model summary
multi_lstm.summary()
#Display model structure
tf.keras.utils.plot_model(multi_lstm)
# %%
