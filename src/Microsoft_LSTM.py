import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.metrics import mean_absolute_error

from utils import microsoft_ws_data, microsoft_ml_data , rmse
from model import setup_lstm_ae_model

import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Activation, Dropout

## Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
## Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
## that you would like to run the code.

# define path to save model
model_path = '../Trained models/Microsoft_regression_LSTM.h5'

# import data
# train_X, train_y, test_X, test_y , scaler = microsoft_ws_data()

train_X, train_y, test_X, test_y , scaler = microsoft_ml_data()
column_name = train_X.columns
train_X = train_X.values.reshape(-1, 1, 27)
test_X  = test_X.values.reshape(-1, 1, 27)


model = setup_lstm_ae_model(train_X)
print(model.summary())

# fit network
history = model.fit(train_X, train_X,  epochs=200, batch_size=1024, validation_data=(test_X, test_X), verbose=2,
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto'), 
                                tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('../plot/microsoft_loss.png')


# make a prediction
Xhat = model.predict(test_X)
Xhat = Xhat.reshape((-1, 27))
test_X = test_X.reshape((-1, 27))

# invert scaling for forecast
inv_Xhat = Xhat.copy()
for loc_,col_ in enumerate(column_name):
    inv_Xhat[:,loc_:loc_+1] = scaler[col_].inverse_transform(Xhat[:,loc_:loc_+1])

# invert scaling for actual
inv_X = test_X.copy()
for loc_,col_ in enumerate(column_name):
    inv_X[:,loc_:loc_+1] = scaler[col_].inverse_transform(test_X[:,loc_:loc_+1])

# calculate RMSE
mae = mean_absolute_error(inv_X, inv_Xhat)
print('Test MAE: %.3f' % mae)


## time steps, every step is one hour (you can easily convert the time step to the actual time index)
## for a demonstration purpose, I only compare the predictions in 200 hours.
train_scored = pd.DataFrame()
train_scored['Loss_mae'] = np.mean(np.abs(inv_Xhat-inv_X), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(train_scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.savefig('../plot/microsoft_loss_mae.png')