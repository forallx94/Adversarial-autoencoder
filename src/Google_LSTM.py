import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#matplotlib inline

from utils import google_data , rmse
from model import setup_lstm_ae_model


# define path to save model
model_path = '../Trained models/Google_regression_LSTM.h5'

# import data
train_X, train_y, test_X, test_y , scaler = google_data()

model = setup_lstm_ae_model(train_X)
print(model.summary())

# fit the network
history = model.fit(train_X, train_X, epochs=200, batch_size=32, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_mse', patience=2, verbose=0, mode='auto'), 
                        keras.callbacks.ModelCheckpoint(model_path,monitor='val_mse', save_best_only=True, mode='min', verbose=0)]
          )

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# # make a prediction
# yhat = model.predict(test_X, verbose=1, batch_size=200)

# test_X = test_X[:,-1,:1]
# # invert scaling for forecast
# inv_yhat = np.concatenate((test_X, yhat), axis=1)
# inv_yhat = scaler[1].inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,1]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_X, test_y), axis=1)
# inv_y = scaler[1].inverse_transform(inv_y)
# inv_y = inv_y[:,1]

# # calculate RMSE
# rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)

# # print("Prediction")
# # print(predicted_value);
# # print("Truth")
# # print(input_data[lookback:test_size + (2 * lookback), 1]);
# plt.plot(yhat, color= 'red')
# plt.plot(test_y, color='green')
# plt.title("Opening price of stocks sold")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Stock Opening Price")
# plt.show()