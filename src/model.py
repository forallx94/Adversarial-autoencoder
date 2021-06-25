from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, InputLayer, Dropout, RepeatVector, TimeDistributed, Dense

from utils import rmse


def setup_lstm_ae_model(train_X):
	hidden_unit = (train_X.shape[1] * train_X.shape[2]) // 2

	model = Sequential()
	model.add(LSTM(
			input_shape=(train_X.shape[1], train_X.shape[2]),
			units=hidden_unit,
			return_sequences=False))
	model.add(Dropout(0.2))
	model.add(RepeatVector(train_X.shape[1]))
	model.add(LSTM(
			units=train_X.shape[2],
			return_sequences=True))
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	return model


def setup_lstm_ae_model_2(train_X):
	hidden_unit = (train_X.shape[1] * train_X.shape[2]) // 2

	model=Sequential([
		LSTM(hidden_unit, activation='relu', input_shape=(train_X.shape[1],train_X.shape[2])),
		Dropout(0.2),
		RepeatVector(train_X.shape[1]),
		LSTM(hidden_unit, activation='relu', return_sequences=True),
		Dropout(0.2),
		TimeDistributed(Dense(train_X.shape[2]))
	])

	model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
	return model