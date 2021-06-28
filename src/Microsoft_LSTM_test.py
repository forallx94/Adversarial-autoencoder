import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, f1_score

from utils import microsoft_ml_data, microsoft_ws_data, fgsm, rmse, bim

## for Deep-learing:
import tensorflow.keras.backend as K
# from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.models import load_model

## Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
## Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
## that you would like to run the code.

model_path = '../Trained models/Microsoft_regression_LSTM.h5'


train_X, train_y, test_X, test_y , scaler = microsoft_ml_data()
column_name = train_X.columns
train_X = train_X.values.reshape(-1, 1, 27)
test_X  = test_X.values.reshape(-1, 1, 27)


def compare_result(x,xhat,name=''):
	scored_train = pd.DataFrame()
	scored_train['Loss_mae'] = np.mean(np.abs(xhat-x), axis = 1)
	scored_train['Threshold'] = TH
	scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
	anomalies = scored_train[scored_train['Anomaly'] == True]
	scored_train['True'] = test_y != 4

	f, (ax1) = plt.subplots(figsize=(16, 4))
	ax1.plot(scored_train.index, scored_train.Loss_mae, label='Loss(MAE)');
	ax1.plot(scored_train.index, scored_train.Threshold, label='Threshold')
	g = sns.scatterplot(x=anomalies.index , y=anomalies.Loss_mae, label='anomaly', color='red')
	g.set(xlim = (0, len(scored_train.index)))
	plt.title('Anomalies')
	plt.xlabel('Data points')
	plt.ylabel('Loss (MAE)')
	plt.legend()
	plt.savefig(f'../plot/microsoft_{name}_mae.png')

	scor_ = accuracy_score(scored_train['True'], scored_train['Anomaly'])
	f1_ = f1_score(scored_train['True'], scored_train['Anomaly'])

	print(f'acc {name} : {scor_} \n f1 {name} : {f1_}')
	print(confusion_matrix(scored_train['True'], scored_train['Anomaly']))

if os.path.isfile(model_path):
	model = load_model(model_path, custom_objects={'rmse': rmse})

	# make adversarial example
	adv_X_fgsm, _ = fgsm(X =test_X, Y=test_X, model=model ,loss_fn = rmse , epsilon=0.2)

	# make bim adversarial example
	adv_X_bim, _ = bim(X =test_X, Y=test_X, model=model ,loss_fn = rmse , epsilon=0.2, alpha=0.001, I=200)

	# make a adv prediction
	adv_Xhat_fgsm = model.predict(adv_X_fgsm)
	adv_Xhat_fgsm = adv_Xhat_fgsm.reshape((test_X.shape[0], test_X.shape[2]))

	# make a adv prediction
	adv_Xhat_bim = model.predict(adv_X_bim)
	adv_Xhat_bim = adv_Xhat_bim.reshape((test_X.shape[0], test_X.shape[2]))

	# make a prediction
	Xhat = model.predict(test_X)
	Xhat = Xhat.reshape((Xhat.shape[0], Xhat.shape[2]))
	test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
	adv_X_fgsm = adv_X_fgsm.reshape((adv_X_fgsm.shape[0], adv_X_fgsm.shape[2]))
	adv_X_bim = adv_X_bim.reshape((adv_X_bim.shape[0], adv_X_bim.shape[2]))

	# invert scaling for forecast
	inv_X = test_X.copy()
	inv_Xhat = Xhat.copy()
	inv_adv_X_fgsm = adv_X_fgsm.copy()
	inv_adv_X_bim = adv_X_bim.copy()
	inv_adv_Xhat_fgsm = adv_Xhat_fgsm.copy()
	inv_adv_Xhat_bim = adv_Xhat_bim.copy()
	for loc_,col_ in enumerate(column_name):
		inv_X[:,loc_:loc_+1] = scaler[col_].inverse_transform(test_X[:,loc_:loc_+1])
		inv_Xhat[:,loc_:loc_+1] = scaler[col_].inverse_transform(Xhat[:,loc_:loc_+1])
		inv_adv_X_fgsm[:,loc_:loc_+1] = scaler[col_].inverse_transform(adv_X_fgsm[:,loc_:loc_+1])
		inv_adv_X_bim[:,loc_:loc_+1] =scaler[col_].inverse_transform(adv_X_bim[:,loc_:loc_+1])
		inv_adv_Xhat_fgsm[:,loc_:loc_+1] = scaler[col_].inverse_transform(adv_Xhat_fgsm[:,loc_:loc_+1])
		inv_adv_Xhat_bim[:,loc_:loc_+1] = scaler[col_].inverse_transform(adv_Xhat_bim[:,loc_:loc_+1])

	mae = mean_absolute_error(inv_X, inv_Xhat)
	print('Test MAE: %.3f' % mae)
	mae = mean_absolute_error(inv_adv_X_fgsm, inv_adv_Xhat_fgsm)
	print('Test adv fgsm MAE: %.3f' % mae)
	mae = mean_absolute_error(inv_adv_X_bim, inv_adv_Xhat_bim)
	print('Test adv bim MAE: %.3f' % mae)

	TH = 10.0

	compare_result(inv_X,inv_Xhat,name='')
	compare_result(inv_adv_X_fgsm,inv_adv_Xhat_fgsm,name='fgsm')
	compare_result(inv_adv_X_bim,inv_adv_Xhat_bim,name='bim')