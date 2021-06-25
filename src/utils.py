import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def power_data():
    df = pd.read_csv('../Dataset/household_power_consumption.txt', sep=';',
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')

    ## finding all columns that have nan:
    droping_list_all=[]
    for j in range(0,7):
        if not df.iloc[:, j].notnull().all():
            droping_list_all.append(j)
            #print(df.iloc[:,j].unique())

    # filling nan with mean in any columns
    for j in range(0,7):
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        dff = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(dff.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(dff.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    ## resampling of data over days
    df_resample = df.resample('h').mean()

    ## * Note: I scale all features in range of [0,1].

    ## If you would like to train based on the resampled data (over hour), then used below
    values = df_resample.values


    ## full data without resampling
    #values = df.values

    # integer encode direction
    # ensure all data is float
    #values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)

    # split into train and test sets
    values = reframed.values

    n_train_time = 365*72
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]


    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

    return train_X, train_y, test_X, test_y , scaler


def window_slide(input_data, window_size):
    X = []
    y = []
    for i in range(len(input_data) - window_size - 1):
        t = []
        for j in range(0, window_size):
            t.append(input_data[[(i + j)], :])
        X.append(t)
        y.append(input_data[i + window_size, 1])
    return np.array(X), np.array(y)


def google_data(window_size=60):
    # import data
    stock_train = pd.read_csv("../Dataset/Google dataset/Google_train.csv",dtype={'Close': 'float64', 'Volume': 'int64','Open': 'float64','High': 'float64', 'Low': 'float64'})
    stock_test = pd.read_csv("../Dataset/Google dataset/Google_train.csv",dtype={'Close': 'float64', 'Volume': 'int64','Open': 'float64','High': 'float64', 'Low': 'float64'})

    stock_train.columns = ['date', 'close/last', 'volume', 'open', 'high', 'low']
    stock_test.columns = ['date', 'close/last', 'volume', 'open', 'high', 'low']

    #create a new column "average" 
    stock_train['average'] = (stock_train['high'] + stock_train['low'])/2
    stock_test['average'] = (stock_test['high'] + stock_test['low'])/2

    #pick the input features (average and volume)
    train_feature = stock_train.iloc[:,[2,6]].values
    train_data = train_feature

    test_feature= stock_test.iloc[:,[2,6]].values
    test_data = test_feature

    #data normalization
    sc= MinMaxScaler(feature_range=(0,1))
    train_data[:,0:2] = sc.fit_transform(train_feature[:,:])

    cs= MinMaxScaler(feature_range=(0,1))
    test_data[:,0:2] = cs.fit_transform(test_feature[:,:])

    scaler = [sc, cs]

    train_X, train_y = window_slide(train_data, window_size)
    test_X, test_y = window_slide(test_data, window_size)

    train_X = train_X.reshape(train_X.shape[0], window_size, 2)
    test_X = test_X.reshape(test_X.shape[0],window_size, 2)

    return train_X, train_y, test_X, test_y , scaler


def microsoft_data(window_size=50):
    telemetry = pd.read_csv('../Dataset/Microsoft Azure/PdM_telemetry.csv')
    errors = pd.read_csv('../Dataset/Microsoft Azure/PdM_errors.csv')
    maint = pd.read_csv('../Dataset/Microsoft Azure//PdM_maint.csv')
    failures = pd.read_csv('../Dataset/Microsoft Azure//PdM_failures.csv')
    machines = pd.read_csv('../Dataset/Microsoft Azure//PdM_machines.csv')

    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

    errors['datetime'] = pd.to_datetime(errors['datetime'],format = '%Y-%m-%d %H:%M:%S')
    errors['errorID'] = errors['errorID'].astype('category')

    maint['datetime'] = pd.to_datetime(maint['datetime'], format='%Y-%m-%d %H:%M:%S')
    maint['comp'] = maint['comp'].astype('category')

    machines['model'] = machines['model'].astype('category')

    failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
    failures['failure'] = failures['failure'].astype('category')

    # create a column for each error type
    # error 별 column 생성
    error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
    error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
    # 같은 시간으로 통합
    error_count = error_count.groupby(['machineID','datetime']).sum().reset_index()
    # telemetry 에 통합
    error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)
    error_count = error_count.dropna()

    # create a column for each error type
    # 각 부품별로 생성후 column 명 설정
    comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
    comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

    # combine repairs for a given machine in a given hour
    # 기계와 시간별로 교체 부품 정리
    comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

    # add timepoints where no components were replaced
    # telemetry 데이터에 데이터 결합
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                        on=['datetime', 'machineID'],
                                                        how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])

    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        # convert indicator to most recent date of component change
        comp_rep.loc[comp_rep[comp] < 1, comp] = None # 0.0인 위치를 모두 None 으로 변경
        comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
        # -comp_rep[comp].isnull() comp_req 의 comp columns 에서 None 이 아닌 부분을 선택(즉 해당 부품의 교체가 발생된 날짜 선택)
        # 해당 부분의 내용을 시간으로 변경
        
        # forward-fill the most-recent date of component change
        # 나머지 빈 부분을 모두 해당 시간으로 교체 (즉 부품 교체일로 변경 날짜)
        comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

    # remove dates in 2014 (may have NaN or future component change dates)    
    # 2014년 이상의 데이터만 선택
    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]

    # replace dates of most recent component change with days since most recent component change
    # 각 comp 내용을 부품 교체일 부터 얼마나 지났는지로 변경
    for comp in components:
        comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]) / np.timedelta64(1, 'D')

    # 통합 과정
    final_feat = telemetry.merge(error_count, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(machines, on=['machineID'], how='left')

    # failure 결합
    labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
    # fillna 가 str일 경우 작동하지 않는 문제 발견 하여 comp 의 마지막 단어를 추출하여 fillna 진행후 comp를 다시 붙이는 방식으로 해결
    labeled_features.failure = 'comp' + labeled_features.failure.str[-1].fillna(method='bfill', limit=23) # 앞선 23칸을 failure 로 설정
    labeled_features.failure = labeled_features.failure.fillna('none') # 나머지를 none 으로 설정

    # pick the feature columns 
    scaler_cols = ['machineID', 'volt', 'rotate', 'pressure', 'vibration',
        'error1', 'error2', 'error3', 'error4', 'error5', 'comp1', 'comp2',
        'comp3', 'comp4', 'age']

    scaler_dict = {}
    for col_ in scaler_cols:
        scaler = MinMaxScaler()
        labeled_features[[col_]] = scaler.fit_transform(labeled_features[[col_]])
        scaler_dict[col_] = scaler

    # make test and training splits
    last_train_date, first_test_date = [pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')]

    train_y = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date, ['machineID','failure']])
    train_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['failure'], 1))

    test_y = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] > first_test_date, ['machineID','failure']])
    test_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] > first_test_date].drop(['failure'], 1))

    # function to reshape features into (samples, time steps, features) 
    def gen_sequence(id_df, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]

    # function to generate labels
    def gen_labels(id_df, seq_length, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[seq_length:num_elements, :]

    # pick the feature columns 
    sequence_cols = ['machineID', 'volt', 'rotate', 'pressure', 'vibration',
        'error1', 'error2', 'error3', 'error4', 'error5', 'comp1', 'comp2',
        'comp3', 'comp4', 'age', 'model_model1', 'model_model2', 'model_model3',
        'model_model4']

    # generator for the sequences
    seq_gen = (list(gen_sequence(train_X[train_X['machineID']==id], window_size, sequence_cols)) 
            for id in train_X['machineID'].unique())

    # generate sequences and convert to numpy array
    train_X = np.concatenate(list(seq_gen)).astype(np.float32)

    # generate labels
    label_gen = [gen_labels(train_y[train_y['machineID']==id], window_size, ['failure_comp1', 'failure_comp2', 
                                                                                'failure_comp3','failure_comp4', 'failure_none']) 
                for id in train_y['machineID'].unique()]
    train_y = np.concatenate(label_gen).astype(np.float32)

    # generator for the sequences
    seq_gen = (list(gen_sequence(test_X[test_X['machineID']==id], window_size, sequence_cols)) 
            for id in test_X['machineID'].unique())

    # generate sequences and convert to numpy array
    test_X = np.concatenate(list(seq_gen)).astype(np.float32)

    # generate labels
    label_gen = [gen_labels(test_y[test_y['machineID']==id], window_size, ['failure_comp1', 'failure_comp2', 
                                                                                'failure_comp3','failure_comp4', 'failure_none']) 
                for id in test_y['machineID'].unique()]
    test_y = np.concatenate(label_gen).astype(np.float32)

    return train_X, train_y, test_X, test_y , scaler


def compute_gradient(model_fn, loss_fn, x, y, targeted):
    """
    cleverhans : https://github.com/cleverhans-lab/cleverhans/blob/1115738a3f31368d73898c5bdd561a85a7c1c741/cleverhans/tf2/utils.py#L171

    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """

    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(y, model_fn(x))
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad


def fgsm(X, Y, model, loss_fn , epsilon, targeted= False):
    ten_X = tf.convert_to_tensor(X)
    grad = compute_gradient(model,loss_fn,ten_X,Y,targeted)
    dir = np.sign(grad)
    return X + epsilon * dir, Y


def bim(X, Y, model, loss_fn, epsilon, alpha, I, targeted= False):
    Xp= np.zeros_like(X)
    for t in range(I):
        ten_X = tf.convert_to_tensor(X)
        grad = compute_gradient(model,loss_fn,ten_X,Y,targeted)
        dir = np.sign(grad)
        Xp = Xp + alpha * dir
        Xp = np.where(Xp > X+epsilon, X+epsilon, Xp)
        Xp = np.where(Xp < X-epsilon, X-epsilon, Xp)
    return Xp, Y