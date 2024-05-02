from numpy import array
import pandas as pd
import numpy as np
from datetime import datetime
from numpy import asarray
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import random
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle
import csv

if __name__ == '__main__':
    multivariate_data = pd.read_csv('data/Multivariate_LSTM_data.csv')
    random.seed(1)

    multivariate_data['DATE'] = pd.to_datetime(multivariate_data['DATE'])
    input_data = multivariate_data[['UNRATE', 'WTI', 'CPI']]
    # scale all the input data between 0 and 1
    scaler = MinMaxScaler()
    scaler.fit(input_data)
    scaled_input_data = scaler.fit_transform(input_data)

    # Omit the last 20 points for our out of sample forecast
    out_of_sample_forecast_input = scaled_input_data[457:, :]
    scaled_input_data = scaled_input_data[0:457, :]

    # let's transform our remaning data into a multivariate supervised learning problem
    n_out = 12
    train_X = []
    train_Y = []

    for i in range(scaled_input_data.shape[0] - 20 - n_out):
        train_X.append(scaled_input_data[i:i + 20, :])
        train_Y.append(scaled_input_data[i + 20:i + 20 + n_out, 2])

    train_X = asarray(train_X)
    train_Y = asarray(train_Y)
    train_Y = train_Y.reshape(train_Y.shape[0], train_Y.shape[1], 1)

    assert 425 + 12 + 20 == 457

    # Divide data in training and testing: keep 50 last inputs for input samples of the test data

    X_test = train_X[375:425, :]
    X_train = train_X[0:375, :]
    Y_test = train_Y[375:425, :]
    Y_train = train_Y[0:375, :]

    # fit the best model on the training data and obtain MSE

    n_steps_in = 20
    # we only consider as input lagged CPI for now, so n_features = 1
    n_features = 3
    n_steps_out = 12

    random.seed(1)

    model = Sequential()
    model.add(LSTM(75, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    # Stacked layer of LSTM's
    model.add(LSTM(75, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    training_model = model.fit(X_train, Y_train, epochs=20, verbose=1)


    scaled_out_of_sample_forecast = model.predict(out_of_sample_forecast_input.reshape(1, 20, 3))
    scaled_out_of_sample_forecast = scaled_out_of_sample_forecast.reshape(12, 1)
    scaled_out_of_sample_forecast_repeated = np.repeat(scaled_out_of_sample_forecast, 3, axis=-1)
    list_forecast = scaler.inverse_transform(scaled_out_of_sample_forecast_repeated)[:, 2].tolist()

    print(list_forecast)
    aList = [ele for ele in list_forecast]

    with open("data/model3.csv", "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(['Data'])
        wr.writerow(aList)




