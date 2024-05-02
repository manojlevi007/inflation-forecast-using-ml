# Import useful libraries
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
    CPI_monthly = pd.read_csv('app/CPI_data.csv')
    random.seed(1)
    CPI_monthly['Dates'] = pd.to_datetime(CPI_monthly['Dates'])

    # scale all the data between 0 and 1
    scaler = MinMaxScaler()
    scaled_CPI = asarray(CPI_monthly['CPIGR']).reshape(-1, 1)


    data = [ele for ele in scaled_CPI[960:, 0].tolist()]
    with open("app/last20data.csv", "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(['Data'])
        wr.writerow(data)

    scaled_CPI = scaler.fit_transform(scaled_CPI)



    p = 20
    # print(scaled_CPI.shape)

    # We omit the last 20 observations for our out of sample forecast
    out_of_sample_forecast_input = scaled_CPI[960:, 0]

    # Retain all the data minus the last 20 observatinos for forecasting
    scaled_CPI = scaled_CPI[:960, 0]


    # let's transform our remaning data into a univariate supervised learning problem

    # Functions transforms our time series sequence into a supervised leaning problem
    def split_sequences(sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence) - (n_steps_in + n_steps_out)):
            append_X = []
            append_y = []
            for j in range(n_steps_in):
                append_X.append(sequence[i + j])
            for k in range(n_steps_out):
                append_y.append(sequence[i + n_steps_in + k + 1])

            X.append(append_X)
            y.append(append_y)

        return np.array(X), np.array(y)


    n_steps_in = 20
    # we only consider as input lagged CPI for now, so n_features = 1
    n_features = 1
    n_steps_out = 12
    # n_steps_in = number of lags we consider
    # n-_steps_out = number of periods to forecast
    X, y = split_sequences(scaled_CPI, n_steps_in=20, n_steps_out=12)

    # Obtain 93 sequential data points for test (10%)
    # we dropped the assumption that inflation was time invariant, so we want our data to train on the closest data to the ones put in the forecast. Thus the test data will be the first 10% of the data

    X_train = X[0:835]
    X_test = X[835:]
    y_train = y[0:835]
    y_test = y[835:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], n_features)

    # Fit the test data into the best model, and obtain MSE:
    model = Sequential()
    model.add(LSTM(75, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    # Stacked layer of LSTM's
    model.add(LSTM(75, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    training_model = model.fit(X_train, y_train, epochs=50, verbose=1)

    pickle.dump(model, open('model2.pkl', 'wb'))

    out_of_sample_forecast_input = asarray(out_of_sample_forecast_input).reshape(1, n_steps_in)
    out_of_sample_forecast = model.predict(out_of_sample_forecast_input, verbose=0)

    list_forecast = scaler.inverse_transform(out_of_sample_forecast).tolist()
    aList = [ele for ele in list_forecast[0]]

    with open("app/model2.csv", "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(['Data'])
        wr.writerow(aList)