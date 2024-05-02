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
    CPI_monthly = pd.read_csv('data/CPI_data.csv')
    random.seed(1)

    # convert dates into date time format

    CPI_monthly['Dates'] = pd.to_datetime(CPI_monthly['Dates'])

    cpi_monthly = CPI_monthly.iloc[0:CPI_monthly.shape[0]]

    # scale all the data between 0 and 1
    scaler = MinMaxScaler()
    scaled_CPI = asarray(CPI_monthly['CPIGR']).reshape(-1, 1)
    scaled_CPI = scaler.fit_transform(scaled_CPI)

    # the number of lags we will consider here (will consider other lags: p = [24,20,19,14,12])
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


    # n_steps_in = number of lags we consider
    # n-_steps_out = number of periods to forecast
    X, y = split_sequences(scaled_CPI, n_steps_in=20, n_steps_out=12)


    # Let's obtain our test data, which is +-10 % of our remaining time series (928 observations):
    # We want the order of our sequences of observations for the test data to be random with respect to time, so:
    # sample randomly 93 values between 1 and 928

    random.seed(1)

    X_test = []
    y_test = []

    for i in range(93):
        index = random.randint(0, 927 - i)
        X_test.append(X[index])
        y_test.append(y[index])
        # Take away the values in our time series that will be used for testing only
        X = np.delete(X, index, 0)
        y = np.delete(y, index, 0)

    X_test = asarray(X_test)
    y_test = asarray(y_test)

    # We also want the order of our sequences of observations for the training data to be random with respect to time, so:
    X, y = shuffle(X, y, random_state=0)

    # the shuffle keeps the correct mapping between X and y, which is very convenient


    # Apply K fold cross validation for the training data of X and y

    # We select k = 5, since we have 928-93 = 835 observations left after dividing between test and training. 835 is divisible by 5, so the numbers work out.
    kf_X = KFold(n_splits=5)
    partitions_X_train = []
    partitions_X_valid = []
    for train_index, validation_index in kf_X.split(X):
        set_train_X = []
        set_valid_X = []
        for j in train_index:
            set_train_X.append(X[j])
        partitions_X_train.append(asarray(set_train_X))
        for k in validation_index:
            set_valid_X.append(X[k])
        partitions_X_valid.append(asarray(set_valid_X))

    kf_y = KFold(n_splits=5)
    partitions_y_train = []
    partitions_y_valid = []
    for train_index, validation_index in kf_y.split(y):
        set_train_y = []
        set_valid_y = []
        for j in train_index:
            set_train_y.append(y[j])
        partitions_y_train.append(asarray(set_train_y))
        for k in validation_index:
            set_valid_y.append(y[k])
        partitions_y_valid.append(asarray(set_valid_y))

    partitions_X_train = asarray(partitions_X_train)
    partitions_y_train = asarray(partitions_y_train)
    partitions_X_valid = asarray(partitions_X_valid)
    partitions_y_valid = asarray(partitions_y_valid)

    n_steps_in = 20
    n_features = 1 # we only consider as input lagged CPI for now, so n_features = 1
    n_steps_out = 12

    Set_of_average_cross_valid_MSE = []

    X= X.reshape(X.shape[0], X.shape[1], n_features)
    y = y.reshape(y.shape[0], y.shape[1], n_features)

    # Fit the test data into the best model, and obtain MSE:
    model = Sequential()
    model.add(LSTM(75, activation = 'relu', return_sequences = True, input_shape = (n_steps_in,n_features)))
    # Stacked layer of LSTM's
    model.add(LSTM(75, activation = 'relu', return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(n_steps_out))
    model.compile(loss = 'mse', optimizer = 'adam')
    training_model = model.fit(X, y, epochs=50,verbose=1)
    pickle.dump(model, open('model1.pkl', 'wb'))

    out_of_sample_forecast_input = asarray(out_of_sample_forecast_input).reshape(1, n_steps_in)
    out_of_sample_forecast = model.predict(out_of_sample_forecast_input, verbose=0)

    list_forecast= scaler.inverse_transform(out_of_sample_forecast).tolist()
    print(list_forecast[0])
    aList= [ele for ele in list_forecast[0]]

    with open("app/model1.csv", "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(['Data'])
        wr.writerow(aList)


