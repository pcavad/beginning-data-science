#!/usr/bin/env python
# coding: utf-8

# In[211]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Activation, Dropout


# In[271]:


def get_n_last_periods(df, series_name, n_periods):
    """
    Extract last n periods of a time series
    """
    
    return df[series_name][-(n_periods):] 

def plot_n_last_periods(df, series_name, n_periods, description='months'):
    """
    Plot last n periods of a time series 
    """
    plt.figure(figsize = (10,5))   
    plt.plot(get_n_last_periods(df, series_name, n_periods), 'k-')
    plt.title('{0} Total sales - {1} {2}'
              .format(series_name, n_periods, description))
    plt.xlabel('Recorded')
    plt.ylabel('Reading')
    plt.grid(alpha=0.3)


# In[282]:


def get_keras_format_series(series):
    """
    Convert a series to a numpy array of shape 
    arguments
    ---------
    time series
    
    returns
    ---------
    [batches, time_steps, features]
    """
    
    series = np.array(series)
    return series.reshape(series.shape[0], series.shape[1], 1) 

def get_train_test_data(df, series_name, series_periods, input_periods, 
                        test_periods, sample_gap=1): 
    """
    Utility processing function that splits time series into 
    train and test with keras-friendly format, according to user-specified
    choice of shape.    
    
    arguments
    ---------
    df (dataframe): dataframe with time series columns
    series_name (string): column name in df
    series_periods (int): total periods to extract
    input_periods (int): length of sequence input to network 
    test_periods (int): length of held-out terminal sequence
    sample_gap (int): step size between start of train sequences; default 1
    
    returns
    ---------
    tuple: train_X, test_X_init, train_y, test_y     
    """
    
    forecast_series = get_n_last_periods(df, series_name, series_periods).values

    train = forecast_series[:-test_periods] 
    test = forecast_series[-test_periods:] 

    train_X, train_y = [], []

    # range 0 through # of train samples - input_months by sample_gap. 
    # This is to create many samples with corresponding
    for i in range(0, train.shape[0]-input_periods, sample_gap): 
        train_X.append(train[i:i+input_periods]) # each training sample is a "series" of length input periods
        train_y.append(train[i+input_periods]) # each y is just the next step after training sample

    train_X = get_keras_format_series(train_X) 
    train_y = np.array(train_y) # make sure y is an array to work properly with keras
    
    # The set that we had held out for testing (must be same length as original train input)
    test_X = test[:input_periods] 
    test_y = test[input_periods:] # test_y is remaining values from test set
    
    return train_X, test_X, train_y, test_y


# In[3]:


def fit_SimpleRNN(train_X, train_y, cell_units, epochs, activation_func='tanh', optimizer='Adam'):
    """
    Fit Simple RNN to data train_X, train_y 
    
    arguments
    ---------
    train_X (array): input sequence samples for training 
    train_y (list): next step in sequence targets
    cell_units (int): number of hidden units for RNN cells  
    epochs (int): number of training epochs   
    """

    # initialize model
    model = Sequential() 
    
    # construct an RNN layer with specified number of hidden units
    # per cell and desired sequence input format 
    model.add(SimpleRNN(cell_units, input_shape=(train_X.shape[1],1)))
    model.add(Activation(activation_func))
    
    # add an output layer to make final predictions 
    model.add(Dense(1))
    
    # define the loss function / optimization strategy, and fit
    # the model with the desired number of passes over the data (epochs) 
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=32, verbose=0)
    
    return history, model


# In[4]:


def fit_LSTM(train_X, train_y, cell_units, epochs, activation_func='tanh', optimizer='Adam'):
    """
    Fit LSTM to data train_X, train_y 
    
    arguments
    ---------
    train_X (array): input sequence samples for training 
    train_y (list): next step in sequence targets
    cell_units (int): number of hidden units for LSTM cells  
    epochs (int): number of training epochs   
    """
    
    # initialize model
    model = Sequential() 
    
    # construct a LSTM layer with specified number of hidden units
    # per cell and desired sequence input format 
    model.add(LSTM(cell_units, input_shape=(train_X.shape[1],1))) 
    model.add(Activation(activation_func))
    
    # add an output layer to make final predictions 
    model.add(Dense(1))
    
    # define the loss function / optimization strategy, and fit
    # the model with the desired number of passes over the data (epochs) 
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=32, verbose=0)
    
    return history, model


# In[5]:


def fit_LSTM_stacked(train_X, train_y, cell_units, epochs, activation_func='tanh', optimizer='Adam'):
    """
    Fit LSTM to data train_X, train_y 
    
    arguments
    ---------
    train_X (array): input sequence samples for training 
    train_y (list): next step in sequence targets
    cell_units (int): number of hidden units for LSTM cells  
    epochs (int): number of training epochs   
    """
    
    # initialize model
    model = Sequential() 
    
    # construct a LSTM layer with specified number of hidden units
    # per cell and desired sequence input format 
    model.add(LSTM(cell_units, input_shape=(train_X.shape[1],1), activation = activation_func, return_sequences=True))
    model.add(LSTM(cell_units, activation = activation_func)) 
    
    # add an output layer to make final predictions 
    model.add(Dense(1))
    
    # define the loss function / optimization strategy, and fit
    # the model with the desired number of passes over the data (epochs) 
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=32, verbose=0)
    
    return history, model


# In[312]:

def predict(X_init, n_steps, model):
    """
    Given an input series matching the model's expected format,
    generates model's predictions for next n_steps in the series      
    """
    
    X_init = X_init.copy().reshape(1,-1,1)
    preds = []
    
    # iteratively take current input sequence, generate next step pred,
    # and shift input sequence forward by a step (to end with latest pred).
    # collect preds as we go.
    for _ in range(n_steps):
        pred = model.predict(X_init)
        preds.append(pred)
        X_init[:,:-1,:] = X_init[:,1:,:] # replace first 11 values with 2nd through 12th
        X_init[:,-1,:] = pred # replace 12th value with prediction
    
    preds = np.array(preds).reshape(-1,1)
    
    return preds

def predict_and_plot(X_init, y, model, title, test_periods):
    """
    Given an input series matching the model's expected format,
    generates model's predictions for next n_steps in the series,
    and plots these predictions against the ground truth for those steps 
    
    arguments
    ---------
    X_init (array): initial sequence, must match model's input shape
    y (array): true sequence values to predict, follow X_init
    model (keras.models.Sequential): trained neural network
    title (string): plot title   
    """
    
    y_preds = predict(X_init, n_steps=len(y), model=model) # predict through length of y
    # Below ranges are to set x-axes
    start_range = range(1, X_init.shape[0]+1) #starting at one through to length of test_X_init to plot X_init
    predict_range = range(X_init.shape[0], test_periods)  #predict range is going to be from end of X_init to length of test_months
    
    #using our ranges we plot X_init
    plt.plot(start_range, X_init)
    #and test and actual preds
    plt.plot(predict_range, y, color='orange')
    plt.plot(predict_range, y_preds, color='teal', linestyle='--')
    
    plt.title(title)
    plt.legend(['Initial Series','Target Series','Predictions'])

