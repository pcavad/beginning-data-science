#!/usr/bin/env python
# coding: utf-8

import os

import keras
from keras.preprocessing import image
from keras import backend as K
from keras.models import Sequential, load_model
from keras import layers
from keras.layers import Input, Dense, Dropout, Flatten, MaxPool2D 
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SeparableConv2D 
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,EarlyStopping 
import itertools
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score 
import seaborn as sns
import tensorflow as tf

def get_path(PATH):
    '''
    This function stores the file paths and the labels for normal and pneumonia images
    '''
    try:
        #saving jpeg only image paths in lists for nromal and penumonia
        paths_norm = [PATH + 'norm/' + p for p in os.listdir(PATH + 'norm/') if p.endswith('.jpeg')] #reads file paths
        paths_pneu = [PATH + 'pneu/' + p for p in os.listdir(PATH + 'pneu/') if p.endswith('.jpeg')] #reads file paths
        #persisting the correspondent class labels
        labels_norm = [0 for i in paths_norm]
        labels_pneu = [1 for i in paths_pneu]
    except Exception as e:
        print(e)
    return paths_norm, paths_pneu, labels_norm, labels_pneu

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def modeling(activation_f=None, optimizer_f=None, input_shape_img=None):
    '''
    This function creates the model based on different parameters, complile and save it.
    '''
    model = Sequential()
    
    #1st block
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = activation_f , input_shape = input_shape_img))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    #2nd block
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = activation_f))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    #3rd block
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = activation_f))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    #4th block
    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = activation_f))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    #5th block
    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = activation_f))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Flatten())
    model.add(Dense(units = 128 , activation = activation_f))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1 , activation = 'sigmoid')) 

    model.compile(optimizer=optimizer_f, loss='binary_crossentropy', metrics=['accuracy']) # compiling ##'MeanSquaredError'
    
    model.save('./data/model.h5') # saving
    
    return model

def plot_validation_curves(result):
    '''
    This function plots the validation curves.
    '''
    result = pd.DataFrame(result)
    fig, axs = plt.subplots(1,2)
    result[['loss','val_loss']].plot(figsize=(10, 3),ax=axs[0])
    axs[0].set_title('Train vs validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    result[['accuracy','val_accuracy']].plot(figsize=(10, 3),ax=axs[1])
    axs[1].set_title('Train vs validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')



