# -*- coding: utf-8 -*-
'''使用keras'''

import tensorflow as tf
from keras import regularizers
from keras.layers import *
from keras.models import *
from keras.optimizers import  Adam
from keras import backend as K
from keras import initializers
import keras
from keras.callbacks import Callback
from keras import metrics


def get_model():
    input=Input(shape=(300,1))
    conv1_=Conv1D(filters=32,kernel_size=3,padding='valid',activation='relu')
    conv1=conv1_(input)
    pool1_=MaxPooling1D(pool_size=3)
    pool1=pool1_(conv1)
    conv2_=Conv1D(filters=64,kernel_size=3,padding='valid',activation='relu')
    conv2=conv2_(pool1)
    pool2_=MaxPooling1D(pool_size=4)
    pool2=pool2_(conv2)
    pool2=Dropout(0.5)(pool2)
    x=Flatten()(pool2)
    dense1_=Dense(640,activation='relu',kernel_regularizer=regularizers.l2(0.01))
    dense1=dense1_(x)
    dense1=Dropout(0.25)(dense1)

    output=Dense(1,activation='sigmoid')(dense1)
    model=keras.Model(inputs=input,outputs=output)
    print(model.summary())
    return model


