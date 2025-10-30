from settings import *
#import tensorflow.python.keras as tfk
#from tensorflow.python.keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, Dropout, AvgPool2D, Activation
#from keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np
from keras.regularizers import L1, L2, L1L2

def get_model(x_train, x_test, input_shape, trainable_encoder=False):
    if model_name == 'resnet50':
        return create_model_resnet50(x_train, x_test, input_shape, len(classes_list), trainable_encoder)

def get_custom_model(input_shape):
    if model_type == 'D5':
        return create_custom_model_D5(input_shape, len(classes_list))
    if model_type == 'D10':
        return create_custom_model_D10(input_shape, len(classes_list))
    if model_type == 'Conv2_2':
        return create_custom_model_Conv2(input_shape, len(classes_list))
    if model_type == 'Conv2_3':
        return create_custom_model_Conv2(input_shape, len(classes_list))

def create_model_resnet50(x_train, x_test, input_shape, classes, trainable_encoder=False):
    x = tf.keras.layers.Input(shape=input_shape, name='input')

    backbone = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet',
                                                input_tensor=x, pooling='avg')

    if not trainable_encoder:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.BatchNormalization):
                layer.trainable = False

    y = backbone.output

    y = tf.keras.layers.Dropout(rate=0.5)(y)
    y = tf.keras.layers.Dense(classes, activation='softmax', name='output')(y)

    model = tf.keras.Model(inputs=x, outputs=y)

    x_train_pp = tf.keras.applications.resnet_v2.preprocess_input(x_train)
    x_test_pp = tf.keras.applications.resnet_v2.preprocess_input(x_test)
    return x_train_pp, x_test_pp, model

def create_custom_model_D5(input_shape, classes):
    input = tf.keras.layers.Input(input_shape, name='input')

    f1 = tf.keras.layers.Flatten()(input)

    d1 =    tf.keras.layers.Dense(100, activation=usual_activation, kernel_regularizer=L2(l2=L2reg), )(f1)#kernel_regularizer=L1(L1reg)#, kernel_regularizer=L1L2(l1=L1reg, l2=L2reg)#kernel_regularizer=L2(L2reg)#kernel_regularizer=L1L2(l1=L1reg, l2=L2reg)
    bn1 =   tf.keras.layers.BatchNormalization()(d1)
    d2 =    tf.keras.layers.Dense(150, activation=usual_activation, kernel_regularizer=L1(l1=L1reg))(bn1)
    d3 =    tf.keras.layers.Dense(150, activation=usual_activation, kernel_regularizer=L1(l1=L1reg))(d2)
    bn2 =   tf.keras.layers.BatchNormalization()(d3)
    d4 =    tf.keras.layers.Dense(150, activation=usual_activation, kernel_regularizer=L1(l1=L1reg))(bn2)
    d5 =    tf.keras.layers.Dense(150, activation=usual_activation, kernel_regularizer=L1(l1=L1reg))(d4)
    #bn3 = tf.keras.layers.BatchNormalization()(d5)
    #a = tf.keras.layers.Activation('relu')(bn3)
    #mp = tf.keras.layers.MaxPool1D(pool_size=(2))(a)

    #drop = tf.keras.layers.Dropout(rate=0.2)(a)
    output = tf.keras.layers.Dense(classes, activation='softmax', name='output')(d5)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model

def create_custom_model_D10(input_shape, classes):
    input = tf.keras.layers.Input(input_shape, name='input')

    f1 = tf.keras.layers.Flatten()(input)

    d1 =    tf.keras.layers.Dense(100, kernel_regularizer=L2(l2=L2reg))(f1) #, kernel_regularizer=L1L2(l1=L1reg, l2=L2reg)#kernel_regularizer=L2(L2reg)#kernel_regularizer=L1L2(l1=L1reg, l2=L2reg)
    #bn1 =  tf.keras.layers.BatchNormalization()(d1)
    a1 =    tf.keras.layers.Activation(usual_activation)(d1)
    bn1 =   tf.keras.layers.BatchNormalization()(a1)
    d2 =    tf.keras.layers.Dense(115, kernel_regularizer=L1(l1=L1reg))(bn1)
    a2 =    tf.keras.layers.Activation(usual_activation)(d2)

    d3 =    tf.keras.layers.Dense(115, kernel_regularizer=L1(l1=L1reg))(a2)
    #bn2 =  tf.keras.layers.BatchNormalization()(d3)
    a3 =    tf.keras.layers.Activation(usual_activation)(d3)
    bn2 =   tf.keras.layers.BatchNormalization()(a3) 
    d4 =    tf.keras.layers.Dense(115, kernel_regularizer=L1(l1=L1reg))(bn2)
    a4 =    tf.keras.layers.Activation(usual_activation)(d4)

    d5 = tf.keras.layers.Dense(115, kernel_regularizer=L1(l1=L1reg))(a4)
    a5 = tf.keras.layers.Activation(usual_activation)(d5)
    d6 = tf.keras.layers.Dense(115, kernel_regularizer=L1(l1=L1reg))(a5)
    a6 = tf.keras.layers.Activation(usual_activation)(d6)

    d7 =    tf.keras.layers.Dense(115, kernel_regularizer=L1(l1=L1reg))(a6)
    #bn3 =  tf.keras.layers.BatchNormalization()(d7)
    a7 =    tf.keras.layers.Activation(usual_activation)(d7)
    bn3 =   tf.keras.layers.BatchNormalization()(a7)
    d8 =    tf.keras.layers.Dense(100, kernel_regularizer=L1(l1=L1reg))(bn3)
    drop8 = tf.keras.layers.Dropout(rate=0.5)(d8)
    a8 =    tf.keras.layers.Activation(usual_activation)(drop8)

    d9 =        tf.keras.layers.Dense(100, kernel_regularizer=L1(l1=L1reg))(a8)
    drop9 =     tf.keras.layers.Dropout(rate=0.5)(d9)
    a9 =        tf.keras.layers.Activation(usual_activation)(drop9)
    d10 =       tf.keras.layers.Dense(100, kernel_regularizer=L1(l1=L1reg))(a9)
    drop10 =    tf.keras.layers.Dropout(rate=0.5)(d10)
    a10 =       tf.keras.layers.Activation(usual_activation)(drop10)
    
    output = tf.keras.layers.Dense(classes, activation='softmax', name='output')(a10)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model

def create_custom_model_Conv2(input_shape, classes):

    input = tf.keras.layers.Input(input_shape, name='input')

    zp1 =   tf.keras.layers.ZeroPadding2D(padding=(2, 2))(input)
    cnv1 =  tf.keras.layers.Conv2D(kernel_size=(5, 5), filters=30, strides=(2, 2), activation=usual_activation, kernel_regularizer=L2(l2=L2reg))(zp1)#, padding='same'
    #a1 =   tf.keras.layers.Activation(usual_activation)(cnv1)
    bn1 =   tf.keras.layers.BatchNormalization()(cnv1)
    zp2 =   tf.keras.layers.ZeroPadding2D(padding=(1, 1))(bn1)
    cnv2 =  tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=50, strides=(2, 2), activation=usual_activation, kernel_regularizer=L2(l2=L2reg))(zp2)#, padding='same'
    #a2 =   tf.keras.layers.Activation(usual_activation)(cnv2)
    bn2 =   tf.keras.layers.BatchNormalization()(cnv2)
    zp3 =   tf.keras.layers.ZeroPadding2D(padding=(1, 1))(bn2)
    cnv3 =  tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=80, strides=(1, 1), activation=usual_activation, kernel_regularizer=L2(l2=L2reg))(zp3)#, padding='same'
    bn3 =   tf.keras.layers.BatchNormalization()(cnv3)
    mp =    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(bn3)
    
    f1 =    tf.keras.layers.Flatten()(mp)
    d1 =    tf.keras.layers.Dense(150, activation=usual_activation, kernel_regularizer=L2(l2=L2reg))(f1)
    #a3 =   tf.keras.layers.Activation(usual_activation)(d1)
    #bn4 =   tf.keras.layers.BatchNormalization()(d1)
    d2 =    tf.keras.layers.Dense(150, activation=usual_activation, kernel_regularizer=L2(l2=L2reg))(d1)
    #a4 =   tf.keras.layers.Activation(usual_activation)(d2)

    output = tf.keras.layers.Dense(classes, activation='softmax', name='output')(d2)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model
