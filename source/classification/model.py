from settings import *
import tensorflow as tf
import numpy as np

def get_model(x_train, x_test, input_shape, trainable_encoder=False):
    if model_name == 'resnet50':
        return create_model_resnet50(x_train, x_test, input_shape, len(classes_list), trainable_encoder)

def get_custom_model(input_shape):
    if model_name == 'D5':
        return create_custom_model_D5(input_shape, len(classes_list))

def create_model_resnet50(x_train, x_test, input_shape, classes, trainable_encoder=False):
    x = tf.keras.layers.Input(shape=input_shape, name='input')

    backbone = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet',
                                                input_tensor=x, pooling='avg')

    if not trainable_encoder:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    y = backbone.output

    y = tf.keras.layers.Dropout(rate=0.5)(y)
    y = tf.keras.layers.Dense(classes, activation='softmax', name='output')(y)

    model = tf.keras.models.Model(inputs=x, outputs=y)

    x_train_pp = tf.keras.applications.resnet_v2.preprocess_input(x_train)
    x_test_pp = tf.keras.applications.resnet_v2.preprocess_input(x_test)
    return x_train_pp, x_test_pp, model

def create_custom_model_D5(input_shape, classes):
    input = tf.keras.layers.Input(input_shape, name='input')

    f1 = tf.keras.layers.Flatten()(input)

    d1 = tf.keras.layers.Dense(100, activation='relu')(f1)
    bn1 = tf.keras.layers.BatchNormalization()(d1)
    d2 = tf.keras.layers.Dense(100, activation='relu')(bn1)
    d3 = tf.keras.layers.Dense(100, activation='relu')(d2)
    bn2 = tf.keras.layers.BatchNormalization()(d3)
    d4 = tf.keras.layers.Dense(100, activation='relu')(bn2)
    d5 = tf.keras.layers.Dense(100, activation='relu')(d4)
    bn3 = tf.keras.layers.BatchNormalization()(d5)
    #a = tf.keras.layers.Activation('relu')(bn3)
    #mp = tf.keras.layers.MaxPool1D(pool_size=(2))(a)

    #drop = tf.keras.layers.Dropout(rate=0.2)(a)
    output = tf.keras.layers.Dense(classes, activation='softmax', name='output')(bn3)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model