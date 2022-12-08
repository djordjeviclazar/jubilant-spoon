from settings import *
import tensorflow as tf
import numpy as np


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

def get_model(x_train, x_test, input_shape, trainable_encoder=False):
    if model_name == 'resnet50':
        return create_model_resnet50(x_train, x_test, input_shape, len(classes_list), trainable_encoder)