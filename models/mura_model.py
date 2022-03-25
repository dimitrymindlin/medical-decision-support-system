# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf
from keras import regularizers

from utils.model_utils import get_input_shape_from_config, get_model_by_name


class WristPredictNet(tf.keras.Model):
    "MuraNet Model Class with various base models"

    def __init__(self, config, weights='imagenet'):
        super(WristPredictNet, self).__init__(name='WristPredictNet')
        self.config = config
        self.weight_regularisation = regularizers.l2(config["train"]["weight_regularisation"]) if config["train"][
            "weight_regularisation"] else None
        self._input_shape = get_input_shape_from_config(self.config)
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = get_model_by_name(self.config, self._input_shape, weights, self.img_input)
        self.base_model.trainable = self.config['train']['train_base']
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="softmax",
                                                name="predictions")

    def call(self, x):
        x = self.base_model(x)
        if self.config["train"]["additional_last_layers"]:
            for layer_count in range(self.config["train"]["additional_last_layers"]):
                print("Adding additional layers...")
                x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
                x = tf.keras.layers.Dropout(self.config["train"]["dropout_value"])(x)
        x = self.classifier(x)
        return x

    def model(self):
        x = self.base_model.output
        if self.config["train"]["additional_last_layers"]:
            for layer_count in range(self.config["train"]["additional_last_layers"]):
                print("Adding additional layers...")
                x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
                x = tf.keras.layers.Dropout(self.config["train"]["dropout_value"])(x)
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)


def get_working_mura_model_hp(config, hp, weight_regularisation_value):
    weight_regularisation = regularizers.l2(weight_regularisation_value)
    additional_layers = hp.Choice('additional_layers', [1, 4])
    dense_neurons = hp.Choice('dense_neurons', [128, 64])
    dropout_value = hp.Choice('dropout_value', [0.4, 0.6])
    _input_shape = get_input_shape_from_config(config)
    img_input = tf.keras.Input(shape=_input_shape)
    base_model = get_model_by_name(config, _input_shape, "imagenet", img_input)
    base_model.trainable = config['train']['train_base']
    classifier = tf.keras.layers.Dense(len(config['data']['class_names']), activation="softmax",
                                            name="predictions")

    # Create a new model on top
    input_image = img_input
    x = base_model(input_image)
    for layer_count in range(additional_layers):
        print("Adding additional layers...")
        x = tf.keras.layers.Dense(dense_neurons, activation='relu', kernel_regularizer=weight_regularisation)(
            x)
        x = tf.keras.layers.Dropout(dropout_value)(x)
    out = classifier(x)
    return tf.keras.Model(inputs=input_image, outputs=out)
