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

class WristPredictNetHP(tf.keras.Model):
    "MuraNet Model Class with various base models"

    def __init__(self, config, hp, weights='imagenet'):
        super(WristPredictNetHP, self).__init__(name='WristPredictNetHP')
        self.config = config
        self.weight_regularisation = regularizers.l2(hp.Choice('weight_regularisation', [0.0004, 0.001]))
        self.additional_layers = hp.Choice('additional_layers', [1, 4])
        self.dense_neurons = hp.Choice('dense_neurons', [128, 64])
        self.dropout_value = hp.Choice('dropout_value', [0.4, 0,6])
        self._input_shape = get_input_shape_from_config(self.config)
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = get_model_by_name(self.config, self._input_shape, weights, self.img_input)
        self.base_model.trainable = self.config['train']['train_base']
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="softmax",
                                                name="predictions")

    def call(self, x):
        x = self.base_model(x)
        for layer_count in range(self.additional_layers):
            print("Adding additional layers...")
            x = tf.keras.layers.Dense(self.dense_neurons, activation='relu', kernel_regularizer=self.weight_regularisation)(x)
            x = tf.keras.layers.Dropout(self.dropout_value)(x)
        x = self.classifier(x)
        return x

    def model(self):
        x = self.base_model.output
        for layer_count in range(self.additional_layers):
            print("Adding additional layers...")
            x = tf.keras.layers.Dense(self.dense_neurons, activation='relu',
                                      kernel_regularizer=self.weight_regularisation)(x)
            x = tf.keras.layers.Dropout(self.dropout_value)(x)
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)


def get_working_mura_model():
    base_model = tf.keras.applications.InceptionV3(
        input_shape=(224, 224, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top

    # Create a new model on top
    input_image = tf.keras.layers.Input((224, 224, 3))
    # x = tf.keras.applications.inception_v3.preprocess_input(input_image)  # Normalisation to [0,1]
    x = base_model(input_image)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  ##### <-
    # x=keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024)(x)  ###
    x = tf.keras.layers.Activation(activation='relu')(x)  ###
    x = tf.keras.layers.Dropout(0.5)(x)  ###
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2)(x)
    out = tf.keras.layers.Activation(activation='softmax')(x)

    return tf.keras.Model(inputs=input_image, outputs=out)
