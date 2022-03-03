# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf

from utils.model_utils import get_input_shape_from_config, get_model_by_name


class WristPredictNet(tf.keras.Model):
    "MuraNet Model Class with various base models"

    def __init__(self, config, weights='imagenet', include_top=True):
        super(WristPredictNet, self).__init__(name='WristPredictNet')

        self.config = config
        self.include_top = include_top
        self._input_shape = get_input_shape_from_config(self.config)
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.base_model = get_model_by_name(self.config, self._input_shape, weights, self.img_input)
        self.base_model.trainable = self.config['train']['train_base']
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="softmax",
                                                name="predictions")

    def call(self, x):
        x = self.base_model(x)
        """x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)"""
        x = self.classifier(x)
        return x


    def model(self):
        x = self.base_model.output
        """x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(2)(x)"""
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
