# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf
import tensorflow_addons as tfa
from utils.model_utils import get_model_by_name, get_input_shape_from_config, get_preprocessing_by_name


class WristPredictNet(tf.keras.Model):
    """MuraDenseNet Model Class with various base models"""

    def __init__(self, config, weights='imagenet', train_base=False):
        super(WristPredictNet, self).__init__(name='WristPredictNet')
        self.config = config
        self._input_shape = get_input_shape_from_config(self.config)
        self.img_input = tf.keras.Input(shape=self._input_shape)
        self.preprocessing_layer = get_preprocessing_by_name(self.config, self._input_shape)
        self.random_flipping_aug = tf.keras.layers.RandomFlip(mode="vertical")
        self.random_rotation_aug = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)
        self.base_model = get_model_by_name(self.config, self.img_input, self._input_shape, weights)
        self.base_model.trainable = train_base
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="sigmoid",
                                                name="predictions")

    def call(self, x):
        x = tfa.image.equalize(x)
        x = self.resize_with_pad(x)
        x = self.preprocessing_layer(x)  # Normalisation to [0,1]
        if self.config['train']['augmentation']:
            x = self.random_flipping_aug(x)
            x = self.random_rotation_aug(x)
        x = self.base_model(x)
        return self.classifier(x)

    def resize_with_pad(self, image):
        return tf.image.resize_with_pad(image,
                                        self.config['data']['image_height'],
                                        self.config['data']['image_width'])
