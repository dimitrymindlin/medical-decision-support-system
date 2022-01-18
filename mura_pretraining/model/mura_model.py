# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf

from utils.model_utils import get_model_by_name


class MuraNet(tf.keras.Model):
    """MuraDenseNet Model Class with various base models"""

    def __init__(self, config, weights='imagenet', train_base=False):
        super(MuraNet, self).__init__(name='WristPredictNet')
        self.config = config
        self._input_shape = (
            self.config['data']['image_height'],
            self.config['data']['image_width'],
            self.config['data']['image_channel']
        )

        self.img_input = tf.keras.Input(shape=self._input_shape)

        self.preprocessing_layer, self.base_model = get_model_by_name(config, self.img_input, self._input_shape,
                                                                      weights)
        self.base_model.trainable = train_base

        self.classifier = tf.keras.layers.Dense(len(config['data']['class_names']), activation="sigmoid",
                                                name="predictions")

        # Augmentation layers
        self.random_flipping_aug = tf.keras.layers.RandomFlip(mode="vertical")
        self.random_rotation_aug = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)

    def call(self, inputs):
        x = self.preprocessing_layer(inputs)
        x = self.random_flipping_aug(x)
        x = self.random_rotation_aug(x)
        x = self.base_model(x)
        return self.classifier(x)
