# -*- coding: utf-8 -*-
"""Mura DenseNet model"""

# external
import tensorflow as tf

from utils.model_utils import get_model_by_name


class HparamsMuraModel(tf.keras.Model):
    """HparamsMuraModel Model Class for parameter optimisation"""

    def __init__(self, config, hp=None, weights='imagenet'):
        super(HparamsMuraModel, self).__init__(name='HparamsMuraModel')
        self.config = config
        self._input_shape = (
            self.config['data']['image_height'],
            self.config['data']['image_width'],
            self.config['data']['image_channel']
        )
        self.img_input = tf.keras.Input(shape=self._input_shape)

        self.preprocessing_layer, self.base_model = get_model_by_name(config, self.img_input, self._input_shape,
                                                                      weights)

        self.base_model.trainable = hp.Boolean("train_base")

        self.classifier = tf.keras.layers.Dense(len(config['data']['class_names']), activation="sigmoid",
                                                name="predictions")

        self.dropout = tf.keras.layers.Dropout(0.5) if hp.Boolean("dropout") else None

        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D() if hp.Boolean("global_avg_pooling") else None

        self.augmentation = True if hp.Boolean("augmentation") else False

        # Augmentation layers
        self.random_flipping_aug = tf.keras.layers.RandomFlip(mode="vertical")
        self.random_rotation_aug = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)

    def call(self, x):
        if self.augmentation:
            self.random_flipping_aug(x)
            self.random_rotation_aug(x)
        x = self.base_model(x)
        if self.global_avg_pooling:
            x = self.global_avg_pooling(x)
        if self.dropout:
            x = self.dropout(x)
        return self.classifier(x)
