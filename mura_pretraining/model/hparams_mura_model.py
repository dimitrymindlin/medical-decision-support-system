# -*- coding: utf-8 -*-
"""Mura DenseNet model"""

# external
import tensorflow as tf
import tensorflow_addons as tfa
from utils.model_utils import get_model_by_name, get_input_shape_from_config, get_preprocessing_by_name


class HparamsMuraModel(tf.keras.Model):
    """HparamsMuraModel Model Class for parameter optimisation"""

    def __init__(self, config, hp, weights='imagenet'):
        super(HparamsMuraModel, self).__init__(name='HparamsMuraModel')
        self.config = config
        #self.congig['model']['name'] = hp.Choice('model_name', ['densenet', 'vgg', 'resnet', 'inception'])
        self._input_shape = get_input_shape_from_config(self.config)
        self.img_input = tf.keras.Input(shape=self._input_shape)

        self.preprocessing_layer = get_preprocessing_by_name(self.config, self._input_shape)
        self.random_flipping_aug = tf.keras.layers.RandomFlip(mode="vertical")
        self.random_rotation_aug = tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)
        self.base_model = get_model_by_name(self.config, self.img_input, self._input_shape, weights)
        self.base_model.trainable = True
        self.classifier = tf.keras.layers.Dense(len(self.config['data']['class_names']), activation="sigmoid",
                                                name="predictions")

        self.dropout = tf.keras.layers.Dropout(0.5) if hp.Boolean("dropout") else None

        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D() if hp.Boolean("global_avg_pooling") else None

        self.augmentation = True if hp.Boolean("augmentation") else False

    def call(self, x):
        x = tfa.image.equalize(x)
        x = self.resize_with_pad(x)
        if self.augmentation:
            self.random_flipping_aug(x)
            self.random_rotation_aug(x)
        x = self.base_model(x)
        if self.global_avg_pooling:
            x = self.global_avg_pooling(x)
        if self.dropout:
            x = self.dropout(x)
        return self.classifier(x)
