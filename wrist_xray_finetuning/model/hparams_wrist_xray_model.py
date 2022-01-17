# -*- coding: utf-8 -*-
"""Mura DenseNet model"""

# external
import tensorflow as tf

from model_utils import get_model_by_name


class HparamsWristXrayModel(tf.keras.Model):
    """HparamsMuraModel Model Class for parameter optimisation"""

    def __init__(self, config, hp=None):
        super(HparamsWristXrayModel, self).__init__(name='HparamsWristXrayModel')
        self.config = config
        self._input_shape = (
            self.config['data']['image_height'],
            self.config['data']['image_width'],
            self.config['data']['image_channel']
        )
        self.img_input = tf.keras.Input(shape=self._input_shape)

        self.weights = 'imagenet'

        self.base_model.trainable = hp.Boolean("train_base")

        self.preprocessing_layer, self.base_layer = get_model_by_name(config, self.img_input, self._input_shape,
                                                                      self.weights)

        self.base_model.trainable = hp.Boolean("train_base")

        self.classifier = tf.keras.layers.Dense(len(config['data']['class_names']), activation="sigmoid",
                                                name="predictions")

        self.dropout = tf.keras.layers.Dropout(0.5) if hp.Boolean("dropout") else None

        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D() if hp.Boolean("global_avg_pooling") else None

    def call(self, inputs):
        x = self.base_model(inputs)
        if self.global_avg_pooling:
            x = self.global_avg_pooling(x)
        if self.dropout:
            x = self.dropout(x)
        return self.classifier(x)

    def model(self):
        x = self.base_model.output
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)
