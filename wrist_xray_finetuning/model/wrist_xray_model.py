# -*- coding: utf-8 -*-
"""WristXrayNet model"""

# external
import tensorflow as tf

from utils.model_utils import get_model_by_name, get_input_shape_from_config


class WristXrayNet(tf.keras.Model):
    """WristXrayNet Model Class"""

    def __init__(self, config, weights='imagenet', train_base=False):
        super(WristXrayNet, self).__init__(name='WristXrayNet')
        self.config = config
        self._input_shape = get_input_shape_from_config(self.config)

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
        x = self.base_model(x)
        return self.classifier(x)

    def model(self):
        x = self.base_model.output
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)
