# -*- coding: utf-8 -*-
"""Mura model"""

# external
import tensorflow as tf


class WristPredictNet(tf.keras.Model):
    """MuraDenseNet Model Class with various base models"""

    def __init__(self, config, weigths='imagenet', train_base=False, model_name="densenet"):
        super(WristPredictNet, self).__init__(name='WristPredictNet')
        self.config = config
        self._input_shape = (
            self.config['data']['image_height'],
            self.config['data']['image_width'],
            self.config['data']['image_channel']
        )

        if model_name == "densenet":
            self.base_model = tf.keras.applications.DenseNet169()
            self.preprocessing_layer = tf.keras.applications.densenet.preprocess_input
        elif model_name == "vgg":
            self.base_model = tf.keras.applications.VGG19()
            self.preprocessing_layer = tf.keras.applications.vgg19.preprocess_input
        elif model_name == "resnet":
            self.base_model = tf.keras.applications.ResNet50()
            self.preprocessing_layer = tf.keras.applications.resnet50.preprocess_input
        elif model_name == "inception":
            self.base_model = tf.keras.applications.InceptionV3()
            self.preprocessing_layer = tf.keras.applications.inception_v3.preprocess_input

        self.base_model.include_top = False,
        self.base_model.input_tensor = self.img_input,
        self.base_model.input_shape = self._input_shape,
        self.base_model.weights = weigths,
        self.base_model.pooling = config['model']['pooling'],
        self.base_model.classes = len(config['data']['class_names'])
        self.base_model.trainable = train_base

        self.img_input = tf.keras.Input(shape=self._input_shape)
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

    def model(self):
        x = self.base_model.output
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)
