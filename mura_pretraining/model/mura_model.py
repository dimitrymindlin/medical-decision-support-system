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

        self.img_input = tf.keras.Input(shape=self._input_shape)

        if model_name == "densenet":
            self.preprocessing_layer = tf.keras.applications.densenet.preprocess_input
            self.base_model = tf.keras.applications.DenseNet169(include_top=False,
                                                                input_tensor=self.img_input,
                                                                input_shape=self._input_shape,
                                                                weights=weigths,
                                                                pooling=config['model']['pooling'],
                                                                classes=len(config['data']['class_names']))
        elif model_name == "vgg":
            self.preprocessing_layer = tf.keras.applications.vgg19.preprocess_input
            self.base_model = tf.keras.applications.VGG19(include_top=False,
                                                          input_tensor=self.img_input,
                                                          input_shape=self._input_shape,
                                                          weights=weigths,
                                                          pooling=config['model']['pooling'],
                                                          classes=len(config['data']['class_names']))
        elif model_name == "resnet":
            self.preprocessing_layer = tf.keras.applications.resnet50.preprocess_input
            self.base_model = tf.keras.applications.ResNet50(include_top=False,
                                                             input_tensor=self.img_input,
                                                             input_shape=self._input_shape,
                                                             weights=weigths,
                                                             pooling=config['model']['pooling'],
                                                             classes=len(config['data']['class_names']))

        elif model_name == "inception":
            self.preprocessing_layer = tf.keras.applications.inception_v3.preprocess_input
            self.base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                                input_tensor=self.img_input,
                                                                input_shape=self._input_shape,
                                                                weights=weigths,
                                                                pooling=config['model']['pooling'],
                                                                classes=len(config['data']['class_names']))


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

    def model(self):
        x = self.base_model.output
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)
