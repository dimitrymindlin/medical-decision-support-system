import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


class KID(keras.metrics.Metric):
    def __init__(self, image_size, name="kid", **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean()

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                layers.InputLayer(input_shape=(image_size, image_size, 3)),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(image_size, image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
                batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


def calc_KID_for_model(translated_images, translation_name, crop_size, train_horses, train_zebras):
    kid = KID(image_size=crop_size)
    kid_value_list = []

    if translation_name == "A2B":
        real_images = train_zebras
        source_domain = train_horses
    else:
        real_images = train_horses
        source_domain = train_zebras

    for i in range(5):
        sample = real_images.take(int(len(translated_images)/2))
        source_domain_sample_count = len(translated_images) - int(len(translated_images)/2)
        real_images_sample = tf.squeeze(tf.convert_to_tensor(list(sample)))
        source_samples = source_domain.take(source_domain_sample_count)
        source_images_sample = tf.squeeze(tf.convert_to_tensor(list(source_samples)))
        all_samples = tf.concat((real_images_sample, source_images_sample), axis=0)
        kid.reset_state()
        kid.update_state(all_samples,
                         tf.convert_to_tensor(translated_images), )
        kid_value_list.append(float("{0:.3f}".format(kid.result().numpy())))

    print(kid_value_list)
    mean = float("{0:.3f}".format(np.mean(kid_value_list) * 100))
    std = float("{0:.3f}".format(np.std(kid_value_list, dtype=np.float64) * 100))
    print("KID mean", mean)
    print("KID STD", std)