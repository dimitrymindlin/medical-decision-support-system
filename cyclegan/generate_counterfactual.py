from datetime import datetime

import matplotlib.pyplot as plt
from tensorflow_addons.layers import InstanceNormalization

import tensorflow as tf
import os

from preprocessing.img_scaling import scale_to_zero_one

execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
writer = tf.summary.create_file_writer(f'logs/' + execution_id)
import numpy as np


def predict_imgs(classifier, g_NP, g_PN, image, label, num_images=1, attention=None):
    generator = g_NP if label == 0 else g_PN

    counterfactual_list = []
    classifications = []
    for i in range(num_images):
        translated = generator.predict(image)
        normalisation_factor = np.max((np.max(translated), np.abs(np.min(translated))))
        translated /= normalisation_factor  # [-1, 1]
        classifications.append(np.argmax(classifier.predict(translated)))
        if attention is not None:
            translated = scale_to_zero_one(translated)
            image = scale_to_zero_one(image)
            plt.imshow(np.squeeze(image))
            plt.show()
            plt.imshow(np.squeeze(translated))
            plt.show()
            translated_foreground = attention * translated
            plt.imshow(np.squeeze(translated_foreground))
            plt.show()
            img_background = (1 - attention) * image
            translated = img_background + translated_foreground
            translated = translated / np.max(translated)
        counterfactual_list.append(np.squeeze(0.5 * translated + 0.5))  # [0,1]

    return counterfactual_list, classifications
