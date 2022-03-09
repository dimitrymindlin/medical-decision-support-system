from datetime import datetime

from tensorflow_addons.layers import InstanceNormalization

import tensorflow as tf
import os

execution_id = datetime.now().strftime("%Y-%m-%d--%H.%M")
writer = tf.summary.create_file_writer(f'logs/' + execution_id)
import numpy as np

def predict_single(classifier, g_NP, g_PN, image, label, num_images=1):
    generator = g_NP if label == 0 else g_PN

    counterfactual_list = []
    classifications = []
    for i in range(num_images):
        translated = generator.predict(image)
        classifications.append(np.argmax(classifier.predict(translated)))
        counterfactual_list.append(np.squeeze(0.5 * translated + 0.5))

    return counterfactual_list, classifications
