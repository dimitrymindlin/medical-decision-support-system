from tf_keras_vis.utils.scores import CategoricalScore
import tensorflow as tf
import numpy as np

def get_gradcam_plus_plus(gradcam, img, label, return_attention_map=False):

    cam = gradcam(CategoricalScore(label), img)  # [0,1]
    cam = tf.expand_dims(cam, axis=-1)
    cam = tf.image.grayscale_to_rgb(tf.convert_to_tensor(cam)) # [1,512,512,3] like img
    normalisation_factor = np.max((np.max(cam), np.abs(np.min(cam))))
    cam /= normalisation_factor  # [-1, 1]
    # Rescale img to 0,1
    img = 0.5 * img + 0.5  # [0, 1]

    # Interpolate by addition and normalise back to 0,1
    img = tf.math.add(img, cam)
    img /= np.max(img)
    if return_attention_map:
        return np.squeeze(img * 2.0 - 1), cam   # [-1,1]
    else:
        return np.squeeze(img * 2.0 - 1)
