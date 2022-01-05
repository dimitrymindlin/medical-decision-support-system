import cv2
import numpy as np
import tensorflow as tf

def image_stack(image_raw, image):
    size = np.max([image_raw.shape[0], image_raw.shape[1], image_raw.shape[2]])

    image_raw = tf.image.resize_with_pad(
            image_raw, size, size, 
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False)
    
    image = tf.image.resize_with_pad(
            image, size, size, 
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False)

    return np.hstack((image_raw, image))