import tensorflow as tf
import matplotlib.cm as cm
import numpy as np
import cv2

from utils.xai_utils import image_stack

def find_target_layer(model):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except AttributeError:
            print("Output ...")
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


def make_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    if not last_conv_layer_name:
        last_conv_layer_name = find_target_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_heatmap(img, heatmap, cam_path="cam.jpg", alpha=0.4, img_raw=None, save=False):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img = image_stack(img_raw, superimposed_img)
    #superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    if save:
        cv2.imwrite(cam_path, superimposed_img)
    return superimposed_img

def get_keras_gradcam(model, image_batch, image_raw, sub_dir, file_name, save=False):
    output_heatmap = make_heatmap(image_batch, model)
    heatmap_png = save_heatmap(image_batch[0], output_heatmap, cam_path=f"xai_results/heatmap/{sub_dir}/{file_name}", alpha=0.005,
                     img_raw=image_raw, save=save)
    return heatmap_png