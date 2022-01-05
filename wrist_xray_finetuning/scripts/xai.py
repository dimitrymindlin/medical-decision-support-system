#!/usr/bin/env python3

import tensorflow as tf
import skimage as ski
import os
import matplotlib.pyplot as plt

from tf_explain.core.grad_cam import GradCAM
from lime import lime_image
from configs.wrist_xray_config import wrist_xray_config
from wrist_xray_finetuning.dataloader import WristXrayDataset
from wrist_xray_finetuning.model.wrist_xray_model import WristXrayDenseNet
import tensorflow_datasets as tfds

from explainable_ai.keras import make_heatmap, save_heatmap
from explainable_ai.utils import image_stack

config = wrist_xray_config

#Create output dirs
dirs = [
    "xai_results/lime/wrong",
    "xai_results/lime/correct",
    "xai_results/GradCAM/wrong",
    "xai_results/GradCAM/correct",
    "xai_results/heatmap/wrong",
    "xai_results/heatmap/correct",
]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Data
dataset = WristXrayDataset(config)
data = dataset.test.take(10)
#fig = tfds.visualization.show_examples(dataset.ds_test, dataset.ds_info)
#plt.show()

model = WristXrayDenseNet(config).model()
model.load_weights("../../checkpoints/mura/best/cp.ckpt")

# Explainer 
explainer_grad_cam = GradCAM()
explainer_lime = lime_image.LimeImageExplainer()

# Iterate over data
for index, example in enumerate(data):
    image_raw, label_raw = example[0].numpy(), example[1].numpy()
    image, label = dataset.preprocess(image_raw, label_raw)
    image = image.numpy()
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(image_array, axis=0)
    # Predict Label
    prediction = model.predict(image_batch)
    prediction = prediction[0][0]
    label_prediction = round(prediction)

    # Sub-Directory Name
    sub_dir = "correct" if label == label_prediction else "wrong"
    file_name = f"index({index})_label({label})_prediction({prediction}).png"

    # tf-explain GradCAM
    output_grad_cam = explainer_grad_cam.explain(([image_array], None), model, class_index=0)
    explainer_grad_cam.save(image_stack(image_raw, output_grad_cam), f"xai_results/GradCAM/{sub_dir}", output_name=file_name)

    # lime
    output_lime = explainer_lime.explain_instance(image.astype('double'), model.predict)
    output_lime_image, output_lime_mask = output_lime.get_image_and_mask(output_lime.top_labels[0], positive_only=False, hide_rest=False)
    output_lime_boundaries = ski.segmentation.mark_boundaries(output_lime_image, output_lime_mask)
    output_lime_stack = image_stack(image_raw, output_lime_boundaries * 255)
    ski.io.imsave(f"xai_results/lime/{sub_dir}/{file_name}", output_lime_stack)

    #heatmap with keras example GradCAM
    output_heatmap = make_heatmap(image_batch, model, "conv5_block16_2_conv")
    save_heatmap(image_batch[0], output_heatmap, cam_path=f"xai_results/heatmap/{sub_dir}/{file_name}", alpha=0.005, img_raw=image_raw)