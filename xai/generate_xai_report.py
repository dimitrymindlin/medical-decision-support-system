import PIL
import numpy as np
from tensorflow_addons.layers import InstanceNormalization
import matplotlib.pyplot as plt
from configs.direct_training_config import direct_training_config
from cyclegan.generate_counterfactual import predict_single
from mura_finetuning.dataloader.mura_generators import MuraGeneratorDataset
import tensorflow as tf
import tensorflow_addons as tfa
import os

from tf_explain.core.grad_cam import GradCAM
from lime import lime_image

from utils.xai_utils import image_stack, get_diff_maps_for_imgs
from xai.keras_gradcam import get_keras_gradcam
from xai.save_lime import get_lime
import skimage as ski

# Create output dirs
from xai.tf_explain_gradcam import get_tf_explain_gradcam

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

classifier_folder = f"../checkpoints/2022-03-04--16.16/cp.ckptmodel"
cyclegan_folder = "../checkpoints/GANterfactual_2022-03-07--23.04"

# Data
mura_data = MuraGeneratorDataset(direct_training_config)

# Classifier Model
metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
classifier = tf.keras.models.load_model(classifier_folder, custom_objects={'f1_score': metric_f1})

# Explainer models
explainer_grad_cam = GradCAM()
explainer_lime = lime_image.LimeImageExplainer()
custom_objects = {"InstanceNormalization": InstanceNormalization}
g_NP = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.h5'),
                                  custom_objects=custom_objects)
g_PN = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.h5'),
                                  custom_objects=custom_objects)

# Iterate over data
for batch_i, batch in enumerate(mura_data.raw_valid_loader):
    if batch_i > 15:
        break
    for example_i, example in enumerate(zip(batch[0], batch[1])):
        # Image Preprocessing
        # img_raw is the rescaled image (img_width, img_height) withouth preprocessing
        image_raw, label_ndarray = example[0].numpy(), example[1]  # [0.0, 255.]
        image = mura_data.preprocess_img(image_raw, model_name="inception")
        image = image.numpy()  # image is the preprocessed image in numpy: [-1, 1]
        # TODO: image and image_array same?
        image_array = tf.keras.utils.img_to_array(image)  # image_array is the preprocessed image in numpy array
        image_batch = tf.expand_dims(image_array, axis=0)  # image_batch is the batched image tensor

        true_y = np.argmax(label_ndarray)
        pred_y = np.argmax(classifier.predict(image_batch))

        # Sub-Directory Name
        sub_dir = "correct" if pred_y == true_y else "wrong"
        file_name = f"index({batch_i})_T({true_y})_P({pred_y}).png"

        # counterfactual with GAN
        counterfactual_list, classifications = predict_single(classifier, g_NP, g_PN, image_batch, true_y, num_images=3)

        # diff_map_list = get_diff_maps_for_imgs(image_raw, counterfactual_list)

        counterfactuals_lime = []
        counterfactuals_gradcam = []
        for counterfactual, classification in zip(counterfactual_list, classifications):
            counterfactual_scaled_minus_1 = counterfactual * 2 - 1.
            counterfactuals_gradcam.append(
                get_tf_explain_gradcam(explainer_grad_cam, counterfactual_scaled_minus_1, classifier, classification,
                                       sub_dir, file_name))
            counterfactuals_lime.append(get_lime(explainer_lime, classifier, counterfactual_scaled_minus_1, sub_dir, file_name))

        # tf-explain GradCAM
        tf_explain_gradcam = get_tf_explain_gradcam(explainer_grad_cam, image_array, classifier, true_y,
                                                    sub_dir, file_name)
        # lime
        lime_result = get_lime(explainer_lime, classifier, image, sub_dir, file_name)

        # heatmap with keras example GradCAM
        # keras_gradcam = get_keras_gradcam(classifier, image_batch, image_raw, sub_dir, file_name)

        img_list = [image_raw, tf_explain_gradcam, lime_result,
                    counterfactual_list[0], counterfactuals_gradcam[0], counterfactuals_lime[0],
                   counterfactual_list[1], counterfactuals_gradcam[1], counterfactuals_lime[1],
                   counterfactual_list[2], counterfactuals_gradcam[2],counterfactuals_lime[2],]

        """stacked_imgs = np.hstack(
            (image_raw, tf_explain_gradcam, lime_result, counterfactual_list[0], counterfactual_list[1]))

        ski.io.imsave(f"xai_results/{file_name}", stacked_imgs)"""

        r = 4
        c = 3
        titles = ['Original', 'GradCam', 'Lime',
                  'Counterfactual 1', 'GradCam', 'Lime',
                  'Counterfactual 2', 'GradCam', 'Lime',
                  'Counterfactual 3', 'GradCam', 'Lime', ]

        fig, axs = plt.subplots(r, c, figsize=(20, 20))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(img_list[cnt][:, :, 0])  # cmap='gray'
                if i == 0:
                    if j == 0:
                        axs[i, j].set_title(f'{titles[cnt]} (T: {true_y} | P: {pred_y})')
                    else:
                        axs[i, j].set_title(f'{titles[cnt]})')
                else:
                    if j == 0:
                        axs[i, j].set_title(f'{titles[cnt]} (P: {classifications[j]})')
                    else:
                        axs[i, j].set_title(f'{titles[cnt]}')

                # axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"xai_results/2022-03-07--23.04/{file_name}")
        plt.close()

