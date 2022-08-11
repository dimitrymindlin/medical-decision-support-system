import numpy as np
from tensorflow_addons.layers import InstanceNormalization
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from configs.direct_training_config import direct_training_config
from cyclegan.generate_counterfactual import predict_imgs
from dataloader.mura_wrist_dataset import MuraDataset
import tensorflow as tf
import os

from tf_explain.core.grad_cam import GradCAM
from lime import lime_image
from os.path import exists
from xai.gradcam_plus_plus import get_gradcam_plus_plus

# Create output dirs
from xai.tf_explain_gradcam import get_tf_explain_gradcam

classifier_folder = f"../checkpoints/2022-03-24--12.42/model"
cyclegan_ckps = ["2022-03-29--00.56"]
RESULTS_PATH = "xai_results_plus_plus"
# Data
mura_data = MuraDataset(direct_training_config)

# Classifier Model
# metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
# classifier = tf.keras.models.load_model(classifier_folder, custom_objects={'f1_score': metric_f1})
classifier = tf.keras.models.load_model(classifier_folder, compile=False)

# Explainer models
explainer_grad_cam = GradCAM()
explainer_grad_cam_plus_plus = GradcamPlusPlus(classifier, model_modifier=ReplaceToLinear(), clone=True)
explainer_lime = lime_image.LimeImageExplainer()
custom_objects = {"InstanceNormalization": InstanceNormalization}


for ckp in cyclegan_ckps:
        try:
            cyclegan_folder = f"../GAN_checkpoints/GANterfactual_{ckp}"
            g_NP = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_np.h5'),
                                              custom_objects=custom_objects)
            g_PN = tf.keras.models.load_model(os.path.join(cyclegan_folder, 'generator_pn.h5'),
                                              custom_objects=custom_objects)
        except OSError:
            print(f"Couldn't open model {ckp}")
            continue


        # Iterate over data
        for batch_i, batch in enumerate(mura_data.selected_test_loader):
            print(f"Model {ckp}")
            for example_i, example in enumerate(zip(batch[0], batch[1])):
                print(f"Processing image {example_i}")
                # Image Preprocessing
                # img_raw is the rescaled image (img_width, img_height) withouth preprocessing
                image_raw, label_ndarray = example[0].numpy(), example[1]  # [0.0, 255.]
                image = mura_data.preprocess_img(image_raw, model_name="inception")
                # image = image.numpy()  # image is the preprocessed image in numpy: [-1, 1]
                # TODO: image and image_array same?
                image_array = tf.keras.utils.img_to_array(image)  # image_array is the preprocessed image in numpy array
                image_batch = tf.expand_dims(image_array, axis=0)  # image_batch is the batched image tensor

                true_y = np.argmax(label_ndarray)
                pred_y = np.argmax(classifier.predict(image_batch))

                # Sub-Directory Name
                sub_dir = "correct" if pred_y == true_y else "wrong"
                file_name = f"index({example_i})_T({true_y})_P({pred_y}).png"

                if exists(f"{RESULTS_PATH}/{ckp}/{file_name}"):
                    print(f"{RESULTS_PATH}/{ckp}/{file_name} already exists.")
                    continue

                gradcam_plus_plus, attention = get_gradcam_plus_plus(explainer_grad_cam_plus_plus, image_array, true_y,
                                                                     return_attention_map=True)

                # counterfactual with GAN
                counterfactual_list, classifications = predict_imgs(classifier, g_NP, g_PN, image_batch, true_y,
                                                                    num_images=1, attention=attention)

                # diff_map_list = get_diff_maps_for_imgs(image_raw, counterfactual_list)

                counterfactuals_lime = []
                counterfactuals_gradcam = []
                counterfactuals_gradcam_plus_plus = []
                for counterfactual, classification in zip(counterfactual_list, classifications):
                    counterfactuals_gradcam.append(
                        get_tf_explain_gradcam(explainer_grad_cam, counterfactual, classifier,
                                               classification,
                                               sub_dir, file_name))

                    counterfactuals_gradcam_plus_plus.append(

                        get_gradcam_plus_plus(explainer_grad_cam_plus_plus, counterfactual, true_y)
                    )

                    """counterfactuals_lime.append(
                        get_lime(explainer_lime, classifier, counterfactual, sub_dir, file_name))"""

                # tf-explain GradCAM
                tf_explain_gradcam = get_tf_explain_gradcam(explainer_grad_cam, image_array, classifier, true_y,
                                                            sub_dir, file_name)

                # lime
                # lime_result = get_lime(explainer_lime, classifier, image, sub_dir, file_name)
                lime_result = None

                # heatmap with keras example GradCAM
                # keras_gradcam = get_keras_gradcam(classifier, image_batch, image_raw, sub_dir, file_name)

                img_list = [image_raw, tf_explain_gradcam, gradcam_plus_plus,
                            counterfactual_list[0], counterfactuals_gradcam[0], counterfactuals_gradcam_plus_plus[0]]

                """stacked_imgs = np.hstack(
                    (image_raw, tf_explain_gradcam, lime_result, counterfactual_list[0], counterfactual_list[1]))
        
                ski.io.imsave(f"xai_results/{file_name}", stacked_imgs)"""

                r = 2
                c = 3
                titles = ['Original', 'GradCam', 'GradCam++',
                          'Counterfactual 1', 'GradCam', 'GradCam++']

                fig, axs = plt.subplots(r, c, figsize=(15, 15))
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i, j].imshow(img_list[cnt][:, :, 0], vmin=np.min(img_list[cnt]),
                                         vmax=np.max(img_list[cnt]))  # cmap='gray'
                        if i == 0:
                            if j == 0:
                                axs[i, j].set_title(f'{titles[cnt]} (T: {true_y} | P: {pred_y})')
                            else:
                                axs[i, j].set_title(f'{titles[cnt]}')
                        else:
                            if j == 0:
                                axs[i, j].set_title(f'{titles[cnt]} (P: {classifications[j]})')
                            else:
                                axs[i, j].set_title(f'{titles[cnt]}')

                        # axs[i, j].set_title(f'{titles[j]} ({correct_classification[cnt]})')
                        axs[i, j].axis('off')
                        cnt += 1
                if not os.path.exists(f"{RESULTS_PATH}/{ckp}"):
                    os.mkdir(f"{RESULTS_PATH}/{ckp}")
                fig.savefig(f"{RESULTS_PATH}/{ckp}/{file_name}")
