from utils.xai_utils import image_stack
import skimage as ski


def get_lime(explainer, model, image, sub_dir, file_name, save=False, image_raw=None):
    output_lime = explainer.explain_instance(image.astype('double'), model.predict)
    output_lime_image, output_lime_mask = output_lime.get_image_and_mask(output_lime.top_labels[0],
                                                                         positive_only=True,
                                                                         hide_rest=False,
                                                                         min_weight=0.1,
                                                                         num_features=10)
    output_lime_boundaries = ski.segmentation.mark_boundaries(output_lime_image / 2 + 0.5, output_lime_mask)
    output_lime_boundaries *= 255
    if save and image_raw:
        output_lime_stack = image_stack(image_raw, output_lime_boundaries)
        ski.io.imsave(f"xai_results/lime/{sub_dir}/{file_name}", output_lime_stack)
    return output_lime_boundaries
