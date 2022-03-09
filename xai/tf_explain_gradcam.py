from utils.xai_utils import image_stack


def get_tf_explain_gradcam(explainer, image_array, model, class_index, sub_dir, file_name, save=False, raw_img=None):
    data = ([image_array], None)
    output_grad_cam = explainer.explain(data, model, class_index=int(class_index))
    if save and image_array:
        explainer.save(image_stack(raw_img, output_grad_cam), f"xai_results/GradCAM/{sub_dir}",
                       output_name=file_name)
    return output_grad_cam
