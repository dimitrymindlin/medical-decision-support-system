from mura import get_mura_ds_by_body_part
from utils.model_utils import get_preprocessing_by_model_name
from utils.training_utils import get_labels_from_tfds


class MuraDataset():
    def __init__(self, config):
        self.config = config
        self.A_B_dataset, self.A_B_dataset_val, self.A_B_dataset_test, self.len_dataset_train = get_mura_ds_by_body_part(
            'XR_WRIST',
            config["data"][
                "tfds_path"],
            config["train"][
                "batch_size"],
            config["data"][
                "image_height"],
            config["data"][
                "image_height"],
            special_normalisation=get_preprocessing_by_model_name(config))
        self.train_y = get_labels_from_tfds(self.A_B_dataset)
        self.valid_y = get_labels_from_tfds(self.A_B_dataset_val)
        self.test_y = get_labels_from_tfds(self.A_B_dataset_test)

