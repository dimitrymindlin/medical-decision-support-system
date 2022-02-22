import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import resource
from mura_finetuning.dataloader.mura_wrist_tfds import MuraWristImages
from mura_pretraining.dataloader.mura_tfds import MuraImages
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate
)
import cv2


class MuraDataset():
    def __init__(self, config, only_wrist_data=False):
        self.config = config
        low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
        dataset = 'MuraImages' if not only_wrist_data else 'MuraWristImages'
        (train, test), info = tfds.load(
            dataset,
            split=['train','test'],
            shuffle_files=True,
            as_supervised=True,
            download=self.config["dataset"]["download"],
            with_info=True,
        )
        self.ds_info = info

        self.train_classweights = self._calc_class_weights_for_ds(train)
        self.valid_classweights = self._calc_class_weights_for_ds(test)
        self.ds_train = self._build_train_pipeline(train)
        #self.ds_val = self._build_test_pipeline(validation)
        self.ds_test = self._build_test_pipeline(test)

    def _build_train_pipeline(self, ds):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(self.ds_info.splits['train'].num_examples)
        ds = ds.batch(self.config['train']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_test_pipeline(self, ds):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.config['test']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _calc_class_weights_for_ds(self, ds):
        vals = np.unique(np.fromiter(ds.map(lambda x, y: y), float), return_counts=True)

        label_distribution = dict()
        for class_num, count in zip(*vals):
            label_distribution[class_num] = count

        total = label_distribution[0] + label_distribution[1]
        print(f"Negative: {label_distribution[0]}")
        print(f"Positive: {label_distribution[1]}")
        print(f"Total: {total}")
        weight_for_0 = (1 / label_distribution[0]) * (total / 2.0)
        weight_for_1 = (1 / label_distribution[1]) * (total / 2.0)

        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

        return {0: weight_for_0, 1: weight_for_1}

    def preprocess(self, image, label):
        height = self.config['data']['image_height']
        width = self.config['data']['image_width']
        image = tf.image.resize_with_pad(tf.convert_to_tensor(image), height, width)
        label = tf.one_hot(tf.cast(label, tf.int32), 2)
        label = tf.cast(label, tf.float32)
        if self.config["train"]["augmentation"]:
            image = tf.numpy_function(func=aug_fn, inp=[image, self.config["data"]["image_height"]], Tout=tf.float32)
        return tf.cast(image, tf.float32), label  # normalize pixel values

    def benchmark(self):
        tfds.benchmark(self.ds_train, batch_size=self.config['train']['batch_size'])


"""def load_cropped_ds(config):
    ds = MuraDataset(config)
    for idx, (image, label) in enumerate(ds.ds_train):
        #print(ds.ds_train[idx][0].shape())
        ds.ds_train[idx][0] = crop_image(image)
        print(ds.ds_train[idx][0].shape())
        print("Dimi")"""

transforms = Compose([
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255)
])


def aug_fn(image, img_size):
    data = {"image": image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img / 255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    return aug_img
