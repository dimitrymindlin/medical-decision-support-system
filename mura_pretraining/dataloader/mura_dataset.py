import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import resource
from mura_finetuning.dataloader.mura_wrist_tfds import MuraWristImages
from mura_pretraining.dataloader.mura_tfds import MuraImages


class MuraDataset():
    def __init__(self, config, only_wrist_data=False):
        self.config = config
        low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
        dataset = 'MuraImages' if not only_wrist_data else 'MuraWristImages'
        (train, validation, test), info = tfds.load(
            dataset,
            split=['train[:80%]', 'train[80%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            download=self.config["dataset"]["download"],
            with_info=True,
        )
        self.ds_info = info

        self.train_classweights = self._calc_class_weights_for_ds(train)
        self.valid_classweights = self._calc_class_weights_for_ds(validation)
        self.ds_train = self._build_train_pipeline(train)
        self.ds_val = self._build_test_pipeline(validation)
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
        print(f"Training negative: {label_distribution[0]}")
        print(f"Training positive: {label_distribution[1]}")
        print(f"Training total: {total}")
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
        print(label.shape)
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
