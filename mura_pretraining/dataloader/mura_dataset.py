import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class MuraDataset():

    def __init__(self, config):
        self.config = config

        (train, validation, test), info = tfds.load(
            'MuraImages',
            split=['train[:80%]', 'train[80%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            download=config["dataset"]["download"],
            with_info=True,
        )
        self.ds_info = info
        #self.class_weights = info.metadata["class_weights"] # TODO: Calc class weights

        self.train_classweights = self._calc_class_weights_for_ds(train)
        self.ds_train = self._build_train_pipeline(train)
        self.ds_val = self._build_test_pipeline(validation)
        self.ds_test = self._build_test_pipeline(test)

    def _build_train_pipeline(self, ds):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(self.ds_info.splits['train'].num_examples)
        ds = ds.batch(self.config['train']['batch_size'])
        if self.config["train"]["augmentation"]:
            ds = ds.map(self.augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_test_pipeline(self, ds):
        ds = ds.map(
            self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.config['test']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _calc_class_weights_for_ds(self, ds):
        vals = np.unique(np.fromiter(ds.map(lambda x, y: y), float), return_counts=True)

        label_distribution = dict()
        for class_num, count in zip(*vals):
            label_distribution[class_num] = count

        total = label_distribution[0] + label_distribution[1]
        weight_for_0 = (1 / label_distribution[0]) * (total / 2.0)
        weight_for_1 = (1 / label_distribution[1]) * (total / 2.0)

        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

        return {0: weight_for_0, 1: weight_for_1}

    def preprocess(self, image, label):
        height = self.config['data']['image_height']
        width = self.config['data']['image_width']
        image = tf.image.resize_with_pad(image, height, width)
        return tf.cast(image, tf.float32) / 255., label  # normalize pixel values
    
    def augment_data(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image, label

    def benchmark(self):
        tfds.benchmark(self.ds_train, batch_size=self.config['train']['batch_size'])
