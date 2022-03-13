from typing import List

from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import cv2
from skimage.io import imread
from sklearn.utils import shuffle
import numpy as np
from keras.utils.all_utils import Sequence
from albumentations import (
    Compose, HorizontalFlip,
    RandomBrightness, RandomContrast, RandomGamma)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
])


class MuraGeneratorDataset():
    def __init__(self, config):
        self.config = config
        self.augment_train = AUGMENTATIONS_TRAIN
        self.preprocess_img = preprocess_img
        self.train_loader, self.valid_loader, self.test_loader, self.raw_valid_loader, self.train_y, self.test_y = get_mura_loaders(
            config,
            batch_size=self.config["train"]["batch_size"])


class MuraGenerator(Sequence):

    def __init__(self, config, image_filenames, labels, batch_size, transform, preprocess=True):
        self.config = config
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.t = transform
        self.preprocess = preprocess

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        x = []
        for file in batch_x:
            img = imread(file)
            if self.t:
                img = self.t(image=img)["image"]
            if len(img.shape) < 3:
                img = tf.expand_dims(img, axis=-1)
            if img.shape[-1] != 3:
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.resize_with_pad(img, self.config["data"]["image_height"], self.config["data"]["image_width"])
            if self.preprocess:
                img = preprocess_img(img, self.config["model"]["name"])
            x.append(img)
        x = tf.stack(x)
        y = np.array(batch_y)
        return x, y


class MuraValidDataGenerator(Sequence):

    def __init__(self, config, image_filenames, labels, batch_size=1):
        self.config = config
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.pos_image_paths = [filename for filename in image_filenames if
                                "positive" in filename]
        self.neg_image_paths = [filename for filename in image_filenames if
                                "negative" in filename]

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        if idx % 2 == 0:
            batches = self.neg_image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
            y = np.array([1, 0])
        else:
            batches = self.pos_image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
            y = np.array([0, 1])
        xs = []
        ys = []
        for file in batches:
            img = imread(file)
            if len(img.shape) < 3:
                img = tf.expand_dims(img, axis=-1)
            if img.shape[-1] != 3:
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.resize_with_pad(img, self.config["data"]["image_height"], self.config["data"]["image_width"])
            xs.append(img)
            ys.append(y)
        xs = tf.stack(xs)
        ys = np.reshape(ys, (-1, 2))
        return xs, ys


def get_mura_loaders(config, batch_size=32):
    # To get the filenames for a task
    def filenames(parts: List[str], train=True):
        root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        # root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            # csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            # csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                    str(x, encoding='utf-8').strip().split('/')[2] in parts]

        # imgs= [x.replace("/", "\\") for x in imgs]
        labels = [x.split('_')[-1].split('/')[0] for x in imgs]
        return imgs, labels

    parts = config["train"]["body_parts"]  # part to work with

    train_x, train_y = filenames(parts=parts)  # train data
    test_x, test_y = filenames(parts=parts, train=False)  # test data
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2,
                                                          random_state=42)  # split train and valid data

    train_x, train_y = to_categorical(train_x, train_y)
    valid_x, valid_y = to_categorical(valid_x, valid_y)
    test_x, test_y = to_categorical(test_x, test_y)

    train_gen = MuraGenerator(config, train_x, train_y, batch_size, AUGMENTATIONS_TRAIN)
    valid_gen = MuraGenerator(config, valid_x, valid_y, batch_size, None)
    test_gen = MuraGenerator(config, test_x, test_y, batch_size, None)
    test_raw_gen = MuraValidDataGenerator(config, test_x, test_y)

    train_amount = train_y.count('positive') + train_y.count('negative')
    valid_amount = valid_y.count('positive') + valid_y.count('negative')
    test_amount = test_y.count('positive') + test_y.count('negative')

    print(f"Train data amount: {train_amount}")
    print(f"Valid data amount: {valid_amount}")
    print(f"Test data amount: {test_amount}")

    return train_gen, valid_gen, test_gen, test_raw_gen, train_y, test_y


def preprocess_img(img, model_name="inception"):
    if model_name == 'densenet':
        return tf.cast(img, tf.float32) / 255.  # between 0 and 1
    else:  # Imagenet
        return tf.cast(img, tf.float32) / 127.5 - 1.  # between -1 and 1


def to_categorical(x, y):
    y = [0 if x == 'negative' else 1 for x in y]
    y = keras.utils.to_categorical(y)
    x, y = shuffle(x, y)
    return x, y
