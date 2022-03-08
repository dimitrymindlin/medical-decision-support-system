from typing import List

from tensorflow import keras
import tensorflow as tf
import cv2
from skimage.io import imread
from sklearn.utils import shuffle
import numpy as np
from keras.utils.all_utils import Sequence
from albumentations import (
    Compose, HorizontalFlip,
    RandomBrightness, RandomContrast, RandomGamma,
    ShiftScaleRotate
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
])

class MuraGeneratorDataset():
    def __init__(self, config):
        self.config = config
        self.preprocess_img = preprocess_img
        self.augment_train = AUGMENTATIONS_TRAIN
        self.train_loader, self.valid_loader, self.raw_valid_loader, self.y_data, self.y_data_valid = get_mura_loaders(
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
                img = preprocess_img(img)
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
            y = np.array([1,0])
        else:
            batches = self.pos_image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
            y = np.array([0,1])
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

def get_mura_loaders(config, batch_size=32, preprocess=True):
    # To get the filenames for a task
    def filenames(parts: List[str], train=True):
        root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        #root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                    str(x, encoding='utf-8').strip().split('/')[2] in parts]

        # imgs= [x.replace("/", "\\") for x in imgs]
        labels = [x.split('_')[-1].split('/')[0] for x in imgs]
        return imgs, labels

    parts = config["train"]["body_parts"]  # part to work with
    imgs, labels = filenames(parts=parts)  # train data
    vimgs, vlabels = filenames(parts=parts, train=False)  # validation data

    train_amount = labels.count('positive') + labels.count('negative')
    valid_amount = vlabels.count('positive') + vlabels.count('negative')

    print(f"Train data amount: {train_amount}")
    print(f"Valid data amount: {valid_amount}")

    y_data = [0 if x == 'negative' else 1 for x in labels]
    y_data = keras.utils.to_categorical(y_data)
    y_data_valid = [0 if x == 'negative' else 1 for x in vlabels]
    y_data_valid = keras.utils.to_categorical(y_data_valid)

    imgs, y_data = shuffle(imgs, y_data)
    train_gen = MuraGenerator(config, imgs, y_data, batch_size, AUGMENTATIONS_TRAIN, preprocess)
    valid_gen = MuraGenerator(config, vimgs, y_data_valid, batch_size, None, preprocess)
    valid_gen_raw = MuraValidDataGenerator(config, vimgs, y_data_valid)
    #valid_gen_raw = MuraGenerator(config, vimgs, y_data_valid, batch_size, None, preprocess=False)


    return train_gen, valid_gen, valid_gen_raw, y_data, y_data_valid


def preprocess_img(img):
    return tf.cast(img, tf.float32) / 127.5 - 1.
