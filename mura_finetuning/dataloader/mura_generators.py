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
    ToFloat, ShiftScaleRotate
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255)
])
AUGMENTATIONS_TEST = Compose([
    # CLAHE(p=1.0, clip_limit=2.0),
    ToFloat(max_value=255)
])


class MuraGeneratorDataset():
    def __init__(self):
        self.preprocess_img = preprocess_img
        self.augment_train = AUGMENTATIONS_TRAIN
        self.augment_valid = AUGMENTATIONS_TEST
        self.train_loader, self.valid_loader, self.y_data, self.y_data_valid = get_mura_loaders()
        _, self.raw_valid_loader, _, _ = get_mura_loaders(preprocess=False)

class MuraGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, transform, preprocess=True):
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
            if self.preprocess:
                img = self.t(image=img)["image"]
                img = preprocess_img(img)
            x.append(img)
        x = tf.stack(x)
        y = np.array(batch_y)
        return x, y


def get_mura_loaders(preprocess=True):
    # To get the filenames for a task
    def filenames(part, train=True):
        # root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            # csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        else:
            # csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]
            else:
                imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                        str(x, encoding='utf-8').strip().split('/')[2] == part]

        # imgs= [x.replace("/", "\\") for x in imgs]
        labels = [x.split('_')[-1].split('/')[0] for x in imgs]
        return imgs, labels

    part = 'XR_WRIST'  # part to work with
    imgs, labels = filenames(part=part)  # train data
    vimgs, vlabels = filenames(part=part, train=False)  # validation data

    train_amount = labels.count('positive') + labels.count('negative')
    valid_amount = vlabels.count('positive') + vlabels.count('negative')

    print(f"Train data amount: {train_amount}")
    print(f"Valid data amount: {valid_amount}")

    y_data = [0 if x == 'negative' else 1 for x in labels]
    y_data = keras.utils.to_categorical(y_data)
    y_data_valid = [0 if x == 'negative' else 1 for x in vlabels]
    y_data_valid = keras.utils.to_categorical(y_data_valid)

    batch_size = 32
    imgs, y_data = shuffle(imgs, y_data)
    train_gen = MuraGenerator(imgs, y_data, batch_size, AUGMENTATIONS_TRAIN, preprocess)
    valid_gen = MuraGenerator(vimgs, y_data_valid, batch_size, AUGMENTATIONS_TEST, preprocess)

    return train_gen, valid_gen, y_data, y_data_valid


def preprocess_img(img):
    if len(img.shape) < 3:
        img = tf.expand_dims(img, axis=-1)
    if img.shape[-1] != 3:
        img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize_with_pad(img, 224, 224)
    return tf.cast(img, tf.float32) / 127.5 - 1.
