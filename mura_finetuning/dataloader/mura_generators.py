from skimage.transform import resize
from tensorflow import keras
import tensorflow as tf
import cv2
from skimage.io import imread
from sklearn.utils import shuffle
import numpy as np
from keras.utils.all_utils import Sequence

def get_mura_data():
    # To get the filenames for a task
    def filenames(part, train=True):
        root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

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

    from albumentations import (
        Compose, HorizontalFlip, CLAHE, HueSaturationValue,
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

    class My_Custom_Generator(Sequence):

        def __init__(self, image_filenames, labels, batch_size, transform):
            self.image_filenames = image_filenames
            self.labels = labels
            self.batch_size = batch_size
            self.t = transform

        def __len__(self):
            return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

        def __getitem__(self, idx):
            batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
            x = []
            for file in batch_x:
                img = imread(file)
                img = self.t(image=img)["image"]
                if len(img.shape) < 3:
                    img = tf.expand_dims(img, axis=-1)
                if img.shape[-1] != 3:
                    img = tf.image.grayscale_to_rgb(img)
                img = tf.image.resize_with_pad(img, 224, 224)
                img = tf.cast(img, tf.float32) / 127.5 - 1.
                x.append(img)
            x = tf.stack(x)
            y = np.array(batch_y)
            return x, y

    train_dir = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train"
    validation_dir = '../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid'

    part = 'XR_WRIST'  # part to work with
    imgs, labels = filenames(part=part)  # train data
    vimgs, vlabels = filenames(part=part, train=False)  # validation data

    training_data = labels.count('positive') + labels.count('negative')
    validation_data = vlabels.count('positive') + vlabels.count('negative')

    y_data = [0 if x == 'negative' else 1 for x in labels]
    y_data = keras.utils.to_categorical(y_data)
    y_data_valid = [0 if x == 'negative' else 1 for x in vlabels]
    y_data_valid = keras.utils.to_categorical(y_data_valid)

    batch_size = 32
    imgs, y_data = shuffle(imgs, y_data)
    my_training_batch_generator = My_Custom_Generator(imgs, y_data, batch_size, AUGMENTATIONS_TRAIN)
    my_validation_batch_generator = My_Custom_Generator(vimgs, y_data_valid, batch_size, AUGMENTATIONS_TEST)

    return training_data, validation_data, y_data, y_data_valid, my_training_batch_generator, my_validation_batch_generator