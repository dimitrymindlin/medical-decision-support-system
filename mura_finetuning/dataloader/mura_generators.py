from typing import List

from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from skimage.io import imread
from sklearn.utils import shuffle
import numpy as np
from keras.utils.all_utils import Sequence
from albumentations import (
    Compose, HorizontalFlip, CLAHE,
    RandomBrightness, RandomContrast, RandomGamma)


class MuraGeneratorDataset():
    def __init__(self, config):
        self.config = config
        self.AUGMENTATIONS_TRAIN = Compose([
            CLAHE(always_apply=True),
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
        ])
        self.AUGMENTATIONS_TEST = Compose([
            CLAHE(always_apply=True)
        ])
        self.preprocess_img = preprocess_img
        self.train_loader, self.valid_loader, self.test_loader, self.selected_test_loader, self.train_y, self.test_y = get_mura_loaders(
            config,
            batch_size=self.config["train"]["batch_size"],
            aug_train=self.AUGMENTATIONS_TRAIN,
            aug_test=self.AUGMENTATIONS_TEST
        )


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

    def __init__(self, config, image_filenames, labels, batch_size=1, transform=None):
        self.config = config
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.t = transform
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
            if self.t:
                img = self.t(image=img)["image"]
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

class Test_img_data_generator(Sequence):
    """
    Take 15 positive self selected examples and try counterfactual generation
    """
    def __init__(self, batch_size, input_size=(512, 512), transform=None):
        self.input_size = input_size
        self.t = transform
        #root = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid/XR_WRIST/"
        root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid/XR_WRIST/'
        """
                                root + "patient11186/study3_positive/image3.png",
                                root + "patient11188/study1_positive/image1.png",
                                root + "patient11188/study1_positive/image2.png",
                                root + "patient11188/study1_positive/image3.png",
                                root + "patient11188/study1_positive/image4.png",
                                root + "patient11190/study1_positive/image1.png",
                                root + "patient11190/study1_positive/image2.png",
                                root + "patient11186/study2_positive/image2.png",
                                root + "patient11192/study1_positive/image1.png",
                                root + "patient11192/study1_positive/image2.png",
                                root + "patient11192/study1_positive/image3.png"
        """
        self.pos_image_paths = [root + "patient11186/study2_positive/image1.png",
                                root + "patient11186/study2_positive/image3.png",
                                root + "patient11186/study3_positive/image1.png",
                                root + "patient11186/study3_positive/image2.png",
                                root + "patient11205/study1_positive/image1.png",
                                root + "patient11205/study1_positive/image2.png",
                                root + "patient11205/study1_positive/image3.png",
                                root + "patient11267/study1_positive/image1.png"]
        self.batch_size = len(self.pos_image_paths)

    def __len__(self):
        return (np.ceil(len(self.pos_image_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # TODO: ONLY FOR BATCH SIZE OF 1
        batch_pos = self.pos_image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        y = np.array([0, 1])
        batches = [batch_pos]
        pos = []
        ys = []
        for i, batch in enumerate(batches):
            for file in batch:
                img = imread(file)
                if self.t:
                    img = self.t(image=img)["image"]
                if len(img.shape) < 3:
                    img = tf.expand_dims(img, axis=-1)
                if img.shape[-1] != 3:
                    img = tf.image.grayscale_to_rgb(img)
                img = tf.image.resize_with_pad(img, self.input_size[0], self.input_size[1])
                pos.append(img)
                ys.append(y)
        pos = tf.stack(pos)
        ys = np.reshape(ys, (-1, 2))
        return pos, ys


def get_mura_loaders(config, batch_size=32, aug_train=None, aug_test=None):
    # To get the filenames for a task
    def filenames(parts: List[str], train=True):
        #root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
        root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
        if train:
            #csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1_transformed/train_image_paths.csv"
            #csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1_transformed/train_image_paths.csv"
        else:
            #csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
            #csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1_transformed/valid_image_paths.csv"
            #csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1_transformed/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            imgs = [root + str(x, encoding='utf-8').strip().replace("MURA-v1.1", "MURA-v1.1_transformed") for x in d if
                    str(x, encoding='utf-8').strip().split('/')[2] in parts]
            #imgs = [img_path.replace("MURA-v1.1", "MURA-v1.1_transformed") for img_path in imgs]

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

    if not config["train"]["augmentation"]:
        aug_train = aug_test  # Just make CLAHE
    train_gen = MuraGenerator(config, train_x, train_y, batch_size, aug_train)
    valid_gen = MuraGenerator(config, valid_x, valid_y, batch_size, aug_test)
    test_gen = MuraGenerator(config, test_x, test_y, batch_size, aug_test)
    #test_raw_gen = MuraValidDataGenerator(config, test_x, test_y, transform=aug_test)
    seleted_test_gen = Test_img_data_generator(batch_size=batch_size, input_size=config["data"]["input_size"], transform=aug_test)

    print(f"Train data amount: {len(train_y)}")
    print(f"Valid data amount: {len(valid_y)}")
    print(f"Test data amount: {len(test_y)}")

    return train_gen, valid_gen, test_gen, seleted_test_gen, train_y, test_y


def preprocess_img(img, model_name="inception"):
    if model_name == 'densenet':
        return tf.keras.applications.densenet.preprocess_input(img)  # between 0 and 1
    else:  # Inception
        return tf.keras.applications.inception_v3.preprocess_input(img)  # between -1 and 1


def to_categorical(x, y):
    y = [0 if x == 'negative' else 1 for x in y]
    y = keras.utils.to_categorical(y)
    x, y = shuffle(x, y)
    return x, y


"""def show_augmentations():
    albumentation_list = [
        Equalize(always_apply=True),
        CLAHE(always_apply=True),
        HorizontalFlip(p=1),
        RandomContrast(limit=0.6, p=1),
        RandomGamma(gamma_limit=(60, 180), p=1),
        RandomBrightness(limit=0.5, p=1),
    ]
    root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
    chosen_image = imread(root + 'MURA-v1.1/train/XR_WRIST/patient00136/study1_positive/image3.png')
    img_matrix_list = []
    for aug_type in albumentation_list:
        img = aug_type(image=chosen_image)['image']
        img_matrix_list.append(img)
    img_3d = tf.expand_dims(chosen_image, axis=-1)
    img = tf.image.resize_with_pad(img_3d, 512, 512)
    img_matrix_list.append(img)

    img_matrix_list.insert(0, chosen_image)

    titles_list = ["Original", "Equalize", "CLAHE", "Horizontal Flip", "Random Contrast", "Random Gamma",
                   "RandomBrightness", "Resizing"]

    ncols = 3
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=True)
    fig.suptitle("Augmentation", fontsize=30)
    # fig.subplots_adjust(wspace=0.3)
    # fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, titles_list)):
        myaxes[i // ncols][i % ncols].imshow(img, cmap='gray')
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()


show_augmentations()
show_augmentations()
show_augmentations()"""
