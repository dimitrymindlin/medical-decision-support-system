from datetime import datetime

from tensorflow import keras
import tensorflow as tf
import cv2
from skimage.transform import resize
import tensorflow_addons as tfa
from skimage.io import imread
from sklearn.utils import shuffle
from datetime import datetime
import numpy as np
from keras.utils.all_utils import Sequence

# Get Dataset
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from configs.direct_training_config import direct_training_config as config
from mura_pretraining.dataloader import MuraDataset

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

def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

model_name = "inception"
timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
TF_LOG_DIR = f'kaggle/kaggle_{model_name}/' + timestamp + "/"
checkpoint_filepath = f'checkpoints/kaggle_{model_name}/' + timestamp + '/cp.ckpt'

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




# **Creating data generator for training and testiing with augmentation:**

# In[9]:


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
            img = resize(img, (224, 224, 3))
            img = crop_center(img, 224, 224)
            img = tf.image.resize_with_pad(img, 224, 224)
            x.append(img)
        x = np.array(x) / 255.0
        y = np.array(batch_y)
        return x, y


train_dir = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train"
validation_dir = '../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid'

part = 'XR_WRIST'  # part to work with
imgs, labels = filenames(part=part)  # train data
vimgs, vlabels = filenames(part=part, train=False)  # validation data

training_data = labels.count('positive') + labels.count('negative')
validation_data = vlabels.count('positive') + vlabels.count('negative')

y_data = [0 if x == 'positive' else 1 for x in labels]
y_data = keras.utils.to_categorical(y_data)
y_data_valid = [0 if x == 'positive' else 1 for x in vlabels]
y_data_valid = keras.utils.to_categorical(y_data_valid)

from sklearn.utils.class_weight import compute_class_weight

y_integers = np.argmax(y_data, axis=1)

class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_integers),
                                     y=y_integers
                                     )
d_class_weights = dict(zip(np.unique(y_integers), class_weights))

batch_size = 32
imgs, y_data = shuffle(imgs, y_data)
my_training_batch_generator = My_Custom_Generator(imgs, y_data, batch_size, AUGMENTATIONS_TRAIN)
my_validation_batch_generator = My_Custom_Generator(vimgs, y_data_valid, batch_size, AUGMENTATIONS_TEST)

# Tensorboard Callback and config logging
my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                    # Callback to save the Keras model or model weights at some frequency.
                                    monitor='val_accuracy',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    save_freq='epoch'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                      # Reduce learning rate when a metric has stopped improving.
                                      factor=0.1,
                                      patience=3,
                                      min_delta=0.001,
                                      verbose=1,
                                      min_lr=0.000000001),
    keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
                                histogram_freq=1,
                                write_graph=True,
                                write_images=False,
                                update_freq='epoch',
                                profile_batch=30,
                                embeddings_freq=1,
                                embeddings_metadata=None
                                ),
    keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                  patience=5,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                                  )
]

base_model = keras.applications.InceptionV3(
    input_shape=(224, 224, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top

# Create a new model on top
input_image = keras.layers.Input((224, 224, 3))
x = tf.keras.applications.inception_v3.preprocess_input(input_image)  # Normalisation to [0,1]
x = base_model(x)

# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)  ##### <-
# x=keras.layers.Flatten()(x)

x = keras.layers.Dense(1024)(x)  ###
x = keras.layers.Activation(activation='relu')(x)  ###
x = keras.layers.Dropout(0.5)(x)  ###
x = keras.layers.Dense(256)(x)
x = keras.layers.Activation(activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(2)(x)
out = keras.layers.Activation(activation='softmax')(x)

model = keras.Model(inputs=input_image, outputs=out)

metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)

metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                threshold=config["test"]["F1_threshold"], average='macro')
kappa = tfa.metrics.CohenKappa(num_classes=2)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=["accuracy", metric_auc, metric_f1, kappa])

# Model Training
# model.load_weights("checkpoints/kaggle_inception/2022-02-21--15.47/cp.ckpt")
history = model.fit_generator(generator=my_training_batch_generator,
                              epochs=40,
                              verbose=1,
                              class_weight=d_class_weights,  # d_class_weights
                              validation_data=my_validation_batch_generator,
                              callbacks=my_callbacks)

print("Kaggel Test Evaluation")
print(history.history.keys())

m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
y_pred = model.predict(my_validation_batch_generator)

yp2 = np.argmax(y_pred, axis=1)
ya2 = np.argmax(y_data_valid, axis=1)
print(y_pred.shape, y_data_valid.shape)
m.update_state(ya2, yp2)
print('Final result: ', m.result().numpy())

vy_data2 = np.argmax(y_data_valid, axis=1)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(vy_data2, yp2)
print(cm)

print(classification_report(vy_data2, yp2))

y_pred = model.predict(my_training_batch_generator)

yp3 = np.argmax(y_pred, axis=1)
y_true3 = np.argmax(y_data, axis=1)

cm2 = confusion_matrix(y_true3, yp3)
print(cm2)

print(classification_report(y_true3, yp3))
