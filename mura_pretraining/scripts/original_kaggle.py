import tensorflow as tf

print("tf version: ", tf.__version__)
# tf.keras.__version__
print("tf.keras version: ", tf.keras.__version__)
import tensorflow.keras as keras

print("keras.__version__: ", keras.__version__)

import os
import pandas as pd
import cv2
from skimage.transform import resize
import keras
import tensorflow_addons as tfa
from skimage.io import imread
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import datetime
import numpy as np
from keras.utils.all_utils import Sequence


# ****Some utility functions****

# In[6]:


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


# To icrop a image from center
def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


# **Data augmentations**

# In[7]:


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

# **Plotting the augmentations**

# In[8]:


albumentation_list = [
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255)
]
"""root = '../input/mura-v11/'
chosen_image = imread(root + 'MURA-v1.1/train/XR_WRIST/patient07988/study1_negative/image3.png')
img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image=chosen_image)['image']
    img_matrix_list.append(img)
img = resize(chosen_image, (300, 300, 3))
img_matrix_list.append(img)
img_matrix_list.append(crop_center(img, 224, 224))

img_matrix_list.insert(0, chosen_image)

titles_list = ["Original", "Horizontal Flip", "Random Contrast", "Random Gamma", "RandomBrightness",
               "Shift Scale Rotate", "Resizing", "Cropping"]


def plot_multiple_img(img_matrix_list, title_list, ncols, main_title="Data Augmentation"):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=2, ncols=ncols, squeeze=True)
    fig.suptitle(main_title, fontsize=30)
    # fig.subplots_adjust(wspace=0.3)
    # fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()


plot_multiple_img(img_matrix_list, titles_list, ncols=4)"""


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
            img = resize(img, (300, 300, 3))
            img = crop_center(img, 224, 224)
            x.append(img)
        x = np.array(x) / 255.0
        y = np.array(batch_y)
        return x, y


# **Some information about data:**

# In[74]:


train_dir = "../input/mura-v11/MURA-v1.1/train"
validation_dir = '../input/mura-v11/MURA-v1.1/valid'
print("Total Training patients: ")
for i in os.listdir(train_dir):
    print('Total Training patients at {}:'.format(i), len(os.listdir("../input/mura-v11/MURA-v1.1/train/{}".format(i))))
print("\nTotal Validation patients: ")
for i in os.listdir(validation_dir):
    print('Total Validation patients at {}:'.format(i),
          len(os.listdir("../input/mura-v11/MURA-v1.1/valid/{}".format(i))))

# **Getting data using the utility functions**

# In[76]:


########################################
# One of the seven listed below:
"""
XR_ELBOW
XR_FINGER
XR_FOREARM
XR_HAND
XR_HUMERUS
XR_SHOULDER
XR_WRIST
"""
# training_bone = 'XR_HUMERUS'
########################################

part = 'XR_WRIST'  # part to work with
imgs, labels = filenames(part=part)  # train data
vimgs, vlabels = filenames(part=part, train=False)  # validation data

print('{} Training positive :'.format(part), labels.count('positive'), '\n', '{} Training negative :'.format(part),
      labels.count('negative'))
training_data = labels.count('positive') + labels.count('negative')
print("Total Training Data at {}: ".format(part), training_data)
print('\n')
print('{} Validation positive :'.format(part), vlabels.count('positive'), '\n', '{} Validation negative :'.format(part),
      vlabels.count('negative'))
validation_data = vlabels.count('positive') + vlabels.count('negative')
print("Total Validation Data: ", validation_data)

y_data = [0 if x == 'positive' else 1 for x in labels]
y_data = keras.utils.to_categorical(y_data)
vy_data = [0 if x == 'positive' else 1 for x in vlabels]
vy_data = keras.utils.to_categorical(vy_data)

# **Calculate class-weight to avoid class-imbalance :**

# In[11]:


from sklearn.utils.class_weight import compute_class_weight

y_integers = np.argmax(y_data, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

# **Create Training and Test daat generator**

# In[12]:


batch_size = 32
imgs, y_data = shuffle(imgs, y_data)
# vimgs, vy_data = shuffle(vimgs, vy_data)
my_training_batch_generator = My_Custom_Generator(imgs, y_data, batch_size, AUGMENTATIONS_TRAIN)
my_validation_batch_generator = My_Custom_Generator(vimgs, vy_data, batch_size, AUGMENTATIONS_TEST)

# **Training callbacks**

# In[13]:


part = 'XR_WRIST'
checkpoint_path = "MURA_model@{}.h5".format(str(part))
checkpoint_dir = os.path.dirname(checkpoint_path)
import json

json_log = open(str(part) + '_experiment_log_MURA.json', mode='wt', buffering=1)
my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
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
    keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch,
                            logs: json_log.write(json.dumps({'epoch': epoch,
                                                             'train_loss': logs['loss'],
                                                             'val_loss': logs['val_loss'],
                                                             'weights': ""}) + '\n'),
        # dict(model.get_weights()) - np.array(model.get_weights()).tolist()
        on_train_end=lambda logs: json_log.close()),
    keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
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

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint


# In[14]:


# np.array(model.get_weights()).tolist()


# **Create a model**

# In[16]:


# Inception=tf.keras.applications.InceptionV3(include_top=False,input_shape=(224,224,3))#InceptionResNetV2
# #for layer in Inception.layers[:4]:
# #  layer.trainable=False
# input_image=keras.layers.Input((224,224,3))
# x=Inception (input_image)

# #x=keras.layers.GlobalAveragePooling2D()(x)
# x=keras.layers.Flatten()(x)
# #x=keras.layers.Dense(1024)(x)
# #x=keras.layers.Activation(activation='relu')(x)
# #x= keras.layers.Dropout(0.5)(x)
# x=keras.layers.Dense(256)(x)
# x=keras.layers.Activation(activation='relu')(x)
# x= keras.layers.Dropout(0.5)(x)
# x=keras.layers.Dense(2)(x)
# out=keras.layers.Activation(activation='softmax')(x)

# model=keras.Model(inputs=input_image,outputs=out)
# model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
# print(model.summary())


# Second model try:
#

# In[17]:


##### Another model try: ########
base_model = keras.applications.InceptionV3(
    #     weights='imagenet',  # Load weights pre-trained on ImageNet.,
    input_shape=(224, 224, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top

# odl from original
# for layer in base_model.layers[:4]:
#  layer.trainable=False

# Freeze base model
# base_model.trainable = False

# Create a new model on top
input_image = keras.layers.Input((224, 224, 3))

# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(input_image)

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
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# *  *Training*

# In[18]:


history = model.fit_generator(generator=my_training_batch_generator,
                              steps_per_epoch=int(training_data // batch_size),
                              epochs=40,
                              verbose=1,
                              class_weight=d_class_weights,
                              validation_data=my_validation_batch_generator,
                              validation_steps=int(validation_data // batch_size),
                              callbacks=my_callbacks)

# In[33]:


# %load_ext tensorboard
# %tensorboard --logdir logs/fit
# !kill 4805


# In[26]:


# list all data in history
print(history.history.keys())
# history.history


# **summarize history for accuracy**

# In[27]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# **Extracting some log to csv and json forms**

# In[29]:


# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history)

# save to json:  
hist_json_file = 'history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# In[67]:


# model.get_weights()


# In[68]:


# convert the model.get_weights() dict to a pandas DataFrame:     
model_weights_df = pd.DataFrame(model.get_weights())

# save to json:  
model_weights_file = 'XR_WRIST-model_weights.json'
with open(hist_json_file, mode='w') as f:
    model_weights_df.to_json(f)

# or save to csv: 
model_weights_file = 'XR_WRIST-model_weights.csv'
with open(hist_csv_file, mode='w') as f:
    model_weights_df.to_csv(f)

# In[78]:


# from distutils.dir_util import copy_tree
# fromDirectory = '../kaggle/working'
# toDirectory = '../tempTry'
# copy_tree(fromDirectory,toDirectory)


# In[48]:


# path = 'MURA_model@XR_WRIST-2.h5' 
# model.save("MURA_model@XR_WRIST.h5")


# **Evaluate the performance by cohen's kappa score**

# In[54]:


# model.save('MURA_model@XR_WRIST.h5')
m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
# model=tf.keras.models.load_model(path)
y_pred = model.predict(my_validation_batch_generator)

yp2 = np.argmax(y_pred, axis=1)
ya2 = np.argmax(vy_data, axis=1)
print(y_pred.shape, vy_data.shape)
m.update_state(ya2, yp2)
print('Final result: ', m.result().numpy())

# In[55]:


vy_data2 = np.argmax(vy_data, axis=1)

from sklearn.metrics import confusion_matrix, classification_report

# **Confusion matrix for validation data:**

# In[58]:


cm = confusion_matrix(vy_data2, yp2)
print(cm)

# In[59]:


print(classification_report(vy_data2, yp2))

# **Confusion matrix for trainning data**

# In[62]:


y_pred = model.predict(my_training_batch_generator)

# In[63]:


yp3 = np.argmax(y_pred, axis=1)
y_true3 = np.argmax(y_data, axis=1)

# In[64]:


cm2 = confusion_matrix(y_true3, yp3)
print(cm2)
plt.show()

# In[65]:


print(classification_report(y_true3, yp3))

# In[ ]:
