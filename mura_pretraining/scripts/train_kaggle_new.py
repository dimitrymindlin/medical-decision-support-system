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
from sklearn.utils.class_weight import compute_class_weight
# Get Dataset
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from configs.direct_training_config import direct_training_config as config
from mura_finetuning.dataloader.mura_generators import get_mura_data
from mura_pretraining.dataloader import MuraDataset


model_name = "inception"
timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
TF_LOG_DIR = f'kaggle/kaggle_new_{model_name}/' + timestamp + "/"
checkpoint_filepath = f'checkpoints/kaggle_new_{model_name}/' + timestamp + '/cp.ckpt'



training_data, validation_data, y_data, y_data_valid, my_training_batch_generator, my_validation_batch_generator = get_mura_data()

y_integers = np.argmax(y_data, axis=1)

class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_integers),
                                     y=y_integers
                                     )
d_class_weights = dict(zip(np.unique(y_integers), class_weights))

# Tensorboard Callback and config logging
my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                    # Callback to save the Keras model or model weights at some frequency.
                                    monitor='val_accuracy',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=True,
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

print("Train History")
print(history)
print("Kaggel Test Evaluation")
result = model.evaluate(my_validation_batch_generator)
for metric, value in zip(model.metrics_names, result):
    print(metric, ": ", value)

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
