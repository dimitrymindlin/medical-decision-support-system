from datetime import datetime

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from configs.direct_training_config import direct_training_config as config
from mura_pretraining.dataloader import MuraDataset

model_name = "inception"
timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
TF_LOG_DIR = f'kaggle/kaggle_{model_name}/' + timestamp + "/"
checkpoint_filepath = f'checkpoints/kaggle_{model_name}/' + timestamp + '/cp.ckpt'

dataset = MuraDataset(config, only_wrist_data=True)
# Checkpoint Callback to only save best checkpoint

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
    keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR,
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
x = tf.keras.applications.inception_v3.preprocess_input(input_image)  # Normalisation to [-1,1]
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

metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=2, from_logits=False)


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=["accuracy", metric_auc])

# Class weights for training underrepresented classes
#class_weight = dataset.train_classweights if config["train"]["use_class_weights"] else None

# Model Training
# model.load_weights("checkpoints/kaggle_inception/2022-02-21--15.47/cp.ckpt")
history = model.fit(dataset.ds_train,
    epochs=config["train"]["epochs"],
    validation_data=dataset.ds_test,
    callbacks=my_callbacks,
    verbose=1,
    class_weight=None
)

print(history.history)
print("Kaggel Test Evaluation")
m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
y_true_2d = np.concatenate([y for x, y in dataset.ds_test], axis=0)
y_pred_2d = model.predict(dataset.ds_test)
y_pred_1d = np.argmax(y_pred_2d, axis=1)
y_true_1d = np.argmax(y_true_2d, axis=1)
print(y_pred_2d.shape, y_true_2d.shape)
m.update_state(y_true_1d, y_pred_1d)
print('Final result: ', m.result().numpy())
print("SKlearn kappa ", cohen_kappa_score(y_true_1d, y_pred_1d))
print(y_true_1d.shape)
print(y_pred_1d.shape)
cm = confusion_matrix(y_true_1d, y_pred_1d)
print(cm)
print(classification_report(y_true_1d, y_pred_1d))

print(history.history)
print("Test Evaluation")
m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
y_pred_valid_2d = model.predict(dataset.ds_test)
y_pred_valid_1d = np.argmax(y_pred_valid_2d, axis=1)
y_true_valid_2d = np.concatenate([y for x, y in dataset.ds_test], axis=0)
y_true_valid_1d = np.argmax(y_pred_valid_2d, axis=1)
print(y_pred_valid_2d.shape, y_true_valid_2d.shape)
m.update_state(y_true_valid_1d, y_pred_valid_1d)
print('Final result: ', m.result().numpy())
print("Sklearn kappa ", cohen_kappa_score(y_true_valid_1d, y_pred_valid_1d))
print(confusion_matrix(y_true_valid_1d, y_pred_valid_1d))
print(classification_report(y_true_valid_1d, y_pred_valid_1d))
print("Train Evaluation")
y_pred_train_2d = model.predict(dataset.ds_train)
y_pred_train_1d = np.argmax(y_pred_train_2d, axis=1)
y_true_train_1d = np.concatenate([y for x, y in dataset.ds_train], axis=0)
print(confusion_matrix(y_pred_train_1d, y_pred_train_1d))
print(classification_report(y_pred_train_1d, y_true_train_1d))
print("Sklearn kappa ", cohen_kappa_score(y_true_train_1d, y_pred_train_1d))
