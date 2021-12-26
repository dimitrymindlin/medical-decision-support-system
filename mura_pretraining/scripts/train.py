#!/usr/bin/env python3

import tensorflow as tf
#import tensorflow_addons as tfa # TODO INCLUDE f1 score on machine
from datetime import datetime

from configs.mura_pretraining_config import mura_config
from mura_pretraining.dataloader.mura_dataset import MuraDataset
from mura_pretraining.model.mura_model import MuraDenseNet
import sys

# set cli arguments
for arg in sys.argv:
    if arg == "--use_class_weights":
        mura_config["train"]["use_class_weights"] = True
    elif arg == "--augmentation":
        mura_config["train"]["augmentation"] = True

input_shape = (None,
    mura_config['data']['image_height'],
    mura_config['data']['image_width'],
    mura_config['data']['image_channel'])

dataset = MuraDataset(mura_config)


# Model Definition
model = MuraDenseNet(mura_config).model()

optimizer = tf.keras.optimizers.Adam(mura_config["train"]["learn_rate"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC',multi_label=True, num_labels=len(mura_config["data"]["class_names"]), from_logits=False)
metric_bin_accuracy= tf.keras.metrics.BinaryAccuracy()
metric_accuracy = tf.keras.metrics.Accuracy()
#metric_f1 = tfa.metrics.F1Score(num_classes=len(mura_config["data"]["class_names"]), threshold=mura_config["test"]["F1_threshold"], average='macro')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy, metric_accuracy], # metric_f1
)

# Tensorboard Callback and config logging
log_dir = 'logs/mura-pretraining/' + datetime.now().strftime("%Y-%m-%d--%H.%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

config_matrix = [[k, str(w)] for k, w in mura_config["train"].items()]
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
  tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

# Checkpoint Callback to only save best checkpoint
checkpoint_filepath = 'checkpoints/mura/' + datetime.now().strftime("%Y-%m-%d--%H.%M") + '/cp.ckpt'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=metric_auc.name,
    mode='max',
    save_best_only=False)

# Early Stopping if loss plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=mura_config['train']['early_stopping_patience'])

# Dynamic Learning Rate
dyn_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=mura_config['train']['patience_learning_rate'],
    mode="min",
    min_lr=mura_config['train']['min_learning_rate'],
)

# Class weights for training underrepresented classes
class_weight = None
if mura_config["train"]["use_class_weights"]:
    class_weight = dataset.train_classweights

# Model Training
model.fit(
    dataset.ds_train,
    epochs=mura_config["train"]["epochs"],
    validation_data=dataset.ds_val,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping, dyn_lr],
    class_weight=class_weight
)

#Model Test
model.load_weights(checkpoint_filepath) #best
result = model.evaluate(
    dataset.ds_test, 
    batch_size=mura_config['test']['batch_size'],
    callbacks=[tensorboard_callback])

result = dict(zip(model.metrics_names, result))
print("Evaluation Result: ", result)
result_matrix = [[k, str(w)] for k, w in result.items()]
with file_writer.as_default():
  tf.summary.text("evaluation", tf.convert_to_tensor(result_matrix), step=0)