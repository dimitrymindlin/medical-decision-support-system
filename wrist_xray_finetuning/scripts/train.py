#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime

from configs.wrist_xray_config import wrist_xray_config
from wrist_xray_finetuning.dataloader import WristXrayDataset
from wrist_xray_finetuning.model.wrist_xray_model import WristXrayDenseNet
import sys

config = wrist_xray_config
# set cli arguments
for arg in sys.argv:
    if arg == "--use_class_weights":
        config["train"]["use_class_weights"] = True
    elif arg == "--augmentation":
        config["train"]["augmentation"] = True

input_shape = (None,
               config['data']['image_height'],
               config['data']['image_width'],
               config['data']['image_channel'])

dataset = WristXrayDataset(config)

# Model Definition
train_base = config['train']['train_base']
model = WristXrayDenseNet(config, train_base=train_base).model()
if config['train']['use_mura_weights']:
    model.load_weights("checkpoint/chexnet/best/cp.ckpt")  # TODO: Adapt weights path

# TODO: Check if you want HPARAMS with Optimizer and LR
"""optimizer_name = hparams[HP_OPTIMIZER]
learning_rate = hparams[HP_L_RATE]
if optimizer_name == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif optimizer_name == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)"""

optimizer = tf.keras.optimizers.Adam(config["train"]["learn_rate"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)
metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()
metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]), threshold=config["test"]["F1_threshold"], average='macro')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy, metric_f1],  #
)

# Tensorboard Callback and config logging
log_dir = 'logs/wrist_xray/' + datetime.now().strftime("%Y-%m-%d--%H.%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

config_matrix = [[k, str(w)] for k, w in config["train"].items()]
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
    tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

# Checkpoint Callback to only save best checkpoint
checkpoint_filepath = 'checkpoints/wrist_xray/' + datetime.now().strftime("%Y-%m-%d--%H.%M") + '/cp.ckpt'
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
    patience=config['train']['early_stopping_patience'])

# Dynamic Learning Rate
dyn_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=config['train']['patience_learning_rate'],
    mode="min",
    min_lr=config['train']['min_learning_rate'],
)

# Class weights for training underrepresented classes
class_weight = None
if config["train"]["use_class_weights"]:
    class_weight = dataset.train_classweights

# Model Training
model.fit(
    dataset.ds_train,
    epochs=config["train"]["epochs"],
    validation_data=dataset.ds_val,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping, dyn_lr],
    class_weight=class_weight
)

# Model Test
model.load_weights(checkpoint_filepath)  # best
result = model.evaluate(
    dataset.ds_test,
    batch_size=config['test']['batch_size'],
    callbacks=[tensorboard_callback])

result = dict(zip(model.metrics_names, result))
print("Evaluation Result: ", result)
result_matrix = [[k, str(w)] for k, w in result.items()]
with file_writer.as_default():
    tf.summary.text("evaluation", tf.convert_to_tensor(result_matrix), step=0)
