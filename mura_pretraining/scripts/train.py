#!/usr/bin/env python3

import tensorflow as tf
#import tensorflow_addons as tfa
from datetime import datetime
from configs.mura_pretraining_config import mura_config
from mura_pretraining.dataloader.mura_dataset import MuraDataset
from mura_pretraining.model.mura_model import WristPredictNet
from utils.path_constants import PathConstants
import sys

from utils.training_utils import get_model_name_from_cli, print_running_on_gpu

config = mura_config
print_running_on_gpu(tf)
get_model_name_from_cli(sys.argv, config)
TF_LOG_DIR = f'{PathConstants.MURA_TENSORBOARD_PREFIX}/mura_{config["model"]["name"]}/' + datetime.now().strftime(
    "%Y-%m-%d--%H.%M")

# Model Definition
model = WristPredictNet(config, train_base=config['train']['train_base'])

# Training Params
optimizer = tf.keras.optimizers.Adam(config["train"]["learn_rate"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)
metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()
#metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]), threshold=config["test"]["F1_threshold"], average='macro')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy],
)

# Tensorboard Callback and config logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=1)

# Checkpoint Callback to only save best checkpoint
checkpoint_filepath = f'checkpoints/mura_{config["model"]["name"]}/' + datetime.now().strftime(
    "%Y-%m-%d--%H.%M") + '/cp.ckpt'

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
    factor=config['train']['factor_learning_rate'],
    patience=config['train']['patience_learning_rate'],
    mode="min",
    min_lr=config['train']['min_learning_rate'],
)

# Get Dataset
dataset = MuraDataset(config)
#dataset = load_cropped_ds(config)

# Class weights for training underrepresented classes
class_weight = dataset.train_classweights if config["train"]["use_class_weights"] else None

# Log Text like config and evaluation
config_matrix = [[k, str(w)] for k, w in config["train"].items()]
file_writer = tf.summary.create_file_writer(TF_LOG_DIR)
with file_writer.as_default():
    tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)
"""    images = dataset.ds_train.take(1)
    tfds.visualization.show_examples(images, dataset.ds_info)
    for image, label in images:
        tf.summary.image("Training examples", image, max_outputs=len(image), step=0)
    #tf.summary.image("training data examples", images, max_outputs=len(images), step=0)
    images = dataset.ds_test.take(1)
    for image, label in images:
        tf.summary.image("Test examples", image, max_outputs=len(image), step=0)
    tf.summary.image("Test examples", images[0][0], max_outputs=len(image), step=0)

exit()"""
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
    tf.summary.text(f"mura_evaluation", tf.convert_to_tensor(result_matrix), step=0)
