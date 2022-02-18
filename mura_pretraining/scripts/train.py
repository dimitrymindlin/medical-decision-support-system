#!/usr/bin/env python3

import tensorflow as tf
from datetime import datetime
from configs.pretraining_config import pretraining_config as config
from mura_pretraining.dataloader.mura_dataset import MuraDataset
from models.mura_model import get_mura_model
from utils.path_constants import PathConstants
import sys

from utils.training_utils import get_model_name_from_cli_to_config, print_running_on_gpu

timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
print_running_on_gpu(tf)
model_name = get_model_name_from_cli_to_config(sys.argv, config)
TF_LOG_DIR = f'{PathConstants.PRETRAIN}/pre_{model_name}/' + timestamp + "/"
checkpoint_filepath = f'checkpoints/pre_{model_name}/' + timestamp + '/cp.ckpt'

# Model Definition
model = get_mura_model(config)

# Training Params
optimizer = tf.keras.optimizers.Adam(config["train"]["learn_rate"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)
metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy]
)

# Tensorboard Callback and config logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=1)

# Checkpoint Callback to only save best checkpoint
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

# Class weights for training underrepresented classes
class_weight = dataset.train_classweights if config["train"]["use_class_weights"] else None

# Log Text like config and evaluation
config_matrix = [[k, str(w)] for k, w in config["train"].items()]
file_writer = tf.summary.create_file_writer(TF_LOG_DIR)
with file_writer.as_default():
    tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

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
