#!/usr/bin/env python3

import tensorflow as tf
# import tensorflow_addons as tfa
from datetime import datetime

from configs.mura_pretraining_config import mura_config
from mura_pretraining.dataloader.mura_dataset import MuraDataset
from mura_pretraining.model.hparams_mura_model import HparamsMuraModel
from utils.path_constants import PathConstants
import keras_tuner as kt
import sys

from utils.training_utils import get_model_name_from_cli, print_running_on_gpu

print_running_on_gpu(tf)
config = mura_config
get_model_name_from_cli(sys.argv, config)
TF_LOG_DIR = f"{PathConstants.MURA_TENSORBOARD_TUNING_PREFIX}/{config['model']['name']}_" + datetime.now().strftime(
    "%Y-%m-%d--%H.%M")

dataset = MuraDataset(config)


# Model Definition
def build_model(hp):
    model = HparamsMuraModel(config['model']['name'], config, hp)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                      from_logits=False)
    metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()
    """metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                    threshold=config["test"]["F1_threshold"], average='macro')"""

    # Optimizer and LR
    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    learning_rate = hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])
    if optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metric_auc, metric_bin_accuracy]  # metric_f1
    )
    return model


tuner = kt.Hyperband(
    build_model,
    objective='val_binary_accuracy',
    max_epochs=30,
    directory=TF_LOG_DIR)

tuner.search(dataset.ds_train,
             validation_data=dataset.ds_val,
             epochs=config["train"]["epochs"],
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=config['train']['early_stopping_patience']),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss",
                            factor=0.1,
                            patience=config['train']['patience_learning_rate'],
                            mode="min",
                            min_lr=config['train']['min_learning_rate'],
                        ),
                        tf.keras.callbacks.TensorBoard(TF_LOG_DIR)],
             )
