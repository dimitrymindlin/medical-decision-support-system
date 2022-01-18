#!/usr/bin/env python3

import tensorflow as tf
# import tensorflow_addons as tfa
from datetime import datetime
import sys

from configs.wrist_xray_config import wrist_xray_config
import keras_tuner as kt

from utils.training_utils import print_running_on_gpu, get_model_name_from_cli
from wrist_xray_finetuning.dataloader import WristXrayDataset
from wrist_xray_finetuning.model.hparams_wrist_xray_model import HparamsWristXrayModel

config = wrist_xray_config
print_running_on_gpu(tf)
get_model_name_from_cli(sys.argv, config)

LOG_DIR = f'logs/wrist_xray_tuning_{config["model"]["name"]}_' + datetime.now().strftime("%Y-%m-%d--%H.%M")
config = wrist_xray_config

dataset = WristXrayDataset(config)


# Model Definition
def build_model(hp):
    model = HparamsWristXrayModel(config, hp).model()

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
    directory=LOG_DIR)

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
                        tf.keras.callbacks.TensorBoard(LOG_DIR)],
             )
