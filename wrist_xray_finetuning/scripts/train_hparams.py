#!/usr/bin/env python3

import tensorflow as tf
# import tensorflow_addons as tfa
from datetime import datetime
import sys

from configs.wrist_xray_config import wrist_xray_config
import keras_tuner as kt

from utils.path_constants import PathConstants
from utils.training_utils import print_running_on_gpu, get_model_name_from_cli_to_config
from wrist_xray_finetuning.dataloader import WristXrayDataset
from wrist_xray_finetuning.model.hparams_wrist_xray_model import HparamsWristXrayModel

config = wrist_xray_config
print_running_on_gpu(tf)
get_model_name_from_cli_to_config(sys.argv, config)
CPU_WEIGHT_PATH = f"../../checkpoints/mura_{config['model']['name']}/best/cp.ckpt"
GPU_WEIGHT_PATH = f"checkpoints/mura_{config['model']['name']}/best/cp.ckpt"

TF_LOGDIR_PATH = f'{PathConstants.WRIST_XRAY_TENSORBOARD_HPARAMS_PREFIX}/{config["model"]["name"]}_' + datetime.now().strftime(
    "%Y-%m-%d--%H.%M")

dataset = WristXrayDataset(config)


# Model Definition
def build_model(hp):
    model = HparamsWristXrayModel(config, hp)
    model.load_weights(GPU_WEIGHT_PATH)
    if hp.Boolean("extra_layers"):
        x = tf.keras.layers.Dropout(0.3)(model.layers[-2].output)  # Regularize with dropout
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    else:
        x = tf.keras.layers.Dropout(0.2)(model.layers[-2].output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=model.layers[-2].input, outputs=x)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                      from_logits=False)
    metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()
    """metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                    threshold=config["test"]["F1_threshold"], average='macro')"""

    # Optimizer and LR
    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    learning_rate = hp.Choice('learning_rate', [0.00001, 0.00005, 0.0001])
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


tuner = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("val_auc", direction="max"),
    directory=TF_LOGDIR_PATH)

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
                        tf.keras.callbacks.TensorBoard(TF_LOGDIR_PATH)],
             )
