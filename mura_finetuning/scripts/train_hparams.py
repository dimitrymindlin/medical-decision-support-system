#!/usr/bin/env python3

import tensorflow as tf
from datetime import datetime

from sklearn.utils import compute_class_weight

from configs.hp_config import hp_config

from models.mura_model import WristPredictNetHP

import keras_tuner as kt
import sys
import numpy as np

from mura_finetuning.dataloader.mura_generators import MuraGeneratorDataset
from utils.training_utils import get_model_name_from_cli_to_config, print_running_on_gpu


TIMESTAMP = datetime.now().strftime("%Y-%m-%d--%H.%M")
MODEL_NAME = get_model_name_from_cli_to_config(sys.argv, hp_config)
TRAIN_MODE = hp_config["train"]["prefix"]  # one of: [pretrain, finetune, frozen]
TF_LOG_DIR = f'tensorboard_logs/logs_{TRAIN_MODE}/{TRAIN_MODE}_{MODEL_NAME}/' + TIMESTAMP + "/"
print_running_on_gpu(tf)

# Dataset
mura_data = MuraGeneratorDataset(hp_config)
y_integers = np.argmax(mura_data.train_y, axis=1)
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_integers),
                                     y=y_integers)
d_class_weights = dict(zip(np.unique(y_integers), class_weights))



# Model Definition
def build_model(hp):
    # Model Definition
    weight_regularisation_value = hp.Choice("weight_regularisation", [0.0004, 0.002])
    print(f"weight regu value: {weight_regularisation_value}")
    model = WristPredictNetHP(hp_config, hp=hp, weight_regularisation_value=weight_regularisation_value)
    # Training params
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(hp_config["data"]["class_names"]),
                               from_logits=False, name="auc")
    bin_accuracy = tf.keras.metrics.BinaryAccuracy(name="bin_accuracy")

    learning_rate = hp.Choice('learning_rate', [0.0001, 0.00001])

    # if optimizer == "adam":
    optimizer = tf.optimizers.Adam(learning_rate)
    """elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)"""

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[auc, bin_accuracy]
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_epochs=30,
    directory=TF_LOG_DIR)

tuner.search(mura_data.train_loader,
             validation_data=mura_data.valid_loader,
             epochs=hp_config["train"]["epochs"],
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=hp_config['train']['early_stopping_patience']),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                                             factor=hp_config["train"]["factor_learning_rate"],
                                                             patience=hp_config["train"]["patience_learning_rate"],
                                                             min_delta=0.001,
                                                             verbose=1,
                                                             min_lr=hp_config["train"]["min_learning_rate"]),
                        tf.keras.callbacks.TensorBoard(TF_LOG_DIR)],
             )
