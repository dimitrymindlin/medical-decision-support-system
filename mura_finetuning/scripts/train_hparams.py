#!/usr/bin/env python3

import tensorflow as tf
# import tensorflow_addons as tfa
from datetime import datetime

from configs.frozen_hp_config import frozen_hp_config as config
from models.finetuning_model import get_finetuning_model_from_pretrained_model_hp
from mura_pretraining.dataloader.mura_dataset import MuraDataset
from models.mura_model import get_mura_model
from utils.path_constants import PathConstants
import keras_tuner as kt
import sys

from utils.training_utils import get_model_name_from_cli_to_config, print_running_on_gpu

timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
print_running_on_gpu(tf)
model_name = get_model_name_from_cli_to_config(sys.argv, config)
GPU_WEIGHT_PATH = f"checkpoints/pre_{model_name}/best/cp.ckpt"
get_model_name_from_cli_to_config(sys.argv, config)
TF_LOG_DIR = f"{PathConstants.FROZEN_HP}/{model_name}_" + timestamp

# Dataset
dataset = MuraDataset(config, only_wrist_data=True)


# Model Definition
def build_model(hp):
    # Model Definition
    config["train"]["augmentation"] = hp.Boolean('augmentation')
    config["train"]["use_class_weights"] = hp.Boolean("use_class_weights")
    config["train"]["batch_size"] = hp.Choice("batch_size", [8, 32])
    model = get_mura_model(config, include_top=False)
    model.load_weights(GPU_WEIGHT_PATH).expect_partial()
    model = get_finetuning_model_from_pretrained_model_hp(model, hp)

    # Training params
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                               from_logits=False, name="auc")
    bin_accuracy = tf.keras.metrics.BinaryAccuracy(name="bin_accuracy")

    """metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                    threshold=config["test"]["F1_threshold"], average='macro')"""

    # Optimizer and LR
    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    learning_rate = hp.Choice('learning_rate', [0.01, 0.001, 0.0001])
    if optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

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
