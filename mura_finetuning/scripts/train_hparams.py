#!/usr/bin/env python3

import tensorflow as tf
# import tensorflow_addons as tfa
from datetime import datetime

from sklearn.utils import compute_class_weight

from configs.finetuning_hp_config import finetuning_hp_config

from models.finetuning_model import get_finetuning_model_from_pretrained_model_hp
from models.mura_model import WristPredictNet

import keras_tuner as kt
import sys
import numpy as np

from mura_finetuning.dataloader.mura_generators import MuraGeneratorDataset
from utils.training_utils import get_model_name_from_cli_to_config, print_running_on_gpu


TIMESTAMP = datetime.now().strftime("%Y-%m-%d--%H.%M")
MODEL_NAME = get_model_name_from_cli_to_config(sys.argv, finetuning_hp_config)
TRAIN_MODE = finetuning_hp_config["train"]["prefix"]  # one of: [pretrain, finetune, frozen]
ckp_stage = finetuning_hp_config["train"]["checkpoint_stage"]
ckp_name = finetuning_hp_config['train']['checkpoint_name']
PRETRAINED_CKP_PATH = f"checkpoints/{ckp_stage}_{MODEL_NAME}/{ckp_name}/cp.ckpt"
TF_LOG_DIR = f'tensorboard_logs/logs_{TRAIN_MODE}/{TRAIN_MODE}_{MODEL_NAME}/' + TIMESTAMP + "/"
checkpoint_path_name = f'checkpoints/{TRAIN_MODE}_{MODEL_NAME}/' + TIMESTAMP + '/cp.ckpt'
checkpoint_path = f'checkpoints/{TRAIN_MODE}_{MODEL_NAME}/' + TIMESTAMP + '/'

print_running_on_gpu(tf)

# Dataset
mura_data = MuraGeneratorDataset(finetuning_hp_config)
y_integers = np.argmax(mura_data.train_y, axis=1)
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_integers),
                                     y=y_integers)
d_class_weights = dict(zip(np.unique(y_integers), class_weights))



# Model Definition
def build_model(hp):
    # Model Definition
    pre_model = WristPredictNet(finetuning_hp_config).model()
    model = get_finetuning_model_from_pretrained_model_hp(pre_model, finetuning_hp_config, hp)
    print(f"Loading pretrained from {finetuning_hp_config['train']['checkpoint_stage']} for {finetuning_hp_config['train']['prefix']}.")
    model.load_weights(PRETRAINED_CKP_PATH)

    # Training params
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(finetuning_hp_config["data"]["class_names"]),
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
             epochs=finetuning_hp_config["train"]["epochs"],
             class_weights=d_class_weights,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=finetuning_hp_config['train']['early_stopping_patience']),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                                          factor=finetuning_hp_config["train"]["factor_learning_rate"],
                                                          patience=finetuning_hp_config["train"]["patience_learning_rate"],
                                                          min_delta=0.001,
                                                          verbose=1,
                                                          min_lr=finetuning_hp_config["train"]["min_learning_rate"]),
                        tf.keras.callbacks.TensorBoard(TF_LOG_DIR)],
             )
