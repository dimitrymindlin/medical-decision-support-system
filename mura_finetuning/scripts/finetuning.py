#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime

from configs.finetuning_config import finetuning_config as config
from models.finetuning_model import get_finetuning_model_from_pretrained_model
from mura_pretraining.dataloader import MuraDataset
from models.mura_model import get_mura_model
from utils.eval_metrics import log_confusion_matrix, log_kappa, log_sklearn_consufions_matrix
from utils.path_constants import PathConstants
from utils.training_utils import print_running_on_gpu, get_model_name_from_cli_to_config
import sys

timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
model_name = get_model_name_from_cli_to_config(sys.argv, config)

for arg in sys.argv:  # Train whole network with low lr
    if arg == "--finetune":
        config["train"]["finetune"] = True

if config["train"]["finetune"]:
    config["train"]["train_base"] = True
    TF_LOG_DIR = f'{PathConstants.FINETUNE}/finetune_{model_name}/' + timestamp
    GPU_WEIGHT_PATH = f"checkpoints/frozen_{model_name}/best/cp.ckpt"
    checkpoint_filepath = f'checkpoints/finetune_{model_name}/' + timestamp + '/cp.ckpt'
else:
    # Train only last layers
    TF_LOG_DIR = f'{PathConstants.FROZEN}/frozen_{model_name}/' + timestamp
    GPU_WEIGHT_PATH = f"checkpoints/pre_{model_name}/best/cp.ckpt"  # for cpu prepend "../../"
    checkpoint_filepath = f'checkpoints/frozen_{model_name}/' + timestamp + '/cp.ckpt'

file_writer = tf.summary.create_file_writer(TF_LOG_DIR)
print_running_on_gpu(tf)

# Dataset Definition
dataset = MuraDataset(config, only_wrist_data=True)

# Model Definition
model = get_mura_model(config, include_top=False)
model.load_weights(GPU_WEIGHT_PATH).expect_partial()
model = get_finetuning_model_from_pretrained_model(model)

# Training Parameter
if config["train"]["finetune"]:
    optimizer = tf.keras.optimizers.Adam(config["train"]["learn_rate_finetuning"])
else:
    optimizer = tf.keras.optimizers.Adam(config["train"]["learn_rate_final_layers"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)
metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()
metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                threshold=config["test"]["F1_threshold"], average='macro')
kappa = tfa.metrics.CohenKappa(num_classes=2)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy, metric_f1, kappa]
)

# Log Config
config_matrix = [[k, str(w)] for k, w in config["train"].items()]

with file_writer.as_default():
    tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

# Tensorboard Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR)

# Save best only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=metric_auc.name,
    mode='max',
    save_best_only=True)

# Early Stopping if loss plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=metric_auc.name,
    min_delta=0,
    patience=config['train']['early_stopping_patience'])

# Dynamic Learning Rate
dyn_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=metric_auc.name,
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
    tf.summary.text(f"{config['model']['name']}_evaluation", tf.convert_to_tensor(result_matrix), step=0)
    log_confusion_matrix(dataset, model)
    log_kappa(dataset, model)
    log_sklearn_consufions_matrix(dataset, model)
