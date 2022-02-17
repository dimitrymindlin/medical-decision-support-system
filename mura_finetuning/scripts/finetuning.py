#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime

from configs.mura_finetuning_config import mura_finetuning_config as config
from mura_finetuning.model.finetuning_model import get_finetuning_model_from_pretrained_model
from mura_pretraining.dataloader import MuraDataset
from mura_pretraining.model.mura_model import get_mura_model
from utils.path_constants import PathConstants
from utils.training_utils import print_running_on_gpu, get_model_name_from_cli_to_config
import sys

get_model_name_from_cli_to_config(sys.argv, config)
TF_LOG_DIR = None
for arg in sys.argv:
    if arg == "--finetune":
        config["train"]["finetune"] = True
        TF_LOG_DIR = f'{PathConstants.WRIST_FINETUNE}/' + datetime.now().strftime("%Y-%m-%d--%H.%M")

if not TF_LOG_DIR:
    TF_LOG_DIR = f'{PathConstants.WRIST_LAST_LAYERS}/' + datetime.now().strftime("%Y-%m-%d--%H.%M")

CPU_WEIGHT_PATH = f"../../checkpoints/wrist_xray/best/cp.ckpt"
GPU_WEIGHT_PATH = f"checkpoints/wrist_xray/best/cp.ckpt"

file_writer = tf.summary.create_file_writer(TF_LOG_DIR)
print_running_on_gpu(tf)

# Dataset Definition
dataset = MuraDataset(config, only_wrist_data=True)

# Model Definition
model = get_mura_model(config, include_top=False)
model.load_weights(GPU_WEIGHT_PATH).expect_partial()
model = get_finetuning_model_from_pretrained_model(model)

# Training Parameter
optimizer = tf.keras.optimizers.Adam(config["train"]["learn_rate_final_layers"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)
metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()
metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                threshold=config["test"]["F1_threshold"], average='macro')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy, metric_f1]
)

# Log Config
config_matrix = [[k, str(w)] for k, w in config["train"].items()]

with file_writer.as_default():
    tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

"""def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the test_images.
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in dataset.ds_test:  # use dataset.unbatch() with repeat
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis=0)
    print(correct_labels)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    print(predicted_labels)
    # Calculate the confusion matrix using sklearn.metrics
    cm = sklearn.metrics.confusion_matrix(correct_labels, predicted_labels)

    figure = plot_confusion_matrix_util(cm, ["Normal", "Abnormal"])
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    print("writing confusion matrix")
    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    print("Done")"""

# Tensorboard Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR, histogram_freq=1)
# cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
checkpoint_filepath = 'checkpoints/wrist_xray/' + datetime.now().strftime("%Y-%m-%d--%H.%M") + '/cp.ckpt'
# Save best only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=metric_auc.name,
    mode='max',
    save_best_only=True)

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
    tf.summary.text(f"{config['model']['name']}_evaluation", tf.convert_to_tensor(result_matrix), step=0)
