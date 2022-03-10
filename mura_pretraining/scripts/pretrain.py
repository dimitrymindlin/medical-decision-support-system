import sys

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from configs.pretraining_config import pretraining_config as config
from models.mura_model import WristPredictNet
from mura_finetuning.dataloader.mura_generators import MuraGeneratorDataset
from utils.eval_metrics import log_and_pring_evaluation
from utils.path_constants import PathConstants
from utils.training_utils import get_model_name_from_cli_to_config

model_prefix = "pre_"
model_name = get_model_name_from_cli_to_config(sys.argv, config)
timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
TF_LOG_DIR = f'{PathConstants.PRETRAIN}/{model_prefix}{model_name}/' + timestamp + "/"
checkpoint_path_name = f'checkpoints/{model_prefix}{model_name}/' + timestamp + '/cp.ckpt'
checkpoint_path = f'checkpoints/{model_prefix}{model_name}/' + timestamp + '/'
file_writer = tf.summary.create_file_writer(TF_LOG_DIR)


mura_data = MuraGeneratorDataset(config)

y_integers = np.argmax(mura_data.y_data, axis=1)

class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_integers),
                                     y=y_integers)

d_class_weights = dict(zip(np.unique(y_integers), class_weights))

# Tensorboard Callback and config logging
my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_name,
                                    # Callback to save the Keras model or model weights at some frequency.
                                    monitor='val_accuracy',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='auto',
                                    save_freq='epoch'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                      # Reduce learning rate when a metric has stopped improving.
                                      factor=0.1,
                                      patience=3,
                                      min_delta=0.001,
                                      verbose=1,
                                      min_lr=0.000000001),
    keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR,
                                histogram_freq=1,
                                write_graph=True,
                                write_images=False,
                                update_freq='epoch',
                                profile_batch=30,
                                embeddings_freq=0,
                                embeddings_metadata=None
                                ),
    keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                  patience=5,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                                  )
]

model = WristPredictNet(config).model()

metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)

metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                threshold=config["test"]["F1_threshold"], average='macro')

model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["train"]["learning_rate"]),
              loss='categorical_crossentropy',
              metrics=["accuracy", metric_auc, metric_f1])

# Model Training
history = model.fit(mura_data.train_loader,
                              epochs=config["train"]["epochs"],
                              verbose=1,
                              class_weight=d_class_weights,
                              validation_data=mura_data.valid_loader,
                              callbacks=my_callbacks)

log_and_pring_evaluation(model, history, mura_data, config, timestamp, file_writer)

#Save whole model
model.save(checkpoint_path + 'model')
