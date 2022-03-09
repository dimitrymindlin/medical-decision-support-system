from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from configs.direct_training_config import direct_training_config as config
from models.mura_model import WristPredictNet, get_working_mura_model
from mura_finetuning.dataloader.mura_generators import MuraGeneratorDataset

model_name = config["model"]["name"]
timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
TF_LOG_DIR = f'kaggle/kaggle_new_{model_name}/' + timestamp + "/"
checkpoint_dir = f'checkpoints/kaggle_new_{model_name}/' + timestamp + '/cp.ckpt'


mura_data = MuraGeneratorDataset(config)

y_integers = np.argmax(mura_data.y_data, axis=1)

class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_integers),
                                     y=y_integers
                                     )
d_class_weights = dict(zip(np.unique(y_integers), class_weights))

# Tensorboard Callback and config logging
my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
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
#model = get_working_mura_model()
#print(model.summary())


metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)

metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                threshold=config["test"]["F1_threshold"], average='macro')


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=["accuracy", metric_auc, metric_f1])
# Model Training
# model.load_weights("checkpoints/kaggle_inception/2022-02-21--15.47/cp.ckpt")
history = model.fit(mura_data.train_loader,
                              epochs=40,
                              verbose=1,
                              class_weight=d_class_weights,  # d_class_weights
                              validation_data=mura_data.valid_loader,
                              callbacks=my_callbacks)

print("Train History")
print(history)
print("Kaggel Test Evaluation")
result = model.evaluate(mura_data.valid_loader)
for metric, value in zip(model.metrics_names, result):
    print(metric, ": ", value)

m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
y_pred = model.predict(mura_data.valid_loader)

yp2 = np.argmax(y_pred, axis=1)
ya2 = np.argmax(mura_data.y_data_valid, axis=1)
print(y_pred.shape, mura_data.y_data_valid.shape)
m.update_state(ya2, yp2)
print('Kappa score result: ', m.result().numpy())

vy_data2 = np.argmax(mura_data.y_data_valid, axis=1)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(vy_data2, yp2)
print(cm)

print(classification_report(vy_data2, yp2))

y_pred = model.predict(mura_data.train_loader)

yp3 = np.argmax(y_pred, axis=1)
y_true3 = np.argmax(mura_data.y_data, axis=1)

cm2 = confusion_matrix(y_true3, yp3)
print(cm2)

print(classification_report(y_true3, yp3))

#Save whole model
model.save(checkpoint_dir + 'model')
