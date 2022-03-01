from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from configs.direct_training_config import direct_training_config as config
from models.mura_model import WristPredictNet
from mura_finetuning.dataloader.mura_generators import MuraGeneratorDataset
from utils.model_utils import get_input_shape_from_config

model_name = "inception"
timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
TF_LOG_DIR = f'kaggle/kaggle_new_{model_name}/' + timestamp + "/"
checkpoint_filepath = f'checkpoints/kaggle_new_{model_name}/' + timestamp + '/cp.ckpt'


"""mura_data = MuraGeneratorDataset()

y_integers = np.argmax(mura_data.y_data, axis=1)

class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_integers),
                                     y=y_integers
                                     )
d_class_weights = dict(zip(np.unique(y_integers), class_weights))"""

# Tensorboard Callback and config logging
my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
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
    keras.callbacks.TensorBoard(log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
                                histogram_freq=1,
                                write_graph=True,
                                write_images=False,
                                update_freq='epoch',
                                profile_batch=30,
                                embeddings_freq=1,
                                embeddings_metadata=None
                                ),
    keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                  patience=5,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                                  )
]

#model = get_working_mura_model()
model = WristPredictNet(config).model()

metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)

metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                threshold=config["test"]["F1_threshold"], average='macro')
kappa = tfa.metrics.CohenKappa(num_classes=2)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=["accuracy", metric_auc, metric_f1, kappa])

def find_target_layer(model):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except AttributeError:
            print("Output ...")
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

#name = find_target_layer(model)
base_model = model.get_layer(index=0)
print(base_model.summary())
grad_model = tf.keras.models.Model(
        [base_model.get_layer(index=0)], [base_model.get_layer(name="conv5_block16_2_conv").output, model.output]
    )
print(model.summary())

quit()
# Model Training
# model.load_weights("checkpoints/kaggle_inception/2022-02-21--15.47/cp.ckpt")
history = model.fit_generator(generator=mura_data.train_loader,
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
ya2 = np.argmax(mura_data.y_data, axis=1)
print(y_pred.shape, mura_data.y_data_valid.shape)
m.update_state(ya2, yp2)
print('Final result: ', m.result().numpy())

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
