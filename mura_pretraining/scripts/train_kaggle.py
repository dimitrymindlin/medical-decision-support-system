from datetime import datetime

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# Get Dataset
from sklearn.metrics import confusion_matrix, classification_report
from configs.direct_training_config import direct_training_config as config
from mura_pretraining.dataloader import MuraDataset
from utils.eval_metrics import log_confusion_matrix, log_kappa, log_sklearn_consufions_matrix

model_name = "inception"
timestamp = datetime.now().strftime("%Y-%m-%d--%H.%M")
TF_LOG_DIR = f'kaggle/kaggle_{model_name}/' + timestamp + "/"
checkpoint_filepath = f'checkpoints/kaggle_{model_name}/' + timestamp + '/cp.ckpt'

dataset = MuraDataset(config, only_wrist_data=True)

for index, example in enumerate(dataset.ds_test):
    image_raw, label_raw = example[0].numpy(), example[1].numpy()
    image, label = dataset.preprocess(image_raw, label_raw)
    print()

##### Another model try: ########
base_model = keras.applications.InceptionV3(
    #     weights='imagenet',  # Load weights pre-trained on ImageNet.,
    input_shape=(224, 224, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top


# odl from original
# for layer in base_model.layers[:4]:
#  layer.trainable=False

# Freeze base model
# base_model.trainable = False

# Create a new model on top
input_image = keras.layers.Input((224, 224, 3))

# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(input_image)

# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)  ##### <-
# x=keras.layers.Flatten()(x)

x = keras.layers.Dense(1024)(x)  ###
x = keras.layers.Activation(activation='relu')(x)  ###
x = keras.layers.Dropout(0.5)(x)  ###
x = keras.layers.Dense(256)(x)
x = keras.layers.Activation(activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(2)(x)
out = keras.layers.Activation(activation='softmax')(x)

model = keras.Model(inputs=input_image, outputs=out)

metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                  from_logits=False)
metric_bin_accuracy = tf.keras.metrics.BinaryAccuracy()
kappa = tfa.metrics.CohenKappa(num_classes=2)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=["accuracy", metric_auc, metric_bin_accuracy, kappa])

# Tensorboard Callback and config logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR)

# Checkpoint Callback to only save best checkpoint
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=metric_auc.name,
    mode='max',
    save_best_only=False)

# Early Stopping if loss plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=config['train']['early_stopping_patience'])

# Dynamic Learning Rate
dyn_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=config['train']['factor_learning_rate'],
    patience=config['train']['patience_learning_rate'],
    mode="min",
    min_lr=config['train']['min_learning_rate'],
)

# Class weights for training underrepresented classes
class_weight = dataset.train_classweights if config["train"]["use_class_weights"] else None

# Log Text like config and evaluation
config_matrix = [[k, str(w)] for k, w in config["train"].items()]
file_writer = tf.summary.create_file_writer(TF_LOG_DIR)
with file_writer.as_default():
    tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

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
    tf.summary.text(f"mura_evaluation", tf.convert_to_tensor(result_matrix), step=0)
    #log_confusion_matrix(dataset, model)
    #log_kappa(dataset, model)
    #log_sklearn_consufions_matrix(dataset, model)

print("Kaggel Evaluation")
m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
labels = np.concatenate([y for x, y in dataset.ds_val], axis=0)
vy_data = keras.utils.to_categorical(labels)
y_pred = model.predict(dataset.ds_val)
yp2 = np.argmax(y_pred, axis=1)
ya2 = np.argmax(vy_data, axis=1)
print(y_pred.shape, vy_data.shape)
m.update_state(ya2, yp2)
print('Final result: ', m.result().numpy())
vy_data2 = np.argmax(vy_data, axis=1)
print(vy_data2.shape)
print(yp2.shape)
cm = confusion_matrix(vy_data2, yp2)
print(cm)
print(classification_report(vy_data2, yp2))
