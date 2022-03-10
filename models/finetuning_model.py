import tensorflow as tf
from keras.models import Model


def get_finetuning_model_from_pretrained_model(pre_model, train_base=False):
    if not train_base:
        # Freeze all the layers
        for layer in pre_model.layers[:]:
            layer.trainable = False
    x = tf.keras.layers.Dense(256, activation='relu')(pre_model.layers[-2].output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
    model = Model(pre_model.layers[0].input, output)
    return model


def get_finetuning_model_from_pretrained_model_hp(model, hp):
    x = model.layers[-1].output
    dropout_rate = hp.Choice('dropout_rate', [0.2, 0.3])
    if hp.Boolean("dropout"):
        x = tf.keras.layers.Dropout(dropout_rate)(x)  # Regularize with dropout
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model.layers[0].input, outputs=x)
    return model
