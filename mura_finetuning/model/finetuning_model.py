import tensorflow as tf


def get_finetuning_model_from_pretrained_model(model):
    x = tf.keras.layers.Dropout(0.2)(model.layers[-1].output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model.layers[0], outputs=x)
    return model


def get_finetuning_model_from_pretrained_model_hp(model, hp):
    """x = model.layers[-1].output
    if hp.Boolean("dropout"):"""
    x = tf.keras.layers.Dropout(0.2)(model.layers[-1].output)  # Regularize with dropout
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model.layers[0].input, outputs=x)
    return model