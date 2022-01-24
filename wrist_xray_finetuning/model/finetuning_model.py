import tensorflow as tf


def get_finetuning_model_from_pretrained_model(model):
    x = tf.keras.layers.Dropout(0.2)(model.layers[-2].output)  # Regularize with dropout
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model.layers[-2].input, outputs=x)
    return model
