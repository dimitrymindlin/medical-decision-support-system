import tensorflow as tf
from keras import regularizers
from keras.models import Model


def get_finetuning_model_from_pretrained_model(pre_model, config):
    if not config["train"]["train_base"]:
        # Freeze all the layers
        print("Freezing all layers...")
        for layer in pre_model.layers[:]:
            layer.trainable = False

    weight_regularisation = regularizers.l2(config["train"]["weight_regularisation"]) if config["train"][
        "weight_regularisation"] else None

    x = pre_model.layers[-2].output
    if config["train"]["additional_last_layers"]:
        for layer_count in range(config["train"]["additional_last_layers"]):
            print("Adding additional layers...")
            x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=weight_regularisation)(x)
            x = tf.keras.layers.Dropout(config["train"]["dropout_value"])(x)
    output = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
    model = Model(pre_model.layers[0].input, output)
    return model


def get_finetuning_model_from_pretrained_model_hp(pre_model, config, hp):
    weight_regularisation = regularizers.l2(hp.Choice("weight_regularisation", [0.01, 0.0005]))

    x = pre_model.layers[-2].output
    for layer_count in range(hp.Choice("additional_layers", [1, 3])):
        print("Adding additional layers...")
        x = tf.keras.layers.Dense(hp.Choice("neurons", [128, 64]), activation='relu', kernel_regularizer=weight_regularisation)(x)
        x = tf.keras.layers.Dropout(hp.Choice("dropout_value", [0.4, 0.6]))(x)
    output = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
    model = Model(pre_model.layers[0].input, output)
    return model
