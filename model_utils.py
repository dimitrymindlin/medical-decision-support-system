import tensorflow as tf


def get_model_by_name(config, img_input, _input_shape, weights):
    if config["model"]["name"] == "densenet":
        preprocessing_layer = tf.keras.applications.densenet.preprocess_input
        base_model = tf.keras.applications.DenseNet169(include_top=False,
                                                       input_tensor=img_input,
                                                       input_shape=_input_shape,
                                                       weights=weights,
                                                       pooling=config['model']['pooling'],
                                                       classes=len(config['data']['class_names']))
    elif config["model"]["name"] == "vgg":
        preprocessing_layer = tf.keras.applications.vgg19.preprocess_input
        base_model = tf.keras.applications.VGG19(include_top=False,
                                                 input_tensor=img_input,
                                                 input_shape=_input_shape,
                                                 weights=weights,
                                                 pooling=config['model']['pooling'],
                                                 classes=len(config['data']['class_names']))
    elif config["model"]["name"] == "resnet":
        preprocessing_layer = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(include_top=False,
                                                    input_tensor=img_input,
                                                    input_shape=_input_shape,
                                                    weights=weights,
                                                    pooling=config['model']['pooling'],
                                                    classes=len(config['data']['class_names']))

    elif config["model"]["name"] == "inception":
        preprocessing_layer = tf.keras.applications.inception_v3.preprocess_input
        base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                       input_tensor=img_input,
                                                       input_shape=_input_shape,
                                                       weights=weights,
                                                       pooling=config['model']['pooling'],
                                                       classes=len(config['data']['class_names']))
    return base_model, preprocessing_layer
