import numpy as np


def get_model_name_from_cli_to_config(sys_argv, config):
    # set cli arguments
    for arg in sys_argv:
        if arg == "--densenet":
            config["model"]["name"] = "densenet"
        elif arg == "--inception":
            config["model"]["name"] = "inception"
    return config["model"]["name"]


def print_running_on_gpu(tf):
    print(f"Tensorflow version: {tf.version.VERSION}")
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")


def get_labels_from_tfds(ds):
    labels = np.concatenate([y for x, y in ds], axis=0)  # categorical labels [[1,0],[0,1]]
    return np.argmax(labels, axis=1)
