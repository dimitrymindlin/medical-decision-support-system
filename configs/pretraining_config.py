pretraining_config = {
    "dataset": {
        "download": True,
    },
    "data": {
        "class_names": ["negative", "positive"],
        "input_size": (512, 512),
        "input_shape": (512, 512, 3),
        "image_height": 512,
        "image_width": 512,
        "image_channel": 3,
    },
    "train": {
        "body_parts": ["XR_HAND", "XR_FINGER", "XR_FOREARM"],
        "prefix": "pretrain",
        "config_name": "pretraining_config",
        "train_base": True,
        "augmentation": True,
        "use_class_weights": True,
        "batch_size": 32,
        "epochs": 60,
        "learning_rate": 0.0001,
        "patience_learning_rate": 1,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 5,
        "additional_last_layers": False,
        "weight_regularisation": None
    },
    "test": {
        "batch_size": 32,
        "F1_threshold": 0.5,
    },
    "model": {
        "name": "inception",
        "pretrained": True,
        "pooling": "avg",
    }
}
