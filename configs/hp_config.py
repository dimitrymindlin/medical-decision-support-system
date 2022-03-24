hp_config = {
    "dataset": {
        "download": True,
    },
    "data": {
        "class_names": ["negative", "positive"],
        "input_size": (512, 512),
        "image_height": 512,
        "image_width": 512,
        "image_channel": 3,
    },
    "train": {
        "prefix": "hp",
        "config_name": "hp",
        "body_parts": ["XR_WRIST"],
        "train_base": True,
        "augmentation": True,
        "use_class_weights": True,
        "batch_size": 16,
        "epochs": 60,
        "learning_rate": 0.0001,
        "patience_learning_rate": 3,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 5,
        "additional_last_layers": 4,
        "dropout_value": 0.5,
        "weight_regularisation": 0.01,
    },
    "test": {
        "batch_size": 16,
        "F1_threshold": 0.5,
    },
    "model": {
        "name": "inception",
        "pretrained": True,
        "pooling": "max",
    }
}
