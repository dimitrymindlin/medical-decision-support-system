direct_training_config = {
    "dataset": {
        "download": True,
        "transformed": True
    },
    "data": {
        "class_names": ["negative", "positive"],
        "input_size": (512, 512),
        "image_height": 512,
        "image_width": 512,
        "image_channel": 3,
        "input_shape": (521, 521, 3)
    },
    "train": {
        "body_parts": ["XR_WRIST"],
        "prefix": "direct",
        "train_base": True,
        "augmentation": True,
        "use_class_weights": True,
        "batch_size": 8,
        "epochs": 60,
        "learning_rate": 0.0001,
        "patience_learning_rate": 2,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-7,
        "early_stopping_patience": 5,
        "additional_last_layers": 1,
        "weight_regularisation": 0.2,
        "dropout_value": 0.6,
        "freezing_layers": 249
    },
    "test": {
        "batch_size": 8,
        "F1_threshold": 0.5,
    },
    "model": {
        "name": "inception",
        "pretrained": True,
        "pooling": "avg",
    }
}
