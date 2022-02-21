direct_training_config = {
    "dataset": {
        "download": True,
    },
    "data": {
        "class_names": ["positive"],
        "input_size": (520, 520),
        "image_height": 520,
        "image_width": 520,
        "image_channel": 3,
    },
    "train": {
        "config_name": "direct_training_config",
        "train_base": True,
        "augmentation": True,
        "use_class_weights": False,
        "batch_size": 8,
        "epochs": 60,
        "learn_rate": 0.0001,
        "patience_learning_rate": 1,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 5
    },
    "test": {
        "batch_size": 8,
        "F1_threshold": 0.5,
    },
    "model": {
        "name": "densenet",
        "pretrained": True,
        "pooling": None,
    }
}
