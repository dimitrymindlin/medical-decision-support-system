direct_training_config = {
    "dataset": {
        "download": True,
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
        "config_name": "direct_training_config",
        "train_base": True,
        "augmentation": True,
        "use_class_weights": True,
        "batch_size": 32,
        "epochs": 60,
        "learn_rate": 0.0001,
        "patience_learning_rate": 3,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 5
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
