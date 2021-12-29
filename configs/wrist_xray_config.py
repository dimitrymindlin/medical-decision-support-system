xray_wrist_config = {
    "dataset": {
        "download": False,
    },
    "data": {
        "class_names": ["positive"],
        "input_size": (224, 224),
        "image_height": 224,
        "image_width": 224,
        "image_channel": 3,
    },
    "train": {
        "augmentation": False,
        "batch_size": 16,
        "learn_rate": 0.0001,
        "epochs": 100,
        "patience_learning_rate": 2,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 8,
        "use_class_weights": False
    },
    "test": {
        "batch_size": 16,
        "F1_threshold": 0.5,
    },
    "model": {
        "pretrained": True,
        "pooling": "avg",
        "feature_extract": True,
    }
}
