wrist_xray_config = {
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
        "train_base": False,
        "augmentation": False,
        "batch_size": 8,
        "learn_rate_finetuning": 0.0001,
        "learn_rate_final_layers": 0.01,
        "epochs": 60,
        "patience_learning_rate": 1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 8,
        "use_class_weights": False,
        "use_mura_weights": True
    },
    "test": {
        "batch_size": 16,
        "F1_threshold": 0.5,
    },
    "model": {
        "pretrained": True,
        "pooling": "avg",
        "name": "densenet"
    }
}
