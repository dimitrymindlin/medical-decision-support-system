frozen_hp_config = {
    "dataset": {
        "download": True,
    },
    "data": {
        "class_names": ["positive"],
        "input_size": (512, 512),
        "image_height": 512,
        "image_width": 512,
        "image_channel": 3,
    },
    "train": {
        "finetune": False,
        "config_name": "frozen_hp_config",
        "train_base": False,
        "augmentation": True,
        "use_class_weights": True,
        "batch_size": 8,
        "epochs": 5,
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
        "pooling": "max",
    }
}
