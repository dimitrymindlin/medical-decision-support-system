finetuning_config = {
    "dataset": {
        "download": True,
    },
    "data": {
        "class_names": ["negative", "positive"],
        "input_size": (520, 520),
        "image_height": 520,
        "image_width": 520,
        "image_channel": 3,
    },
    "train": {
        "prefix": "finetuning",
        "config_name": "finetuning_config",
        "body_parts": ["XR_WRIST"],
        "checkpoint_stage": "frozen",
        "checkpoint_name": "2022-03-11--08.25",
        "train_base": False,
        "augmentation": True,
        "use_class_weights": True,
        "batch_size": 32,
        "epochs": 60,
        "learning_rate": 0.0001,
        "patience_learning_rate": 3,
        "factor_learning_rate": 0.1,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 5,
        "additional_last_layers": False
    },
    "test": {
        "batch_size": 32,
        "F1_threshold": 0.5,
    },
    "model": {
        "name": "inception",
        "pretrained": True,
        "pooling": "max",
    }
}
