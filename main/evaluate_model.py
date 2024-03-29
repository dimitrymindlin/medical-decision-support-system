import sys

from configs.direct_training_config import direct_training_config
from configs.finetuning_config import finetuning_config
from configs.frozen_config import frozen_config
from configs.pretraining_config import pretraining_config
from dataloader.mura_wrist_dataset import MuraDataset
from utils.eval_metrics import log_and_print_evaluation
import tensorflow as tf
import tensorflow.keras as keras

for arg in sys.argv:
    if arg == "--pretrain":
        config = pretraining_config
        continue
    elif arg == "--frozen":  # Freeze base and train last layers
        config = frozen_config
        continue
    elif arg == "--finetuning":  # Finetune whole model with low lr
        config = finetuning_config
        continue
    elif arg == "--direct":
        config = direct_training_config
        continue
    if arg.startswith("--2022"):
        print(arg)
        clf_path = arg[2:]
        clf_path = "checkpoints/direct_densenet/" + clf_path  # TODO: Make not fixed

# TFDS_PATH = "../tensorflow_datasets"
TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets"
clf_path = "../checkpoints/2022-03-24--12.42/model"  # Inception
# clf_path = "../checkpoints/2022-08-10--03.04/model" # DenseNet
config = direct_training_config
config["data"]["tfds_path"] = TFDS_PATH
config["model"]["name"] = "inception"


def evaluate_model(config, clf_path):
    # clf_path = f"../checkpoints/2022-06-11--00.44/model"
    if "/model" not in clf_path:
        clf_path += "/model"

    # metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
    metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                      from_logits=False)
    model = tf.keras.models.load_model(clf_path)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["train"]["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=["accuracy", metric_auc])
    # Load data and class weights
    mura_data = MuraDataset(config)
    # mura_data = MuraGeneratorDataset(config)

    log_and_print_evaluation(model, mura_data, config, None)


evaluate_model(config, clf_path)
