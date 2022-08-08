import sys

from configs.direct_training_config import direct_training_config
from configs.finetuning_config import finetuning_config
from configs.frozen_config import frozen_config
from configs.pretraining_config import pretraining_config
from dataloader.mura_wrist_dataset import MuraDataset
from models.mura_model import WristPredictNet
from utils.eval_metrics import log_and_pring_evaluation
import tensorflow_addons as tfa
import tensorflow as tf

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

if len(tf.config.list_physical_devices('GPU')) == 0:
    TFDS_PATH = "/Users/dimitrymindlin/tensorflow_datasets"
else:
    TFDS_PATH = "../tensorflow_datasets"
config["data"]["tfds_path"] = TFDS_PATH


def evaluate_model(config, clf_path):
    # clf_path = f"../checkpoints/2022-06-11--00.44/model"
    if "/model" not in clf_path:
        clf_path += "/cp.ckpt"
    pre_model = WristPredictNet(config).model()
    model = pre_model.load_weights(clf_path)
    # metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
    # model = tf.keras.models.load_model(clf_path, custom_objects={'f1_score': metric_f1})

    # Load data and class weights
    mura_data = MuraDataset(config)

    log_and_pring_evaluation(model, mura_data, config, None)
    model.save(clf_path + 'model')


evaluate_model(config, clf_path)
