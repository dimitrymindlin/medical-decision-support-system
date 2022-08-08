import sys

from configs.direct_training_config import direct_training_config
from configs.finetuning_config import finetuning_config
from configs.frozen_config import frozen_config
from configs.pretraining_config import pretraining_config
from dataloader.mura_wrist_dataset import MuraDataset
from utils.eval_metrics import log_and_pring_evaluation
import tensorflow_addons as tfa
import tensorflow as tf

for arg in sys.argv:
    if arg == "--pretrain":
        config = pretraining_config
        break
    elif arg == "--frozen":  # Freeze base and train last layers
        config = frozen_config
        break
    elif arg == "--finetuning":  # Finetune whole model with low lr
        config = finetuning_config
        break
    elif arg == "--direct":
        config = direct_training_config
        break
    if arg.startswith("--2022"):
        clf_path = arg


def evaluate_model(config, clf_path):
    # clf_path = f"../checkpoints/2022-06-11--00.44/model"
    if not clf_path.contains("/model"):
        clf_path += "/model"
    metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
    model = tf.keras.models.load_model(clf_path, custom_objects={'f1_score': metric_f1})

    # Load data and class weights
    mura_data = MuraDataset(config)

    log_and_pring_evaluation(model, mura_data, config, None)


evaluate_model(config, clf_path)
