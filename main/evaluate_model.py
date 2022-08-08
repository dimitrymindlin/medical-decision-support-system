from dataloader.mura_wrist_dataset import MuraDataset
from utils.eval_metrics import log_and_pring_evaluation
import tensorflow_addons as tfa
import tensorflow as tf


def evaluate_model(config, clf_path):
    # clf_path = f"../checkpoints/2022-06-11--00.44/model"
    if not clf_path.contains("/model"):
        clf_path += "/model"
    metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
    model = tf.keras.models.load_model(clf_path, custom_objects={'f1_score': metric_f1})

    # Load data and class weights
    mura_data = MuraDataset(config)

    log_and_pring_evaluation(model, mura_data, config, None)
