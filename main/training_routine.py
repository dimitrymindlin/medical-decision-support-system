from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from models.finetuning_model import get_finetuning_model_from_pretrained_model
from models.mura_model import WristPredictNet
from dataloader.mura_wrist_dataset import MuraDataset, get_labels_from_tfds
from utils.eval_metrics import log_and_pring_evaluation


def train_model(config, print_console=True):
    # Get Settings and set names and paths
    TRAIN_MODE = config["train"]["prefix"]  # one of: [pretrain, finetune, frozen]
    MODEL_NAME = config["model"]["name"]
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d--%H.%M")
    TF_LOG_DIR = f'tensorboard_logs/logs_{TRAIN_MODE}/{TRAIN_MODE}_{MODEL_NAME}/' + TIMESTAMP + "/"
    checkpoint_path_name = f'checkpoints/{TRAIN_MODE}_{MODEL_NAME}/' + TIMESTAMP + '/cp.ckpt'
    checkpoint_path = f'checkpoints/{TRAIN_MODE}_{MODEL_NAME}/' + TIMESTAMP + '/'
    file_writer = tf.summary.create_file_writer(TF_LOG_DIR)
    if TRAIN_MODE in ["finetuning", "frozen"]:
        ckp_stage = config["train"]["checkpoint_stage"]
        ckp_name = config['train']['checkpoint_name']
        PRETRAINED_CKP_PATH = f"checkpoints/{ckp_stage}_{MODEL_NAME}/{ckp_name}/cp.ckpt"

    # Tensorboard config matrix
    config_matrix = [[k, str(w)] for k, w in config["train"].items()]
    with file_writer.as_default():
        tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

    # Load data and class weights
    mura_data = MuraDataset(config)
    y_integers = get_labels_from_tfds(mura_data.A_B_dataset)
    if config["train"]["use_class_weights"]:
        class_weights = compute_class_weight(class_weight="balanced",
                                             classes=np.unique(y_integers),
                                             y=y_integers)
        d_class_weights = dict(zip(np.unique(y_integers), class_weights))
    else:
        d_class_weights = None

    # Callbacks
    """def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
            
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler),"""

    my_callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_name,
                                        # Callback to save the Keras model or model weights at some frequency.
                                        monitor='val_accuracy',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_freq='epoch'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                          # Reduce learning rate when a metric has stopped improving.
                                          factor=config["train"]["factor_learning_rate"],
                                          patience=config["train"]["patience_learning_rate"],
                                          min_delta=0.001,
                                          verbose=1,
                                          min_lr=config["train"]["min_learning_rate"]),
        keras.callbacks.TensorBoard(log_dir=TF_LOG_DIR,
                                    histogram_freq=1,
                                    write_graph=True,
                                    write_images=False,
                                    update_freq='epoch',
                                    profile_batch=30,
                                    embeddings_freq=0,
                                    embeddings_metadata=None
                                    ),
        keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                      patience=config["train"]["early_stopping_patience"],
                                      mode="max",
                                      baseline=None,
                                      restore_best_weights=True,
                                      )
    ]

    # Load model and set train params and metrics
    if config["train"]["prefix"] in ["pretrain", "direct"]:
        model = WristPredictNet(config).model()
    elif config["train"]["prefix"] in ["frozen"]:
        pre_model = WristPredictNet(config).model()
        print(f"Loading pretrained from {config['train']['checkpoint_stage']} for {config['train']['prefix']}.")
        pre_model.load_weights(PRETRAINED_CKP_PATH)
        model = get_finetuning_model_from_pretrained_model(pre_model, config)
    else:
        pre_model = WristPredictNet(config).model()
        model = get_finetuning_model_from_pretrained_model(pre_model, config)
        print(f"Loading pretrained from {config['train']['checkpoint_stage']} for {config['train']['prefix']}.")
        model.load_weights(PRETRAINED_CKP_PATH)

    metric_auc = tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=len(config["data"]["class_names"]),
                                      from_logits=False)

    """metric_f1 = tfa.metrics.F1Score(num_classes=len(config["data"]["class_names"]),
                                    threshold=config["test"]["F1_threshold"], average='macro')"""

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["train"]["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=["accuracy", metric_auc])

    if config["train"]["freezing_layers"] > 0:
        for layer in model.layers[:config["train"]["freezing_layers"]]:
            layer.trainable = False

    # Model Training
    history = model.fit(mura_data.A_B_dataset,
                        epochs=config["train"]["epochs"],
                        verbose=1,
                        class_weight=d_class_weights,
                        validation_data=mura_data.A_B_dataset_val,
                        callbacks=my_callbacks)

    # Evaluation
    print("Train History")
    print(history)
    print(f"Test Evaluation for {TIMESTAMP}")
    log_and_pring_evaluation(model, mura_data, config, file_writer)

    # Save whole model
    model.save(checkpoint_path + 'model')
    return TIMESTAMP


def evaluate_model(config):
    classifier_folder = f"../checkpoints/2022-06-11--00.44/model"
    metric_f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='macro')
    model = tf.keras.models.load_model(classifier_folder, custom_objects={'f1_score': metric_f1})

    # Load data and class weights
    mura_data = MuraDataset(config)

    log_and_pring_evaluation(model, mura_data, config, None)
