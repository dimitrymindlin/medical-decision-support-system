from keras.callbacks import TensorBoard
from tensorboard.plugins.pr_curve import summary as pr_summary
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

from dataloader.mura_wrist_dataset import MuraDataset
from utils.training_utils import get_labels_from_tfds


def log_confusion_matrix(dataset, model):
    # Use the model to predict the values from the validation dataset.
    print("CM, Starting")
    datasets = [dataset.ds_val, dataset.ds_test]
    ds_names = ["Validation", "Test"]
    for ds_name, ds in zip(ds_names, datasets):
        pred = model.predict(ds)
        pred = np.concatenate(np.where(pred > 0.5, 1, 0))
        labels = np.concatenate([y for x, y in ds], axis=0)
        con_mat = tf.math.confusion_matrix(labels=labels, predictions=pred).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        print(f"{ds_name}")
        print(con_mat)
        print("__")
        print(con_mat_norm)


def log_sklearn_consufions_matrix(dataset, model):
    from sklearn.metrics import confusion_matrix
    print("SKLEARN CM, Starting")
    datasets = [dataset.ds_val, dataset.ds_test]
    ds_names = ["Validation", "Test"]
    for ds_name, ds in zip(ds_names, datasets):
        pred = model.predict(ds)
        pred = np.concatenate(np.where(pred > 0.5, 1, 0))
        labels = np.concatenate([y for x, y in ds], axis=0)
        con_mat = confusion_matrix(y_true=labels, y_pred=pred)
        print(f"{ds_name}")
        print(con_mat)
        print(classification_report(y_true=labels, y_pred=pred))


def log_kappa(dataset, model):
    m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=False)
    y_pred = model.predict(dataset.ds_test)
    y_pred = np.concatenate(np.where(y_pred > 0.5, 1, 0))
    labels = np.concatenate([y for x, y in dataset.ds_test], axis=0)

    print(y_pred.shape, labels.shape)
    m.update_state(y_true=labels, y_pred=y_pred)
    print('Final Kappa result: ', m.result().numpy())


class PRTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        # One extra argument to indicate whether or not to use the PR curve summary.
        self.pr_curve = kwargs.pop('pr_curve', True)
        super(PRTensorBoard, self).__init__(*args, **kwargs)

    def set_model(self, model):
        super(PRTensorBoard, self).set_model(model)

        if self.pr_curve:
            # Get the prediction and label tensor placeholders.
            predictions = self.model._feed_outputs[0]
            labels = tf.cast(self.model._feed_targets[0], tf.bool)
            # Create the PR summary OP.
            self.pr_summary = pr_summary.op(name='pr_curve',
                                            predictions=predictions,
                                            labels=labels,
                                            display_name='Precision-Recall Curve')

    def on_epoch_end(self, epoch, logs=None):
        super(PRTensorBoard, self).on_epoch_end(epoch, logs)

        if self.pr_curve and self.validation_data:
            # Get the tensors again.
            tensors = self.model._feed_targets + self.model._feed_outputs
            # Predict the output.
            predictions = self.model.predict(self.validation_data[:-2])
            # Build the dictionary mapping the tensor to the data.
            val_data = [self.validation_data[-2], predictions]
            feed_dict = dict(zip(tensors, val_data))
            # Run and add summary.
            result = self.sess.run([self.pr_summary], feed_dict=feed_dict)
            self.writer.add_summary(result[0], epoch)
        self.writer.flush()


def log_and_print_evaluation(model, data, config, file_writer=None):
    result = model.evaluate(data.A_B_dataset_test)
    result = dict(zip(model.metrics_names, result))
    result_matrix = [[k, str(w)] for k, w in result.items()]
    print(result_matrix)

    m = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=True)
    y_pred = model.predict(data.A_B_dataset_test)

    y_predicted = np.argmax(y_pred, axis=1)
    y_true = get_labels_from_tfds(data.A_B_dataset_test)
    print(classification_report(y_true, y_predicted))
    print(y_predicted.shape, y_true.shape)
    print(confusion_matrix(y_true, y_predicted))
    m.update_state(y_true, y_predicted)
    sk_learn_kapa = cohen_kappa_score(y_true, y_predicted)
    print('TFA Kappa score result: ', m.result().numpy())
    print('SKLEARN Kappa score result: ', sk_learn_kapa)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_predicted)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    result_matrix.append(["TFA Kappa score", str(m.result().numpy())])
    result_matrix.append(["SKLEARN Kappa score", str(sk_learn_kapa)])
    result_matrix.append(["TN", str(tn)])
    result_matrix.append(["FP", str(fp)])
    result_matrix.append(["FN", str(fn)])
    result_matrix.append(["TP", str(tp)])
    result_matrix.append(["Precision", str(precision)])
    result_matrix.append(["Recall", str(recall)])
    result_matrix.append(["F1", str(f1_score)])
    if file_writer:
        with file_writer.as_default():
            tf.summary.text(f"{config['model']['name']}_evaluation", tf.convert_to_tensor(result_matrix), step=0)

    print("Result matrix")
    print(result_matrix)

    print("Classification Report")

    print(classification_report(y_true, y_predicted))

    """print("ON TRAIN SET")
    y_pred = model.predict(data.A_B_dataset)

    yp3 = np.argmax(y_pred, axis=1)
    y_true3 = data.train_y

    cm2 = confusion_matrix(y_true3, yp3)
    print(cm2)

    print(classification_report(y_true3, yp3))"""
