from keras.callbacks import TensorBoard
from sklearn.metrics import cohen_kappa_score
from tensorboard.plugins.pr_curve import summary as pr_summary
import tensorflow as tf
import numpy as np


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
        print("Kappa")
        print(cohen_kappa_score(labels, pred))
        print("_________")


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
