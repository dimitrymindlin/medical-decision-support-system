import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def get_num_of_samples(dataset):
    try:
        print(f"Test examples: {dataset.ds_info.splits['test'].num_examples}")
        print(f"Train examples: {dataset.ds_info.splits['train'].num_examples}")
        print(f"Total examples: {dataset.ds_info.splits.total_num_examples}")
    except ValueError:
        # No test set provided
        print(f"Total examples: {dataset.ds_info.splits.total_num_examples}")
    print(f"Train class weights: {dataset.train_classweights}")


def get_train_label_distribution_plot(dataset):
    df = tfds.as_dataframe(dataset.ds_train, dataset.ds_info)
    df = df.astype({"label": int})
    ax = df.plot.hist()
    ax.locator_params(integer=True)
    plt.show()


def show_train_examples(dataset):
    fig = tfds.visualization.show_examples(dataset.ds_train, dataset.ds_info)
    plt.show()


