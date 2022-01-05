def get_num_of_samples(dataset):
    try:
        print(f"Test examples: {dataset.ds_info.splits['test'].num_examples}")
        print(f"Train examples: {dataset.ds_info.splits['train'].num_examples}")
        print(f"Total examples: {dataset.ds_info.splits.total_num_examples}")
    except ValueError:
        # No test set provided
        print(f"Total examples: {dataset.ds_info.splits.total_num_examples}")
    print(f"Train class weights: {dataset.train_classweights}")

