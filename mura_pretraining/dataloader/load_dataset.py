from configs.mura_pretraining_config import mura_config
from eda.data_analysis import get_num_of_samples
from mura_pretraining.dataloader.mura_dataset import MuraDataset

mura_config["dataset"]["download"] = False

"""(train, validation, test), info = tfds.load(
            'MuraImages:1.0.1',
            split=['train[:80%]', 'train[80%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            download=mura_config["dataset"]["download"],
            with_info=True,
        )

fig = tfds.visualization.show_examples(train, info)"""

dataset = MuraDataset(mura_config)

#dataset.benchmark()

get_num_of_samples(dataset)
