from configs.mura_pretraining_config import mura_config
from eda.data_analysis import get_num_of_samples
from mura_pretraining.dataloader.mura_dataset import MuraDataset
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf

from mura_pretraining.dataloader.mura_tfds import MuraImages
from mura_finetuning.dataloader.mura_finetuning_tfds import MuraFinetuningImages

mura_config["dataset"]["download"] = True

(train, validation, test), info = tfds.load(
            'MuraFinetuningImages',
            split=['train[:80%]', 'train[80%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            download=mura_config["dataset"]["download"],
            with_info=True,
        )

fig = tfds.visualization.show_examples(train, info)
plt.show()
exit()
dataset = MuraDataset(mura_config)

#dataset.benchmark()

get_num_of_samples(dataset)
