from configs.wrist_xray_config import wrist_xray_config
import tensorflow_datasets as tfds

from eda.data_analysis import get_num_of_samples
from wrist_xray_finetuning.dataloader.wrist_xray_dataset import WristXrayDataset
from wrist_xray_finetuning.dataloader.wrist_xray_tfds import WristXrayImages

wrist_xray_config["dataset"]["download"] = False

"""(train, validation, test), info = tfds.load(
            'WristXrayImages',
            split=['train[:70%]', 'train[70%:90%]', 'train[:10%]'],
            shuffle_files=True,
            as_supervised=True,
            download=wrist_xray_config["dataset"]["download"],
            with_info=True,
        )

fig = tfds.visualization.show_examples(train, info)"""

dataset = WristXrayDataset(wrist_xray_config)

dataset.benchmark()

get_num_of_samples(dataset)