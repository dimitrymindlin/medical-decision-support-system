from configs.wrist_xray_config import xray_wrist_config
import tensorflow_datasets as tfds

from writs_xray_finetuning.dataloader.wrist_xray_dataset import WristXrayDataset
from writs_xray_finetuning.dataloader.wrist_xray_tfds import WristXrayImages

xray_wrist_config["dataset"]["download"] = True

"""(train, validation, test), info = tfds.load(
            'XrayWristImages',
            split=['train[:70%]', 'train[70%:90%]', 'train[:10%]'],
            shuffle_files=True,
            as_supervised=True,
            download=xray_wrist_config["dataset"]["download"],
            with_info=True,
        )

fig = tfds.visualization.show_examples(train, info)"""

dataset = WristXrayDataset(xray_wrist_config)

#dataset.benchmark()